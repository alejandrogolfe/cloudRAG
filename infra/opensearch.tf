# ── OpenSearch Serverless ─────────────────────────────────────────────────────
#
# OpenSearch Serverless requires THREE independent policy types.
# This is the most common point of confusion — all three must exist and be
# consistent, or access will silently fail with 403s.
#
#  1. Encryption policy  → which KMS key encrypts the collection data
#  2. Network policy     → whether the collection is public or private (VPC)
#  3. Data access policy → which IAM principals can perform index operations
#
# Note: "network public" here means the collection endpoint is reachable from
# the internet via HTTPS + SigV4. It does NOT mean unauthenticated access.
# Every request must still be signed with valid AWS credentials.
# For production with stricter requirements you would switch to VPC access.

locals {
  collection_name = "cloudrag-${var.environment}"
}

# ── 1. Encryption policy ──────────────────────────────────────────────────────
resource "aws_opensearchserverless_security_policy" "encryption" {
  name        = "${local.collection_name}-enc"
  type        = "encryption"
  description = "AWS-managed KMS key for ${local.collection_name}"

  policy = jsonencode({
    Rules = [
      {
        ResourceType = "collection"
        Resource     = ["collection/${local.collection_name}"]
      }
    ]
    AWSOwnedKey = true
  })
}

# ── 2. Network policy ─────────────────────────────────────────────────────────
resource "aws_opensearchserverless_security_policy" "network" {
  name        = "${local.collection_name}-net"
  type        = "network"
  description = "Public HTTPS access for ${local.collection_name}"

  # AllowFromPublic = true allows access from any IP over HTTPS.
  # Requests still require valid SigV4 signatures — this is NOT open access.
  # For stricter prod environments, replace with a VPC endpoint source.
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "collection"
          Resource     = ["collection/${local.collection_name}"]
        },
        {
          ResourceType = "dashboard"
          Resource     = ["collection/${local.collection_name}"]
        }
      ]
      AllowFromPublic = true
    }
  ])
}

# ── 3. The collection ─────────────────────────────────────────────────────────
resource "aws_opensearchserverless_collection" "main" {
  name        = local.collection_name
  type        = "VECTORSEARCH"
  description = "cloudRAG vector store - ${var.environment}"

  # The collection cannot be created before its encryption policy exists.
  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
  ]
}

# ── IAM role for the application ──────────────────────────────────────────────
# In production this role is assumed by ECS tasks.
# In development you can assume it locally with:
#   aws sts assume-role --role-arn <role_arn> --role-session-name local-dev
# Or, more conveniently, just add your personal ARN to admin_iam_principal_arn.

resource "aws_iam_role" "app" {
  name        = "cloudrag-app-${var.environment}"
  description = "Runtime role for cloudRAG - assumed by ECS (prod) or locally (dev)"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ecs-tasks.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "app_opensearch" {
  name = "opensearch-access"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["aoss:APIAccessAll"]
        Resource = aws_opensearchserverless_collection.main.arn
      }
    ]
  })
}

# ── 4. Data access policy ─────────────────────────────────────────────────────
# This is separate from IAM — OpenSearch Serverless has its own authorization
# layer on top of IAM. Even if an IAM policy allows aoss:APIAccessAll, you
# still need a data access policy that explicitly grants index-level permissions.
#
# We grant access to two principals:
#   - The app IAM role (ECS in prod, assume-role in dev)
#   - Your personal IAM principal (for running run_upload.py directly from local)

resource "aws_opensearchserverless_access_policy" "data" {
  name        = "${local.collection_name}-data"
  type        = "data"
  description = "Index-level access for app role and admin"

  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "index"
          Resource     = ["index/${local.collection_name}/*"]
          Permission = [
            "aoss:CreateIndex",
            "aoss:DeleteIndex",
            "aoss:UpdateIndex",
            "aoss:DescribeIndex",
            "aoss:ReadDocument",
            "aoss:WriteDocument",
          ]
        },
        {
          ResourceType = "collection"
          Resource     = ["collection/${local.collection_name}"]
          Permission   = ["aoss:DescribeCollectionItems"]
        }
      ]
      Principal = [
        aws_iam_role.app.arn,
        var.admin_iam_principal_arn,
      ]
    }
  ])
}
