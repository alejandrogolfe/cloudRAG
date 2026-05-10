variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-1"
}

variable "aws_profile" {
  description = "AWS CLI profile to use. Leave empty to use the default credential chain."
  type        = string
  default     = ""
}

variable "environment" {
  description = "Deployment environment: dev or prod"
  type        = string
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be 'dev' or 'prod'."
  }
}

variable "opensearch_index" {
  description = "Name of the OpenSearch index that will hold document chunks"
  type        = string
  default     = "cloudrag-docs"
}

# ── Who can access OpenSearch ─────────────────────────────────────────────────
# The data access policy below grants full index permissions to two principals:
#   1. The IAM role created by this module (used by ECS in prod, or assumed
#      locally via `aws sts assume-role` during development).
#   2. Your personal IAM user/role, so you can call run_upload.py from local
#      without assuming a separate role.
#
# Find your ARN with: aws sts get-caller-identity --query Arn --output text

variable "admin_iam_principal_arn" {
  description = "ARN of your personal IAM user or role for local admin access to OpenSearch (e.g. arn:aws:iam::123456789:user/yourname)"
  type        = string
}
