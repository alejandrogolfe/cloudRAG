output "opensearch_endpoint" {
  description = "OpenSearch Serverless collection endpoint — use this as OPENSEARCH_ENDPOINT in your .env"
  value       = aws_opensearchserverless_collection.main.collection_endpoint
}

output "opensearch_collection_arn" {
  description = "ARN of the OpenSearch Serverless collection"
  value       = aws_opensearchserverless_collection.main.arn
}

output "app_role_arn" {
  description = "ARN of the application IAM role (used by ECS in prod)"
  value       = aws_iam_role.app.arn
}
