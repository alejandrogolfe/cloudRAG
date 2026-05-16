output "opensearch_endpoint" {
  description = "OpenSearch Serverless collection endpoint — use this as OPENSEARCH_ENDPOINT in your .env"
  value       = aws_opensearchserverless_collection.main.collection_endpoint
}

output "opensearch_collection_arn" {
  description = "ARN of the OpenSearch Serverless collection"
  value       = aws_opensearchserverless_collection.main.arn
}

output "app_role_arn" {
  description = "ARN of the application IAM role (used by ECS tasks)"
  value       = aws_iam_role.app.arn
}

output "ecr_repository_url" {
  description = "ECR repository URL — used by GitHub Actions to push Docker images"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name — used by GitHub Actions to trigger deploys"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name — used by GitHub Actions to trigger deploys"
  value       = aws_ecs_service.app.name
}

output "alb_url" {
  description = "Public URL of the load balancer — use this to call the API"
  value       = "http://${aws_lb.main.dns_name}"
}
