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

variable "admin_iam_principal_arn" {
  description = "ARN of your personal IAM user or role for local admin access to OpenSearch"
  type        = string
}

# ── ECS variables ─────────────────────────────────────────────────────────────

variable "task_cpu" {
  description = "CPU units for the ECS task (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memory in MB for the ECS task"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of ECS task instances to run"
  type        = number
  default     = 1
}
