terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state for now.
  # When ready for production: replace this block with an S3 backend.
  # See: https://developer.hashicorp.com/terraform/language/backend/s3
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project     = "cloudRAG"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
