# ── ECR Repository ────────────────────────────────────────────────────────────
# Stores the Docker image for the online pipeline.
# GitHub Actions builds and pushes to this registry on every merge to main.

resource "aws_ecr_repository" "app" {
  name                 = "cloudrag-${var.environment}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true  # allows destroy even when images exist

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the last 5 images to avoid storage costs
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = { type = "expire" }
      }
    ]
  })
}
