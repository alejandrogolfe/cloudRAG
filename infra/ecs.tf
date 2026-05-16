# ── ECS Cluster ───────────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "cloudrag-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ── CloudWatch Log Group ───────────────────────────────────────────────────────
# Container logs go here — visible in AWS Console → CloudWatch → Log Groups

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/cloudrag-${var.environment}"
  retention_in_days = 7  # keep 7 days of logs in dev to save costs
}

# ── IAM Role for ECS Task Execution ───────────────────────────────────────────
# This role is used by ECS to:
#   - Pull the Docker image from ECR
#   - Read secrets from Secrets Manager
#   - Write logs to CloudWatch
# Different from the app role (which is used by the running container).

resource "aws_iam_role" "ecs_execution" {
  name        = "cloudrag-ecs-execution-${var.environment}"
  description = "ECS task execution role - pulls images and reads secrets"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow the execution role to read secrets from Secrets Manager
resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "read-cloudrag-secrets"
  role = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
      ]
      Resource = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/*"
    }]
  })
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# ── ECS Task Definition ────────────────────────────────────────────────────────
# Defines the container spec: image, CPU, memory, environment variables.
# Each deploy creates a new task definition revision — ECS keeps the history.

resource "aws_ecs_task_definition" "app" {
  family                   = "cloudrag-${var.environment}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.app.arn  # app role from opensearch.tf

  container_definitions = jsonencode([{
    name  = "cloudrag"
    image = "${aws_ecr_repository.app.repository_url}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    # Secrets are injected as environment variables from Secrets Manager.
    # ECS fetches the values at container startup — the app code doesn't change.
    secrets = [
      { name = "OPENAI_API_KEY",         valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/OPENAI_API_KEY" },
      { name = "LANGCHAIN_API_KEY",      valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/LANGCHAIN_API_KEY" },
      { name = "LANGCHAIN_ENDPOINT",     valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/LANGCHAIN_ENDPOINT" },
      { name = "LANGCHAIN_PROJECT",      valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/LANGCHAIN_PROJECT" },
      { name = "LANGCHAIN_TRACING_V2",   valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/LANGCHAIN_TRACING_V2" },
      { name = "OPENSEARCH_ENDPOINT",    valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/OPENSEARCH_ENDPOINT" },
      { name = "OPENSEARCH_INDEX",       valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/OPENSEARCH_INDEX" },
      { name = "AWS_REGION",             valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:cloudrag/AWS_REGION" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    # Health check — ECS restarts the container if /health stops responding
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

# ── ECS Service ────────────────────────────────────────────────────────────────
# Keeps the desired number of task instances running and registers them
# with the ALB target group for load balancing.

resource "aws_ecs_service" "app" {
  name            = "cloudrag-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true  # needed for Fargate to pull ECR images without NAT gateway
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "cloudrag"
    container_port   = 8000
  }

  # Allow rolling deploys without downtime:
  # new tasks start before old ones stop
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [
    aws_lb_listener.http,
    aws_iam_role_policy_attachment.ecs_execution_policy,
  ]

  # Ignore task_definition changes from Terraform after initial deploy —
  # GitHub Actions manages the image tag and task definition updates.
  lifecycle {
    ignore_changes = [task_definition]
  }
}
