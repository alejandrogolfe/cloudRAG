# ── Cost-tracker Lambda ────────────────────────────────────────────────────────
#
# Reads yesterday's infrastructure cost from AWS Cost Explorer (grouped by
# service, filtered by Project=cloudRAG tag) and publishes InfraCostDaily
# metrics to CloudWatch once a day at 06:00 UTC.
#
# CI/CD (deploy-cost-lambda.yml) handles code updates after the initial apply.
# Terraform creates the function and manages all surrounding infrastructure.

# Zip the handler for the initial deploy — CI/CD takes over code updates after that
data "archive_file" "cost_lambda" {
  type        = "zip"
  source_file = "${path.module}/../cost_lambda/handler.py"
  output_path = "${path.module}/cost_lambda.zip"
}

# ── IAM role for the Lambda ───────────────────────────────────────────────────

resource "aws_iam_role" "cost_lambda" {
  name        = "cloudrag-cost-lambda-${var.environment}"
  description = "Execution role for the daily cost-tracker Lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "cost_lambda" {
  name = "cost-explorer-cloudwatch-logs"
  role = aws_iam_role.cost_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ce:GetCostAndUsage"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ── CloudWatch permission for the ECS app role ────────────────────────────────
# The API container (aws_iam_role.app) calls cloudwatch:PutMetricData
# to publish per-query cost and latency metrics in real time.

resource "aws_iam_role_policy" "app_cloudwatch" {
  name = "cloudwatch-put-metrics"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["cloudwatch:PutMetricData"]
      Resource = "*"
    }]
  })
}

# ── Lambda function ───────────────────────────────────────────────────────────

resource "aws_lambda_function" "cost_tracker" {
  function_name    = "cloudrag-cost-tracker-${var.environment}"
  role             = aws_iam_role.cost_lambda.arn
  runtime          = "python3.11"
  handler          = "handler.lambda_handler"
  filename         = data.archive_file.cost_lambda.output_path
  source_code_hash = data.archive_file.cost_lambda.output_base64sha256
  timeout          = 60

  # After initial deploy, code updates are handled by CI/CD (update-function-code).
  # Ignore code fields so a subsequent terraform apply doesn't revert CI/CD deploys.
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

# ── EventBridge schedule — 06:00 UTC daily ───────────────────────────────────

resource "aws_cloudwatch_event_rule" "cost_tracker_schedule" {
  name                = "cloudrag-cost-tracker-daily-${var.environment}"
  description         = "Triggers cost-tracker Lambda at 06:00 UTC — Cost Explorer has full previous-day data by then"
  schedule_expression = "cron(0 6 * * ? *)"
}

resource "aws_cloudwatch_event_target" "cost_tracker" {
  rule      = aws_cloudwatch_event_rule.cost_tracker_schedule.name
  target_id = "cost-tracker-lambda"
  arn       = aws_lambda_function.cost_tracker.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cost_tracker.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cost_tracker_schedule.arn
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "cost_lambda_function_name" {
  description = "Lambda function name — used by CI/CD to deploy code updates"
  value       = aws_lambda_function.cost_tracker.function_name
}
