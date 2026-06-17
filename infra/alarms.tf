# ── SNS topic for alarm notifications ─────────────────────────────────────────

resource "aws_sns_topic" "alerts" {
  name = "cloudrag-alerts-${var.environment}"
}

resource "aws_sns_topic_subscription" "alert_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
  # AWS sends a confirmation email to this address on first apply.
  # The subscription stays in "pending confirmation" until the link is clicked.
}

# NOTE: CloudWatch metric alarms do not support SEARCH() expressions.
# The app publishes two copies of each metric:
#   - With dimensions (ConfigName, Model, UserId) → used by the dashboard
#   - Without dimensions (rollup)                 → used by these alarms
# Both go to the same namespace "cloudRAG/Costs".

# ── Alarm 1: avg cost per query exceeds fixed threshold ────────────────────────
#
# Threshold default: $0.01/query. Adjust after a few days of real data
# once you have a baseline average to reference.

resource "aws_cloudwatch_metric_alarm" "cost_per_query" {
  alarm_name          = "cloudrag-cost-per-query-${var.environment}"
  alarm_description   = "Avg cost per query exceeded $${var.cost_per_query_threshold_usd} USD"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = var.cost_per_query_threshold_usd
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]

  namespace   = "cloudRAG/Costs"
  metric_name = "CostPerQuery"
  statistic   = "Average"
  period      = 86400
}

# ── Alarm 2: daily accumulated LLM cost exceeds budget ────────────────────────

resource "aws_cloudwatch_metric_alarm" "daily_cost_budget" {
  alarm_name          = "cloudrag-daily-cost-budget-${var.environment}"
  alarm_description   = "Daily total LLM cost exceeded $${var.daily_cost_budget_usd} USD"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = var.daily_cost_budget_usd
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]

  namespace   = "cloudRAG/Costs"
  metric_name = "CostPerQuery"
  statistic   = "Sum"
  period      = 86400
}

# ── Alarm 3: anomaly detection on avg cost per query ──────────────────────────
#
# Trains a statistical model on the dimensionless CostPerQuery rollup metric
# and alerts when it exceeds the upper bound of the expected range.
#
# Note: anomaly detection needs ~2 weeks of data to build a reliable model.
# False positives are expected in the first days after deployment.

resource "aws_cloudwatch_metric_alarm" "cost_anomaly" {
  alarm_name          = "cloudrag-cost-anomaly-${var.environment}"
  alarm_description   = "CostPerQuery is above the anomaly detection upper band (${var.anomaly_detection_std_devs} std devs)"
  comparison_operator = "GreaterThanUpperThreshold"
  evaluation_periods  = 1
  threshold_metric_id = "anomaly_band"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]

  metric_query {
    id = "m1"
    metric {
      namespace   = "cloudRAG/Costs"
      metric_name = "CostPerQuery"
      period      = 86400
      stat        = "Average"
    }
    return_data = true
  }

  metric_query {
    id          = "anomaly_band"
    expression  = "ANOMALY_DETECTION_BAND(m1, ${var.anomaly_detection_std_devs})"
    label       = "Expected Cost Range"
    return_data = true
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "sns_alerts_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications"
  value       = aws_sns_topic.alerts.arn
}
