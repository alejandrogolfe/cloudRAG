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

# ── Alarm 1: avg cost per query exceeds fixed threshold ────────────────────────
#
# Uses SEARCH to aggregate CostPerQuery across all ConfigName/Model/UserId
# combinations, then averages them. This way the alarm reflects overall
# query cost regardless of which config or user is active.
#
# Threshold default: $0.01/query. Set var.cost_per_query_threshold_usd after
# a few days of real data when you have a baseline average to reference.

resource "aws_cloudwatch_metric_alarm" "cost_per_query" {
  alarm_name          = "cloudrag-cost-per-query-${var.environment}"
  alarm_description   = "Avg cost per query exceeded $${var.cost_per_query_threshold_usd} USD"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = var.cost_per_query_threshold_usd
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "all_costs"
    period      = 86400
    expression  = "SEARCH('{cloudRAG/Costs} MetricName=\"CostPerQuery\"', 'Average', 86400)"
    label       = "CostPerQuery (all dims)"
    return_data = false
  }

  metric_query {
    id          = "avg_cost"
    period      = 86400
    expression  = "AVG(all_costs)"
    label       = "Avg Cost per Query"
    return_data = true
  }
}

# ── Alarm 2: daily accumulated LLM cost exceeds budget ────────────────────────
#
# Sums CostPerQuery across all dimension combinations for the day.
# With period=86400, this evaluates once per day and reflects the full
# daily spend when it fires.

resource "aws_cloudwatch_metric_alarm" "daily_cost_budget" {
  alarm_name          = "cloudrag-daily-cost-budget-${var.environment}"
  alarm_description   = "Daily total LLM cost exceeded $${var.daily_cost_budget_usd} USD"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = var.daily_cost_budget_usd
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]

  metric_query {
    id          = "all_costs"
    period      = 86400
    expression  = "SEARCH('{cloudRAG/Costs} MetricName=\"CostPerQuery\"', 'Sum', 86400)"
    label       = "CostPerQuery (all dims)"
    return_data = false
  }

  metric_query {
    id          = "total_cost"
    period      = 86400
    expression  = "SUM(all_costs)"
    label       = "Daily Total LLM Cost"
    return_data = true
  }
}

# ── Alarm 3: anomaly detection on avg cost per query ──────────────────────────
#
# Trains a statistical model on the historical average cost per query and
# alerts when the value exceeds the upper bound of the expected range.
# Uses ANOMALY_DETECTION_BAND with var.anomaly_detection_std_devs (default: 2).
#
# This alarm complements Alarm 1: Alarm 1 catches absolute violations of your
# cost budget; Alarm 3 catches sudden relative spikes even if the absolute
# value is still within budget (e.g. cost doubles due to a prompt regression).
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
    id          = "all_costs"
    period      = 86400
    expression  = "SEARCH('{cloudRAG/Costs} MetricName=\"CostPerQuery\"', 'Average', 86400)"
    label       = "CostPerQuery (all dims)"
    return_data = false
  }

  metric_query {
    id          = "avg_cost"
    period      = 86400
    expression  = "AVG(all_costs)"
    label       = "Avg Cost per Query"
    return_data = false
  }

  metric_query {
    id          = "anomaly_band"
    period      = 86400
    expression  = "ANOMALY_DETECTION_BAND(avg_cost, ${var.anomaly_detection_std_devs})"
    label       = "Expected Cost Range"
    return_data = true
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "sns_alerts_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications"
  value       = aws_sns_topic.alerts.arn
}
