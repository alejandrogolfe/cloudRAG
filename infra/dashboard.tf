# ── CloudWatch Dashboard ───────────────────────────────────────────────────────
#
# All "grouped by" panels use CloudWatch Metrics Insights (SQL) so they work
# dynamically with any ConfigName or Model value — no need to hardcode dimension
# values in Terraform.
#
# UserId is NOT a CloudWatch metric dimension (high cardinality would blow up
# custom-metric cost and console usability). The per-user widgets below query
# the structured log line in monitoring/cost_tracking.py instead, via Logs Insights.
#
# Layout: 4 rows × 2 columns (each widget is 12 units wide, 6 tall)
#
#  Row 1 │ Avg cost / query by config  │ Daily total LLM cost            │
#  Row 2 │ Cost distribution by model  │ Query volume by config           │
#  Row 3 │ Avg latency by config       │ Infra cost + LLM cost combined   │
#  Row 4 │ Top users by cost (logs)    │ Query volume by user (logs)      │

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "cloudRAG-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [

      # ── Row 1, Col 1 ─────────────────────────────────────────────────────────
      # period=86400 (daily) so comparisons between configs are reliable averages,
      # not short-window noise.
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Avg Cost per Query by Config (daily)"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [[{
            expression = "SELECT AVG(CostPerQuery) FROM \"cloudRAG/Costs\" GROUP BY ConfigName"
            id         = "q1"
            label      = ""
            period     = 86400
          }]]
          yAxis = { left = { label = "USD", showUnits = false } }
        }
      },

      # ── Row 1, Col 2 ─────────────────────────────────────────────────────────
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Daily Total LLM Cost (all configs)"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [[{
            expression = "SELECT SUM(CostPerQuery) FROM \"cloudRAG/Costs\""
            id         = "q1"
            label      = "Total LLM cost"
            period     = 86400
          }]]
          yAxis = { left = { label = "USD", showUnits = false } }
        }
      },

      # ── Row 2, Col 1 ─────────────────────────────────────────────────────────
      # Bar chart shows cost contribution per model over the selected time range.
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Cost Distribution by Model"
          view   = "bar"
          region = var.aws_region
          metrics = [[{
            expression = "SELECT SUM(CostPerQuery) FROM \"cloudRAG/Costs\" GROUP BY Model"
            id         = "q1"
            label      = ""
            period     = 86400
          }]]
          yAxis = { left = { label = "USD", showUnits = false } }
        }
      },

      # ── Row 2, Col 2 ─────────────────────────────────────────────────────────
      # Volume confirms that cost comparisons between configs have enough sample size.
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Query Volume by Config (daily)"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [[{
            expression = "SELECT SUM(QueryCount) FROM \"cloudRAG/Costs\" GROUP BY ConfigName"
            id         = "q1"
            label      = ""
            period     = 86400
          }]]
          yAxis = { left = { label = "Queries", showUnits = false } }
        }
      },

      # ── Row 3, Col 1 ─────────────────────────────────────────────────────────
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6
        properties = {
          title  = "Avg Latency by Config (daily)"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [[{
            expression = "SELECT AVG(LatencySeconds) FROM \"cloudRAG/Costs\" GROUP BY ConfigName"
            id         = "q1"
            label      = ""
            period     = 86400
          }]]
          yAxis = { left = { label = "Seconds", showUnits = false } }
        }
      },

      # ── Row 3, Col 2 ─────────────────────────────────────────────────────────
      # Combined infra + LLM cost for total spend visibility.
      #
      # NOTE: InfraCostDaily only publishes 1 data point per day (from the daily
      # Lambda). This panel will appear empty for the first 24-48 hours after
      # initial deployment while the Lambda accumulates its first data points.
      {
        type   = "metric"
        x      = 12
        y      = 12
        width  = 12
        height = 6
        properties = {
          title  = "Total Cost: Infra (by Service) + LLM (daily)"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [
            [{
              expression = "SELECT SUM(InfraCostDaily) FROM \"cloudRAG/Costs\" GROUP BY Service"
              id         = "q1"
              label      = "Infra"
              period     = 86400
            }],
            [{
              expression = "SELECT SUM(CostPerQuery) FROM \"cloudRAG/Costs\""
              id         = "q2"
              label      = "LLM"
              period     = 86400
            }]
          ]
          yAxis = { left = { label = "USD", showUnits = false } }
        }
      },

      # ── Row 4, Col 1 ─────────────────────────────────────────────────────────
      # Top users by accumulated cost, derived from the structured log line in
      # monitoring/cost_tracking.py (no CloudWatch dimension cardinality cost).
      {
        type   = "log"
        x      = 0
        y      = 18
        width  = 12
        height = 6
        properties = {
          title  = "Top 10 Users by Cost (accumulated, from logs)"
          view   = "bar"
          region = var.aws_region
          query  = <<-EOQ
            SOURCE '${aws_cloudwatch_log_group.app.name}'
            | filter message like /CloudWatch metrics published/
            | parse message "user_id=* config_name=* model=* latency_seconds=* cost_usd=*" as user_id, config_name, model, latency_seconds, cost_usd
            | filter cost_usd != "null"
            | stats sum(cost_usd) as total_cost by user_id
            | sort total_cost desc
            | limit 10
          EOQ
        }
      },

      # ── Row 4, Col 2 ─────────────────────────────────────────────────────────
      {
        type   = "log"
        x      = 12
        y      = 18
        width  = 12
        height = 6
        properties = {
          title  = "Top 10 Users by Query Volume (from logs)"
          view   = "bar"
          region = var.aws_region
          query  = <<-EOQ
            SOURCE '${aws_cloudwatch_log_group.app.name}'
            | filter message like /CloudWatch metrics published/
            | parse message "user_id=* config_name=* model=* latency_seconds=* cost_usd=*" as user_id, config_name, model, latency_seconds, cost_usd
            | stats count(*) as query_count by user_id
            | sort query_count desc
            | limit 10
          EOQ
        }
      }

    ]
  })
}
