import os
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_trace_cost(run_id: str, max_attempts: int = 3, wait_seconds: float = 1.5) -> Optional[float]:
    """Fetch cost of a specific LangSmith trace with retry backoff.

    LangSmith calculates cost asynchronously after the run completes, so we
    retry a few times before giving up. Returns None if tracing is disabled,
    the cost hasn't arrived in time, or the API call fails.
    """
    if os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() != "true":
        return None

    try:
        from langsmith import Client
        client = Client()
        for attempt in range(max_attempts):
            run = client.read_run(run_id)
            if run.total_cost is not None:
                return float(run.total_cost)
            if attempt < max_attempts - 1:
                time.sleep(wait_seconds)
        logger.warning("LangSmith cost not yet available for run %s after %d attempts", run_id, max_attempts)
        return None
    except Exception as e:
        logger.warning("Failed to read LangSmith run %s: %s", run_id, e)
        return None


def _publish_to_cloudwatch(
    cost_usd: Optional[float],
    latency_seconds: float,
    config_name: str,
    model: str,
    user_id: str,
    run_id: str,
) -> None:
    try:
        import boto3
        cloudwatch = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "eu-west-1"))

        dimensions = [
            {"Name": "ConfigName", "Value": config_name},
            {"Name": "Model",      "Value": model},
            {"Name": "UserId",     "Value": user_id},
        ]
        now = datetime.now(timezone.utc)

        metric_data = [
            {"MetricName": "QueryCount",     "Dimensions": dimensions, "Value": 1,               "Unit": "Count",   "Timestamp": now},
            {"MetricName": "LatencySeconds", "Dimensions": dimensions, "Value": latency_seconds, "Unit": "Seconds", "Timestamp": now},
            # Dimensionless rollup metrics — used by CloudWatch alarms (SEARCH not supported in alarms)
            {"MetricName": "QueryCount",     "Dimensions": [],         "Value": 1,               "Unit": "Count",   "Timestamp": now},
            {"MetricName": "LatencySeconds", "Dimensions": [],         "Value": latency_seconds, "Unit": "Seconds", "Timestamp": now},
        ]
        if cost_usd is not None:
            metric_data.extend([
                {"MetricName": "CostPerQuery", "Dimensions": dimensions, "Value": cost_usd, "Unit": "None", "Timestamp": now},
                {"MetricName": "CostPerQuery", "Dimensions": [],         "Value": cost_usd, "Unit": "None", "Timestamp": now},
            ])

        cloudwatch.put_metric_data(Namespace="cloudRAG/Costs", MetricData=metric_data)
        logger.info(
            "CloudWatch metrics published — run_id=%s latency=%.3fs cost=%s",
            run_id, latency_seconds,
            f"${cost_usd:.6f}" if cost_usd is not None else "unavailable",
        )
    except Exception as e:
        logger.warning("Failed to publish CloudWatch metrics for run %s: %s", run_id, e)


def track_query_async(
    run_id: str,
    latency_seconds: float,
    config_name: str,
    model: str,
    user_id: str,
) -> None:
    """Fetch LangSmith trace cost and publish metrics to CloudWatch in a background thread.

    Runs as a daemon thread so it never blocks the API response.
    """
    def _worker():
        cost = fetch_trace_cost(run_id)
        _publish_to_cloudwatch(cost, latency_seconds, config_name, model, user_id, run_id)

    threading.Thread(target=_worker, daemon=True, name=f"cost-tracking-{run_id}").start()
