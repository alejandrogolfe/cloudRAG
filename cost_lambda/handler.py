"""
Daily Lambda: reads previous day's AWS infrastructure cost via Cost Explorer,
grouped by service and filtered by Project=cloudRAG tag, then publishes each
service cost as an InfraCostDaily metric to CloudWatch.

Triggered at 06:00 UTC daily via EventBridge — at that hour Cost Explorer
already has the complete data for the previous calendar day.

NOTE: Cost Explorer is a global API that must be called against us-east-1,
regardless of where this Lambda runs.
"""

import os
import logging
from datetime import date, timedelta

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TAG_KEY   = "Project"
TAG_VALUE = "cloudRAG"
NAMESPACE = "cloudRAG/Costs"

# Cost Explorer service name → short label used as CloudWatch dimension value.
# If a service is missing from this map its full AWS name is used (truncated to 256 chars).
SERVICE_MAP = {
    "Amazon Elastic Container Service":          "ECS",
    "Amazon OpenSearch Service":                 "OpenSearch",
    "Amazon OpenSearch Serverless":              "OpenSearch",
    "Amazon EC2 Container Registry (ECR)":       "ECR",
    "AWS Secrets Manager":                       "SecretsManager",
    "Amazon Elastic Load Balancing":             "ALB",
    "Amazon Virtual Private Cloud":              "VPC",
    "AmazonCloudWatch":                          "CloudWatch",
}


def lambda_handler(event, context):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today     = date.today().isoformat()

    # Cost Explorer is always called against us-east-1 (global service endpoint)
    ce          = boto3.client("ce", region_name="us-east-1")
    cloudwatch  = boto3.client("cloudwatch", region_name=os.environ["AWS_REGION"])

    response = ce.get_cost_and_usage(
        TimePeriod={"Start": yesterday, "End": today},
        Granularity="DAILY",
        Filter={"Tags": {"Key": TAG_KEY, "Values": [TAG_VALUE]}},
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        Metrics=["UnblendedCost"],
    )

    groups = response["ResultsByTime"][0]["Groups"] if response["ResultsByTime"] else []

    if not groups:
        logger.warning(
            "No costs found for tag %s=%s on %s. "
            "Resources are likely missing the Project tag — verify tagging on ECS, "
            "OpenSearch Serverless, and ECR in the AWS Console.",
            TAG_KEY, TAG_VALUE, yesterday,
        )
        return {"status": "no_data", "date": yesterday}

    metric_data = []
    for group in groups:
        service_name = group["Keys"][0]
        amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
        if amount == 0.0:
            continue
        label = SERVICE_MAP.get(service_name, service_name[:256])
        logger.info("  %s → %s = $%.6f", service_name, label, amount)
        metric_data.append({
            "MetricName": "InfraCostDaily",
            "Dimensions": [{"Name": "Service", "Value": label}],
            "Value": amount,
            "Unit": "None",
        })

    if not metric_data:
        logger.warning("All tagged services reported $0 for %s — check Cost Explorer tag filtering.", yesterday)
        return {"status": "zero_costs", "date": yesterday}

    cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
    logger.info("Published %d InfraCostDaily metrics for %s", len(metric_data), yesterday)
    return {"status": "ok", "date": yesterday, "metrics_published": len(metric_data)}
