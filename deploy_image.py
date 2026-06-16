"""
Builds the Docker image and pushes it to ECR.
Run this once manually before CI/CD is set up, or to force a deploy.

Usage:
    python deploy_image.py

Requirements:
    - Docker running locally
    - AWS CLI configured
    - Terraform already applied (ECR repository must exist)
"""

import os
import subprocess
import sys
import boto3
from dotenv import load_dotenv
load_dotenv()

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"[error] Environment variable {name} is not set")
        sys.exit(1)
    return value

AWS_REGION      = _require_env("AWS_REGION")
ECR_REPOSITORY  = _require_env("ECR_REPOSITORY")
ECS_CLUSTER     = _require_env("ECS_CLUSTER")
ECS_SERVICE     = _require_env("ECS_SERVICE")

# Derive account ID from configured AWS credentials — no hardcoding needed
AWS_ACCOUNT_ID  = boto3.client("sts").get_caller_identity()["Account"]
ECR_REPO        = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY}"


def run(cmd: str, check=True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


def main():
    print(f"\n{'='*52}")
    print(f"  cloudRAG — build and push to ECR")
    print(f"  ECR: {ECR_REPO}")
    print(f"{'='*52}\n")

    # 1. Authenticate Docker with ECR
    print("[1/4] Authenticating with ECR...")
    login_cmd = (
        f"aws ecr get-login-password --region {AWS_REGION} | "
        f"docker login --username AWS --password-stdin "
        f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"
    )
    run(login_cmd)

    # 2. Build the image
    print("\n[2/4] Building Docker image...")
    run("docker build -t cloudrag .")

    # 3. Tag and push
    print("\n[3/4] Pushing to ECR...")
    run(f"docker tag cloudrag:latest {ECR_REPO}:latest")
    run(f"docker push {ECR_REPO}:latest")

    # 4. Force ECS to redeploy with the new image
    print("\n[4/4] Triggering ECS redeployment...")
    run(
        f"aws ecs update-service "
        f"--cluster {ECS_CLUSTER} "
        f"--service {ECS_SERVICE} "
        f"--force-new-deployment "
        f"--region {AWS_REGION}"
    )

    print(f"\n{'='*52}")
    print(f"  Deploy complete")
    print(f"  ECS is pulling the new image — takes ~2 min")
    print(f"  Check status:")
    print(f"  aws ecs describe-services --cluster {ECS_CLUSTER} --services {ECS_SERVICE} --region {AWS_REGION}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
