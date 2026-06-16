"""
Creates or updates AWS Secrets Manager secrets from your .env file.
Run this once before deploying to ECS — Terraform will reference these secrets
in the task definition so the container picks them up at runtime.

Usage:
    python create_secrets.py

What it does:
    - Reads your .env file
    - For each variable, creates or updates a secret in Secrets Manager
    - Secrets are named: cloudrag/{KEY} (e.g. cloudrag/OPENAI_API_KEY)
    - Skips empty values and comment lines

Requirements:
    - AWS CLI configured with your credentials
    - pip install boto3 python-dotenv
"""

import os
import boto3
from botocore.exceptions import ClientError
from dotenv import dotenv_values

# Which keys to push to Secrets Manager
# Add or remove keys here as needed
KEYS_TO_PUSH = [
    "OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_TRACING_V2",
    "OPENSEARCH_ENDPOINT",
    "OPENSEARCH_INDEX",
    "AWS_REGION",
    "COHERE_API_KEY",
    "API_KEY",
]

SECRET_PREFIX = "cloudrag"


def main():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print("[error] .env file not found")
        return

    env_values = dotenv_values(env_path)

    client = boto3.client(
        "secretsmanager",
        region_name=env_values.get("AWS_REGION", "eu-west-1"),
    )

    print(f"\n{'='*52}")
    print(f"  cloudRAG — push secrets to AWS Secrets Manager")
    print(f"  Prefix: {SECRET_PREFIX}/")
    print(f"{'='*52}\n")

    for key in KEYS_TO_PUSH:
        # For LANGCHAIN_PROJECT, always use cloudRAG-online in ECS
        if key == "LANGCHAIN_PROJECT":
            value = "cloudRAG-online"
        else:
            value = env_values.get(key, "")

        if not value:
            print(f"[skip]   {key} — empty in .env")
            continue

        secret_name = f"{SECRET_PREFIX}/{key}"

        try:
            client.create_secret(
                Name=secret_name,
                SecretString=value,
                Description=f"cloudRAG — {key}",
            )
            print(f"[create] {secret_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                client.update_secret(
                    SecretId=secret_name,
                    SecretString=value,
                )
                print(f"[update] {secret_name}")
            else:
                print(f"[error]  {secret_name}: {e}")

    print(f"\n[done] Secrets available at: AWS Secrets Manager → {SECRET_PREFIX}/*")
    print(f"       Terraform will reference these in the ECS task definition.\n")


if __name__ == "__main__":
    main()
