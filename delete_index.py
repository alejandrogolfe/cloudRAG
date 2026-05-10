"""
Deletes the OpenSearch index so it can be recreated with the correct mapping.
Run this when the index exists with a wrong mapping and uploads are failing.

Usage:
    python delete_index.py
"""

import os
import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

load_dotenv()


def main():
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ["AWS_REGION"], "aoss")
    endpoint = os.environ["OPENSEARCH_ENDPOINT"].replace("https://", "").rstrip("/")

    client = OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    index_name = os.environ["OPENSEARCH_INDEX"]

    if not client.indices.exists(index=index_name):
        print(f"[delete] Index '{index_name}' does not exist, nothing to delete")
        return

    client.indices.delete(index=index_name)
    print(f"[delete] Index '{index_name}' deleted — run run_upload.py to recreate it")


if __name__ == "__main__":
    main()
