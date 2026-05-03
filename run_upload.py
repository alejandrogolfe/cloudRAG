"""
Upload pipeline: takes locally saved chunks and indexes them into OpenSearch.
Run this only when you're satisfied with the chunking strategy.

Usage:
    python run_upload.py --chunks-path ./output/chunks_structure.json
"""

import os
import json
import argparse
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

load_dotenv()

VECTOR_DIMENSION = 1024


def _get_client():
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ["AWS_REGION"], "aoss")
    return OpenSearch(
        hosts=[{"host": os.environ["OPENSEARCH_ENDPOINT"], "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def _ensure_index(client, index_name: str):
    if client.indices.exists(index=index_name):
        print(f"[upload] Index '{index_name}' already exists")
        return
    mapping = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIMENSION,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
                "content": {"type": "text"},
                "source": {"type": "keyword"},
                "title": {"type": "text"},
                "url": {"type": "keyword"},
                "header_1": {"type": "text"},
                "header_2": {"type": "text"},
                "header_3": {"type": "text"},
                "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "char_count": {"type": "integer"},
                "strategy": {"type": "keyword"},
            }
        }
    }
    client.indices.create(index=index_name, body=mapping)
    print(f"[upload] Created index '{index_name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-path", required=True, help="Path to chunks JSON file")
    args = parser.parse_args()

    index_name = os.environ["OPENSEARCH_INDEX"]

    print(f"\n{'='*50}")
    print(f" cloudRAG — upload to OpenSearch")
    print(f" Chunks:  {args.chunks_path}")
    print(f" Index:   {index_name}")
    print(f"{'='*50}\n")

    with open(args.chunks_path) as f:
        chunks = json.load(f)

    client = _get_client()
    _ensure_index(client, index_name)

    indexed, errors = 0, []
    for chunk in chunks:
        try:
            client.index(
                index=index_name,
                id=chunk["chunk_id"],
                body={
                    "embedding": chunk["embedding"],
                    "content": chunk["content"],
                    **chunk["metadata"],
                }
            )
            indexed += 1
        except Exception as e:
            errors.append(f"{chunk['chunk_id']}: {e}")

    print(f"\n[upload] Indexed {indexed} chunks, {len(errors)} errors")
    if errors:
        for err in errors[:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
