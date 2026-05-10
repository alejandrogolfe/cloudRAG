"""
Upload pipeline: takes locally saved chunks and indexes them into OpenSearch Serverless.
Run this only when you're satisfied with the chunking strategy.

Usage:
    python run_upload.py --chunks-path ./output/chunks_structure.json

Authentication:
    Uses your AWS credentials via the default credential chain
    (AWS CLI profile, env vars, or IAM role). No password needed.
    Your IAM principal must be listed in the OpenSearch data access policy.
"""

import os
import json
import argparse
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.serializer import JSONSerializer
import boto3

load_dotenv()

# OpenSearch Serverless uses 1024-dim Titan v2 embeddings in prod.
# During local dev we use OpenAI text-embedding-3-small which also outputs
# 1536 dims by default — but we configure it to 1024 in config/chunking.py
# to match the prod dimension. Both map to the same index schema.
VECTOR_DIMENSION = 1536


def _get_client() -> OpenSearch:
    """
    Builds an OpenSearch client authenticated via AWS SigV4.

    OpenSearch Serverless uses the service name 'aoss' (not 'es').
    The endpoint should NOT include 'https://' — opensearch-py adds it.
    """
    credentials = boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE") or None
    ).get_credentials()

    auth = AWSV4SignerAuth(
        credentials,
        os.environ["AWS_REGION"],
        "aoss",  # aoss = OpenSearch Serverless; 'es' = OpenSearch managed
    )

    endpoint = os.environ["OPENSEARCH_ENDPOINT"]
    # Strip protocol if accidentally included in the env var
    endpoint = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

    return OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
        serializer=JSONSerializer(),
    )


def _ensure_index(client: OpenSearch, index_name: str) -> None:
    """
    Creates the index with kNN mapping if it doesn't exist.

    Key difference from managed OpenSearch:
    - OpenSearch Serverless does NOT support the 'nmslib' engine.
    - Use 'faiss' instead. The rest of the mapping is identical.
    - space_type 'cosinesimil' works with faiss on Serverless.
    """
    if client.indices.exists(index=index_name):
        print(f"[upload] Index '{index_name}' already exists — skipping creation")
        return

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,  # higher = better recall, slower
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIMENSION,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss",  # nmslib is NOT supported on Serverless
                        "parameters": {
                            "m": 16,          # number of bi-directional links per node
                            "ef_construction": 100,  # build-time accuracy vs speed
                        },
                    },
                },
                "content":       {"type": "text"},
                "source":        {"type": "keyword"},
                "title":         {"type": "text"},
                "url":           {"type": "keyword"},
                "header_1":      {"type": "text"},
                "header_2":      {"type": "text"},
                "header_3":      {"type": "text"},
                "chunk_index":   {"type": "integer"},
                "total_chunks":  {"type": "integer"},
                "char_count":    {"type": "integer"},
                "strategy":      {"type": "keyword"},
                "chunk_id":      {"type": "keyword"},
            }
        },
    }

    client.indices.create(index=index_name, body=json.dumps(mapping))
    print(f"[upload] Created index '{index_name}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-path",
        required=True,
        help="Path to chunks JSON file (output of run_local.py)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of documents per bulk request (default: 50)",
    )
    args = parser.parse_args()

    index_name = os.environ["OPENSEARCH_INDEX"]

    with open(args.chunks_path) as f:
        chunks = json.load(f)

    print(f"\n{'='*52}")
    print(f"  cloudRAG — upload to OpenSearch Serverless")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Index  : {index_name}")
    print(f"{'='*52}\n")

    client = _get_client()
    _ensure_index(client, index_name)

    indexed, errors = 0, []

    # Upload in batches to avoid request size limits (Serverless: 10MB max)
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        batch_num = start // args.batch_size + 1
        total_batches = -(-len(chunks) // args.batch_size)
        print(f"[upload] Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        for chunk in batch:
            try:
                doc = {
                    "chunk_id":  chunk["chunk_id"],
                    "embedding": chunk["embedding"],
                    "content":   chunk["content"],
                    **chunk["metadata"],
                }
                # DEBUG — remove after fixing
                if indexed == 0 and not errors:
                    print("DEBUG embedding type :", type(doc["embedding"]))
                    print("DEBUG embedding[:3]  :", doc["embedding"][:3])
                    serialized = json.dumps(doc)
                    print("DEBUG json preview   :", serialized[:300])
                client.index(
                    index=index_name,
                    body=doc,
                )
                indexed += 1
            except Exception as e:
                errors.append(f"{chunk['chunk_id']}: {e}")

    print(f"\n[upload] Done — {indexed} indexed, {len(errors)} errors")
    if errors:
        print("\nFirst errors:")
        for err in errors[:5]:
            print(f"  - {err}")
    else:
        print(f"\n[upload] All chunks in '{index_name}'. Verify with:")
        print(f"  python verify_upload.py --index {index_name}")


if __name__ == "__main__":
    main()
