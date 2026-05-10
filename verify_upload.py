"""
Verify that chunks were correctly uploaded to OpenSearch Serverless.

Usage:
    python verify_upload.py --index cloudrag-docs

What it checks:
    1. Collection is reachable and the index exists
    2. Document count matches what you uploaded
    3. A sample kNN query returns sensible results
    4. Spot-checks a random chunk to confirm embeddings were stored
"""

import os
import json
import random
import argparse
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

load_dotenv()


def _get_client() -> OpenSearch:
    credentials = boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE") or None
    ).get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ["AWS_REGION"], "aoss")
    endpoint = os.environ["OPENSEARCH_ENDPOINT"]
    endpoint = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    return OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--chunks-path",
        default=None,
        help="Optional: path to local chunks JSON to cross-check document count",
    )
    args = parser.parse_args()

    client = _get_client()

    print(f"\n{'='*52}")
    print(f"  cloudRAG — verify index '{args.index}'")
    print(f"{'='*52}\n")

    # ── 1. Index exists ──────────────────────────────────────────────────────
    if not client.indices.exists(index=args.index):
        print(f"[FAIL] Index '{args.index}' does not exist")
        return
    print(f"[OK] Index exists")

    # ── 2. Document count ────────────────────────────────────────────────────
    count_resp = client.count(index=args.index)
    doc_count = count_resp["count"]
    print(f"[OK] Documents indexed: {doc_count}")

    if args.chunks_path:
        with open(args.chunks_path) as f:
            local_chunks = json.load(f)
        local_count = len(local_chunks)
        if doc_count == local_count:
            print(f"[OK] Count matches local file ({local_count})")
        else:
            print(f"[WARN] Count mismatch: OpenSearch={doc_count}, local={local_count}")

    # ── 3. Spot-check a random document ──────────────────────────────────────
    sample = client.search(
        index=args.index,
        body={"size": 1, "query": {"match_all": {}}},
    )
    hit = sample["hits"]["hits"][0]
    src = hit["_source"]

    has_embedding = "embedding" in src and len(src["embedding"]) > 0
    has_content = "content" in src and len(src["content"]) > 10

    print(f"[OK] Sample doc ID : {hit['_id']}")
    print(f"     Embedding dim  : {len(src.get('embedding', []))}")
    print(f"     Content preview: {src.get('content', '')[:80]}...")
    print(f"     Strategy       : {src.get('strategy', 'unknown')}")

    if not has_embedding:
        print(f"[WARN] Embedding missing or empty in sample doc")
    if not has_content:
        print(f"[WARN] Content missing or too short in sample doc")

    # ── 4. kNN query smoke test ───────────────────────────────────────────────
    # Use the stored embedding of the sample doc as a query vector.
    # Top result should be the doc itself (score = 1.0).
    query_vector = src["embedding"]
    knn_resp = client.search(
        index=args.index,
        body={
            "size": 3,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": 3,
                    }
                }
            },
        },
    )
    knn_hits = knn_resp["hits"]["hits"]
    top_id = knn_hits[0]["_id"]
    top_score = knn_hits[0]["_score"]

    if top_id == hit["_id"] and top_score > 0.99:
        print(f"[OK] kNN query works — top result is self (score={top_score:.4f})")
    else:
        print(f"[WARN] kNN top result is not self: {top_id} (score={top_score:.4f})")
        print(f"       This can happen with cosine sim if embeddings are unnormalized.")

    print(f"\n{'='*52}")
    print(f"  Verification complete")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
