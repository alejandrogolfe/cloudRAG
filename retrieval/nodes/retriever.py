"""
Retriever node: embeds the question and searches OpenSearch.

Supports two strategies configured via config/retrieval.py:
  - knn    : pure semantic search using vector similarity
  - hybrid : combines kNN + BM25 using Reciprocal Rank Fusion (RRF)

Why RRF instead of score combination:
    kNN scores (cosine similarity) and BM25 scores are on different scales
    and cannot be directly combined. RRF works with rank positions instead
    of raw scores, making it scale-independent and robust.

    RRF formula: score(chunk) = 1/(k + rank_knn) + 1/(k + rank_bm25)
    where k=60 is a standard constant that dampens the influence of top ranks.
"""

import os
from typing import List, Dict
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

from retrieval.state import RetrievalState
from retrieval.models import RetrievedChunk
from config.retrieval import (
    RETRIEVAL_STRATEGY,
    RETRIEVAL_TOP_K_CANDIDATES,
)

openai_client = OpenAI()
RRF_K = 60  # standard RRF constant


def _get_opensearch_client() -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ["AWS_REGION"], "aoss")
    endpoint = os.environ["OPENSEARCH_ENDPOINT"].replace("https://", "").rstrip("/")
    return OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def _embed_question(question: str) -> List[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )
    return response.data[0].embedding


def _search_knn(client: OpenSearch, index: str, embedding: List[float], top_k: int) -> List[Dict]:
    response = client.search(
        index=index,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": top_k,
                    }
                }
            },
            "_source": {"excludes": ["embedding"]},
        },
    )
    return response["hits"]["hits"]


def _search_bm25(client: OpenSearch, index: str, question: str, top_k: int) -> List[Dict]:
    response = client.search(
        index=index,
        body={
            "size": top_k,
            "query": {
                "match": {
                    "content": {
                        "query": question,
                    }
                }
            },
            "_source": {"excludes": ["embedding"]},
        },
    )
    return response["hits"]["hits"]


def _reciprocal_rank_fusion(
    knn_hits: List[Dict],
    bm25_hits: List[Dict],
    top_k: int,
) -> List[Dict]:
    """
    Combines kNN and BM25 results using Reciprocal Rank Fusion.

    Each hit gets a score based on its rank in each result list.
    Hits that appear high in both lists get the highest combined score.
    Hits that only appear in one list still get partial credit.
    """
    scores: Dict[str, float] = {}
    hits_by_id: Dict[str, Dict] = {}

    for rank, hit in enumerate(knn_hits):
        # Use chunk_id from source if available, otherwise fall back to _id
        doc_id = hit["_source"].get("chunk_id") or hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)
        hits_by_id[doc_id] = hit

    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["_source"].get("chunk_id") or hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)
        hits_by_id[doc_id] = hit

    # Sort by combined RRF score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    # Return hits with RRF score replacing original score
    result = []
    for doc_id in sorted_ids:
        hit = dict(hits_by_id[doc_id])
        hit["_score"] = scores[doc_id]
        result.append(hit)

    return result


def retrieve_node(state: RetrievalState) -> dict:
    question = state["question"]
    index_name = os.environ["OPENSEARCH_INDEX"]

    query_embedding = _embed_question(question)
    client = _get_opensearch_client()

    if RETRIEVAL_STRATEGY == "hybrid":
        knn_hits = _search_knn(client, index_name, query_embedding, RETRIEVAL_TOP_K_CANDIDATES)
        bm25_hits = _search_bm25(client, index_name, question, RETRIEVAL_TOP_K_CANDIDATES)
        hits = _reciprocal_rank_fusion(knn_hits, bm25_hits, RETRIEVAL_TOP_K_CANDIDATES)
    else:
        hits = _search_knn(client, index_name, query_embedding, RETRIEVAL_TOP_K_CANDIDATES)

    chunks = []
    for hit in hits:
        src = hit["_source"]
        # Collect strategy-specific metadata fields
        extra_metadata = {}
        if src.get("parent_content"):
            extra_metadata["parent_content"] = src["parent_content"]
            extra_metadata["parent_id"] = src.get("parent_id")
        if src.get("window_content"):
            extra_metadata["window_content"] = src["window_content"]

        chunks.append(RetrievedChunk(
            chunk_id=src.get("chunk_id", hit["_id"]),
            content=src.get("content", ""),
            score=hit["_score"],
            title=src.get("title", ""),
            source=src.get("source", ""),
            url=src.get("url", ""),
            header_1=src.get("header_1"),
            header_2=src.get("header_2"),
            header_3=src.get("header_3"),
            metadata=extra_metadata,
        ))

    print(f"[retrieve/{RETRIEVAL_STRATEGY}] '{question[:60]}' → {len(chunks)} candidates (top score: {chunks[0].score:.4f})")

    return {
        "query_embedding": query_embedding,
        "retrieved_chunks": chunks,
    }
