"""
Retriever node: embeds the question and searches OpenSearch with kNN.

Embedding model must match what was used during ingestion:
- Local dev : OpenAI text-embedding-3-small (1536 dims)
- Production: Amazon Bedrock Titan Embeddings v2 (1024 dims)
"""

import os
from typing import List
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

from retrieval.state import RetrievalState
from retrieval.models import RetrievedChunk

openai_client = OpenAI()


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


def retrieve_node(state: RetrievalState) -> dict:
    question = state["question"]
    top_k = state["top_k"]
    index_name = os.environ["OPENSEARCH_INDEX"]

    # Embed the question
    query_embedding = _embed_question(question)

    # kNN search
    client = _get_opensearch_client()
    response = client.search(
        index=index_name,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
            # Exclude the embedding from the response to keep payloads small
            "_source": {"excludes": ["embedding"]},
        },
    )

    chunks = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
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
        ))

    print(f"[retrieve] '{question[:60]}...' → {len(chunks)} chunks (top score: {chunks[0].score:.4f})")

    return {
        "query_embedding": query_embedding,
        "retrieved_chunks": chunks,
    }
