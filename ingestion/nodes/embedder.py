"""
Embed node: generates embeddings via Amazon Bedrock Titan v2.
Uses credentials from environment (AWS CLI or .env).
"""

import os
import json
import time
import boto3
from typing import List
from ingestion.state import IngestionState
from ingestion.models import Chunk
from config.chunking import EMBEDDING


def _get_client():
    return boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])


def _embed(client, text: str) -> List[float]:
    response = client.invoke_model(
        modelId=EMBEDDING["model_id"],
        body=json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING["dimensions"],
            "normalize": EMBEDDING["normalize"],
        }),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def embed_node(state: IngestionState) -> dict:
    chunks = state["chunks"]
    client = _get_client()
    embedded: List[Chunk] = []
    batch_size = EMBEDDING["batch_size"]
    total_batches = -(-len(chunks) // batch_size)  # ceiling division

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"[embed] Batch {i // batch_size + 1}/{total_batches} ({len(batch)} chunks)...")
        for chunk in batch:
            chunk.embedding = _embed(client, chunk.content)
            embedded.append(chunk)
        time.sleep(1)  # avoid Bedrock ThrottlingException

    print(f"[embed] {len(embedded)} chunks embedded")
    return {"embedded_chunks": embedded}
