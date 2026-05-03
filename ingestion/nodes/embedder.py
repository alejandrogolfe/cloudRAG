"""
Embed node: generates embeddings via OpenAI text-embedding-3-small.

Used during local development. In production, swap to Amazon Bedrock
Titan Embeddings v2 by changing the _embed function and updating
config/chunking.py EMBEDDING section.
"""

import os
import time
from typing import List
from openai import OpenAI
from ingestion.state import IngestionState
from ingestion.models import Chunk
from config.chunking import EMBEDDING

client = OpenAI()


def _embed(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING["model_id"],
        input=text,
    )
    return response.data[0].embedding


def embed_node(state: IngestionState) -> dict:
    chunks = state["chunks"]
    embedded: List[Chunk] = []
    batch_size = EMBEDDING["batch_size"]
    total_batches = -(-len(chunks) // batch_size)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"[embed] Batch {i // batch_size + 1}/{total_batches} ({len(batch)} chunks)...")
        for chunk in batch:
            chunk.embedding = _embed(chunk.content)
            embedded.append(chunk)
        time.sleep(0.5)  # light throttle to avoid OpenAI rate limits

    print(f"[embed] {len(embedded)} chunks embedded")
    return {"embedded_chunks": embedded}
