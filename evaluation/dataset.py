"""
Generates synthetic (query, chunk_id) pairs using OpenAI gpt-4o-mini.
Each chunk gets N questions that it directly answers.
This gives us a ground truth dataset for MRR evaluation without manual labeling.
"""

import os
import json
from typing import List, Tuple
from openai import OpenAI
from ingestion.models import Chunk
from config.chunking import EVAL

client = OpenAI()


def _generate_queries(chunk: Chunk) -> List[str]:
    """Asks Claude to generate N questions answered by this chunk."""
    n = EVAL["queries_per_chunk"]
    prompt = f"""You are building an evaluation dataset for a RAG system.

Given the following text chunk, generate exactly {n} questions that:
- Are directly and fully answered by this text
- Sound like real user questions (not academic)
- Are specific enough that only this chunk would answer them

Text:
{chunk.content}

Return ONLY a JSON array of strings with the questions. No explanation, no markdown.
Example: ["What is X?", "How do I configure Y?"]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def generate_dataset(chunks: List[Chunk]) -> List[Tuple[str, str]]:
    """
    Returns a list of (query, chunk_id) pairs.
    chunk_id is the ground truth: the chunk that should be retrieved for that query.
    """
    dataset = []
    for i, chunk in enumerate(chunks):
        print(f"[dataset] Generating queries for chunk {i+1}/{len(chunks)}: {chunk.chunk_id}")
        try:
            queries = _generate_queries(chunk)
            for query in queries:
                dataset.append((query, chunk.chunk_id))
        except Exception as e:
            print(f"[dataset] ERROR on chunk {chunk.chunk_id}: {e}")

    print(f"[dataset] Generated {len(dataset)} query-chunk pairs from {len(chunks)} chunks")
    return dataset
