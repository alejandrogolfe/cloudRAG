"""
Faithfulness evaluation: checks if a retrieved chunk fully answers a query.

For each (query, retrieved_chunk) pair, asks Claude:
  "Does this chunk completely answer the question?"

Returns a score between 0 and 1 and a reason.
This detects chunks with cut ideas that MRR cannot catch.

Possible scores:
  1.0 → chunk fully answers the question
  0.5 → chunk partially answers (has relevant info but incomplete)
  0.0 → chunk does not answer the question
"""

import json
from typing import List, Tuple
from openai import OpenAI
from ingestion.models import Chunk

client = OpenAI()

FAITHFULNESS_PROMPT = """You are evaluating a RAG system. Given a question and a text chunk, assess whether the chunk fully answers the question.

Question: {query}

Chunk:
{chunk_content}

Respond ONLY with a JSON object like this:
{{
  "score": 1.0,
  "reason": "The chunk fully explains how to configure X including all required steps."
}}

Scoring rules:
- 1.0 → chunk completely answers the question, no missing information
- 0.5 → chunk is relevant and partially answers but is missing key information (idea seems cut off or incomplete)
- 0.0 → chunk does not answer the question at all

No explanation outside the JSON."""


def _evaluate_faithfulness(query: str, chunk: Chunk) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(
        query=query,
        chunk_content=chunk.content,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def evaluate_faithfulness(
    dataset: List[Tuple[str, str]],       # (query, correct_chunk_id)
    retrieved_chunks: List[Chunk],         # top-1 chunk retrieved per query
) -> dict:
    """
    Evaluates faithfulness for each (query, retrieved_chunk) pair.

    retrieved_chunks must be aligned with dataset:
      retrieved_chunks[i] is the top-1 chunk retrieved for dataset[i]

    Returns aggregated faithfulness scores and per-query details.
    """
    scores = []
    details = []

    for i, ((query, correct_chunk_id), chunk) in enumerate(zip(dataset, retrieved_chunks)):
        print(f"[faithfulness] Evaluating query {i+1}/{len(dataset)}...")
        try:
            result = _evaluate_faithfulness(query, chunk)
            score = float(result["score"])
            reason = result.get("reason", "")
        except Exception as e:
            print(f"[faithfulness] ERROR on query {i+1}: {e}")
            score = 0.0
            reason = f"Evaluation failed: {e}"

        scores.append(score)
        details.append({
            "query": query,
            "correct_chunk_id": correct_chunk_id,
            "retrieved_chunk_id": chunk.chunk_id,
            "faithfulness_score": score,
            "reason": reason,
            # Flag chunks that seem cut off (partial score)
            "possibly_cut": score == 0.5,
        })

    avg_faithfulness = sum(scores) / len(scores) if scores else 0.0
    fully_answered = sum(1 for s in scores if s == 1.0)
    partially_answered = sum(1 for s in scores if s == 0.5)
    not_answered = sum(1 for s in scores if s == 0.0)

    return {
        "avg_faithfulness": round(avg_faithfulness, 4),
        "fully_answered": fully_answered,
        "partially_answered": partially_answered,
        "not_answered": not_answered,
        "total_queries": len(dataset),
        "details": details,
    }
