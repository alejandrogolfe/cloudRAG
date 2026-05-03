"""
Evaluation metrics: MRR and Hit Rate.

MRR (Mean Reciprocal Rank):
  For each query, finds the rank of the correct chunk in the retrieved results.
  Score = 1/rank. If not found in top_k, score = 0.
  MRR = mean of all scores. Range: 0 to 1. Higher is better.

Hit Rate @ k:
  For each query, checks if the correct chunk appears anywhere in top_k results.
  Hit Rate = (queries where correct chunk found) / (total queries)
  Range: 0 to 1. Higher is better.
"""

from typing import List, Tuple
from ingestion.models import Chunk
from evaluation.retriever import retrieve
from config.chunking import EVAL


def _reciprocal_rank(retrieved_chunks: List[Chunk], correct_chunk_id: str) -> float:
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.chunk_id == correct_chunk_id:
            return 1.0 / rank
    return 0.0


def _hit(retrieved_chunks: List[Chunk], correct_chunk_id: str) -> bool:
    return any(c.chunk_id == correct_chunk_id for c in retrieved_chunks)


def evaluate(
    dataset: List[Tuple[str, str]],          # (query, correct_chunk_id)
    query_embeddings: List[List[float]],      # precomputed query embeddings
    chunks: List[Chunk],                      # all indexed chunks with embeddings
    top_k: int = None,
) -> dict:
    """
    Runs evaluation over the full dataset.
    Returns a dict with MRR, Hit Rate, and per-query details.
    """
    if top_k is None:
        top_k = EVAL["top_k"]

    rr_scores = []
    hits = []
    details = []

    for (query, correct_chunk_id), query_embedding in zip(dataset, query_embeddings):
        results = retrieve(query_embedding, chunks, top_k)
        retrieved_chunks = [chunk for chunk, _ in results]

        rr = _reciprocal_rank(retrieved_chunks, correct_chunk_id)
        hit = _hit(retrieved_chunks, correct_chunk_id)

        rr_scores.append(rr)
        hits.append(hit)
        details.append({
            "query": query,
            "correct_chunk_id": correct_chunk_id,
            "reciprocal_rank": rr,
            "hit": hit,
            "rank": int(1 / rr) if rr > 0 else None,
        })

    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    hit_rate = sum(hits) / len(hits) if hits else 0.0

    return {
        "mrr": round(mrr, 4),
        "hit_rate": round(hit_rate, 4),
        "total_queries": len(dataset),
        "top_k": top_k,
        "details": details,
    }
