"""
Local retriever: cosine similarity search using numpy.
No OpenSearch needed — works entirely in memory.
This is mathematically identical to what OpenSearch does internally.
"""

import numpy as np
from typing import List, Tuple
from ingestion.models import Chunk
from config.chunking import EMBEDDING, EVAL


def _cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a query vector and a matrix of chunk vectors.
    Since Bedrock normalizes vectors (normalize=True), this is just a dot product.
    Returns an array of similarity scores.
    """
    return np.dot(chunk_vecs, query_vec)


def retrieve(
    query_embedding: List[float],
    chunks: List[Chunk],
    top_k: int = None,
) -> List[Tuple[Chunk, float]]:
    """
    Returns the top_k most similar chunks to the query embedding.
    Returns list of (chunk, similarity_score) sorted by score descending.
    """
    if top_k is None:
        top_k = EVAL["top_k"]

    query_vec = np.array(query_embedding)
    chunk_vecs = np.array([c.embedding for c in chunks])

    scores = _cosine_similarity(query_vec, chunk_vecs)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(chunks[i], float(scores[i])) for i in top_indices]
