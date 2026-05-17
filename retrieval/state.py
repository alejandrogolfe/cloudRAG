from typing import TypedDict, List
from retrieval.models import RetrievedChunk


class RetrievalState(TypedDict):
    question: str
    top_k: int

    query_embedding: List[float]            # filled by retriever node
    retrieved_chunks: List[RetrievedChunk]  # filled by retriever node (candidates)
    reranked_chunks: List[RetrievedChunk]   # filled by reranker node (final selection)
    prompt: str                             # filled by augmenter node
    answer: str                             # filled by generator node
