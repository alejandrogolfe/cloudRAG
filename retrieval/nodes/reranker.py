"""
Reranker node: reorders retrieved chunks using Cohere Rerank API.

Why reranking:
    kNN retrieval ranks by vector similarity — fast but imprecise.
    A reranker is a cross-encoder that sees (query, chunk) together
    and scores relevance more accurately. The result is a better
    top-k selection before sending context to GPT-4o.

Flow:
    retrieve (top-20 candidates) → rerank → augment (top-5 final)

If RERANKING_ENABLED = False in config/retrieval.py, this node
simply returns the first RETRIEVAL_TOP_K_FINAL candidates unchanged.
"""

import os
import logging
from typing import List
import cohere
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

from retrieval.state import RetrievalState
from retrieval.models import RetrievedChunk
from config.retrieval import RERANKING_ENABLED, RERANKING_MODEL, RETRIEVAL_TOP_K_FINAL


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def _call_cohere_rerank(question: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
    cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
    response = cohere_client.rerank(
        model=RERANKING_MODEL,
        query=question,
        documents=[chunk.content for chunk in candidates],
        top_n=RETRIEVAL_TOP_K_FINAL,
    )
    return [
        RetrievedChunk(
            chunk_id=candidates[r.index].chunk_id,
            content=candidates[r.index].content,
            score=r.relevance_score,
            title=candidates[r.index].title,
            source=candidates[r.index].source,
            url=candidates[r.index].url,
            header_1=candidates[r.index].header_1,
            header_2=candidates[r.index].header_2,
            header_3=candidates[r.index].header_3,
            metadata=candidates[r.index].metadata,
        )
        for r in response.results
    ]


@traceable(name="rerank", run_type="chain")
def rerank_node(state: RetrievalState) -> dict:
    question = state["question"]
    candidates = state["retrieved_chunks"]

    if not RERANKING_ENABLED:
        final_chunks = candidates[:RETRIEVAL_TOP_K_FINAL]
        logger.info(f"rerank disabled — using top {len(final_chunks)} kNN results directly")
        return {"reranked_chunks": final_chunks}

    try:
        reranked = _call_cohere_rerank(question, candidates)
        logger.info(f"rerank — {len(candidates)} candidates → top {len(reranked)} (top score: {reranked[0].score:.4f})")
        return {"reranked_chunks": reranked}
    except Exception as e:
        logger.warning(f"Cohere rerank failed after retries ({e}) — falling back to kNN top-{RETRIEVAL_TOP_K_FINAL}")
        return {"reranked_chunks": candidates[:RETRIEVAL_TOP_K_FINAL]}
