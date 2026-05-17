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
import cohere

from retrieval.state import RetrievalState
from retrieval.models import RetrievedChunk
from config.retrieval import RERANKING_ENABLED, RERANKING_MODEL, RETRIEVAL_TOP_K_FINAL


def rerank_node(state: RetrievalState) -> dict:
    question = state["question"]
    candidates = state["retrieved_chunks"]

    if not RERANKING_ENABLED:
        # No reranking — just take the top RETRIEVAL_TOP_K_FINAL from kNN results
        final_chunks = candidates[:RETRIEVAL_TOP_K_FINAL]
        print(f"[rerank] disabled — using top {len(final_chunks)} kNN results directly")
        return {"reranked_chunks": final_chunks}

    # Call Cohere Rerank API
    cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    response = cohere_client.rerank(
        model=RERANKING_MODEL,
        query=question,
        documents=[chunk.content for chunk in candidates],
        top_n=RETRIEVAL_TOP_K_FINAL,
    )

    # Reorder chunks according to Cohere's ranking
    reranked = []
    for result in response.results:
        chunk = candidates[result.index]
        # Replace kNN score with Cohere relevance score for transparency
        reranked.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=result.relevance_score,
            title=chunk.title,
            source=chunk.source,
            url=chunk.url,
            header_1=chunk.header_1,
            header_2=chunk.header_2,
            header_3=chunk.header_3,
        ))

    print(f"[rerank] {len(candidates)} candidates → top {len(reranked)} (top score: {reranked[0].score:.4f})")

    return {"reranked_chunks": reranked}
