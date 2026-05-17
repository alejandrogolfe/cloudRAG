"""
FastAPI app for the cloudRAG online pipeline.

Endpoints:
    POST /query   → receives a question, returns answer + sources
    GET  /health  → health check for ECS and load balancer

Usage (local):
    uvicorn retrieval.api:app --host 0.0.0.0 --port 8000 --reload

Usage (Docker):
    docker run -p 8000:8000 --env-file .env cloudrag
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import dataclasses

from retrieval.graph import build_retrieval_graph
from retrieval.models import RetrievedChunk

app = FastAPI(
    title="cloudRAG",
    description="Retrieval-Augmented Generation API",
    version="1.0.0",
)

# Build the graph once at startup — not on every request
graph = build_retrieval_graph()


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

    model_config = {"json_schema_extra": {"example": {"question": "What is context engineering?", "top_k": 5}}}


class SourceResponse(BaseModel):
    chunk_id: str
    content: str
    score: float
    title: str
    source: str
    url: str
    header_1: Optional[str] = None
    header_2: Optional[str] = None
    header_3: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — used by ECS and ALB to verify the container is running."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Accepts a question and returns an answer generated from relevant document chunks.

    The pipeline:
      1. Embeds the question with OpenAI text-embedding-3-small
      2. Searches OpenSearch Serverless with kNN
      3. Assembles a prompt with the top-k chunks as context
      4. Calls GPT-4o to generate the answer
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        state = graph.invoke({
            "question": request.question,
            "top_k": request.top_k,
            "query_embedding": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "prompt": "",
            "answer": "",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        SourceResponse(**dataclasses.asdict(chunk))
        for chunk in state["retrieved_chunks"]
    ]

    return QueryResponse(
        question=request.question,
        answer=state["answer"],
        sources=sources,
    )
