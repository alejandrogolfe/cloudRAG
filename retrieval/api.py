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

import os
import logging
import json
from fastapi import FastAPI, Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import List, Optional
import dataclasses
import time
import uuid
from langchain_core.runnables import RunnableConfig


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)

_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)

logger = logging.getLogger(__name__)

from retrieval.graph import build_retrieval_graph
from retrieval.models import RetrievedChunk
from monitoring.cost_tracking import track_query_async
from retrieval.nodes.generator import llm as _generator_llm
from config.retrieval import RETRIEVAL_STRATEGY, RERANKING_ENABLED
from config.chunking import CHUNKING_STRATEGY

_CONFIG_NAME = (
    f"{CHUNKING_STRATEGY}_{RETRIEVAL_STRATEGY}_"
    f"{'rerank' if RERANKING_ENABLED else 'norerank'}"
)
_MODEL_NAME = _generator_llm.model_name

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _require_api_key(api_key: str = Security(_API_KEY_HEADER)) -> str:
    expected = os.environ.get("API_KEY", "")
    if not expected:
        raise RuntimeError("API_KEY environment variable is not set")
    if api_key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
    return api_key

_RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="cloudRAG",
    description="Retrieval-Augmented Generation API",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit(_RATE_LIMIT)
def query(
    request: Request,
    body: QueryRequest,
    _: str = Security(_require_api_key),
    user_id: str = Header(..., alias="X-User-Id"),
):
    """
    Accepts a question and returns an answer generated from relevant document chunks.

    The pipeline:
      1. Embeds the question with OpenAI text-embedding-3-small
      2. Searches OpenSearch Serverless with kNN
      3. Assembles a prompt with the top-k chunks as context
      4. Calls GPT-4o to generate the answer
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    run_id = uuid.uuid4()
    run_config = RunnableConfig(
        run_id=run_id,
        metadata={
            "config_name": _CONFIG_NAME,
            "user_id": user_id,
            "model": _MODEL_NAME,
        },
    )

    t0 = time.perf_counter()
    try:
        state = graph.invoke({
            "question": body.question,
            "top_k": body.top_k,
            "query_embedding": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "prompt": "",
            "answer": "",
        }, config=run_config)
    except Exception as e:
        logger.error("Pipeline error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    latency = time.perf_counter() - t0

    track_query_async(
        run_id=str(run_id),
        latency_seconds=latency,
        config_name=_CONFIG_NAME,
        model=_MODEL_NAME,
        user_id=user_id,
    )

    sources = [
        SourceResponse(**dataclasses.asdict(chunk))
        for chunk in state["reranked_chunks"]
    ]

    return QueryResponse(
        question=body.question,
        answer=state["answer"],
        sources=sources,
    )
