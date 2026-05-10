from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float
    title: str = ""
    source: str = ""
    url: str = ""
    header_1: Optional[str] = None
    header_2: Optional[str] = None
    header_3: Optional[str] = None


@dataclass
class QueryRequest:
    question: str
    top_k: int = 5  # number of chunks to retrieve


@dataclass
class QueryResponse:
    answer: str
    sources: List[RetrievedChunk]
    question: str
