from dataclasses import dataclass, field
from typing import List, Optional, Dict


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
    metadata: Dict = field(default_factory=dict)  # strategy-specific fields (parent_content, window_content)


@dataclass
class QueryRequest:
    question: str
    top_k: int = 5


@dataclass
class QueryResponse:
    answer: str
    sources: List[RetrievedChunk]
    question: str
