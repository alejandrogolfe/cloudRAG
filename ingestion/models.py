from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Document:
    source: str
    title: str
    url: str
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)
