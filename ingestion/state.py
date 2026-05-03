from typing import TypedDict, List
from ingestion.models import Document, Chunk


class IngestionState(TypedDict):
    docs_path: str
    strategy: str                      # "fixed" | "structure"

    raw_documents: List[Document]
    filtered_documents: List[Document]
    filtered_out: List[Document]
    cleaned_documents: List[Document]
    chunks: List[Chunk]
    embedded_chunks: List[Chunk]       # chunks with embeddings (filled by embed node)
