"""
Chunk node: supports two strategies, selected via config.

fixed     → RecursiveCharacterTextSplitter with fixed size, no structure awareness
structure → splits by Markdown headers, recursive fallback if section too long

Both strategies produce Chunk objects with full metadata.
"""

import hashlib
import re
from typing import List, Tuple, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.state import IngestionState
from ingestion.models import Document, Chunk
from config.chunking import CHUNKING_STRATEGY, FIXED_SIZE, STRUCTURE

HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _make_chunk_id(source: str, index: int) -> str:
    return f"{hashlib.md5(source.encode()).hexdigest()[:8]}_{index:04d}"


# ── Fixed-size strategy ────────────────────────────────────────────────────────

def _chunk_fixed(doc: Document, start_index: int) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_SIZE["chunk_size"],
        chunk_overlap=FIXED_SIZE["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )
    texts = splitter.split_text(doc.content)
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.source, start_index + i),
            content=text.strip(),
            metadata={
                "source": doc.source,
                "title": doc.title,
                "url": doc.url,
                "strategy": "fixed",
                "chunk_size": FIXED_SIZE["chunk_size"],
                "chunk_overlap": FIXED_SIZE["chunk_overlap"],
                "chunk_index": start_index + i,
                "char_count": len(text.strip()),
            }
        ))
    return chunks


# ── Structure-aware strategy ───────────────────────────────────────────────────

def _extract_sections(content: str) -> List[Tuple[Dict, str]]:
    """Splits by Markdown headers. Returns list of (headers_dict, text)."""
    sections = []
    current_headers = {"header_1": None, "header_2": None, "header_3": None}
    last_end = 0
    matches = list(HEADER_PATTERN.finditer(content))

    if not matches:
        return [(dict(current_headers), content.strip())]

    for match in matches:
        if match.start() > last_end:
            text = content[last_end:match.start()].strip()
            if text:
                sections.append((dict(current_headers), text))

        level = len(match.group(1))
        header_text = match.group(2).strip()
        if level == 1:
            current_headers = {"header_1": header_text, "header_2": None, "header_3": None}
        elif level == 2:
            current_headers["header_2"] = header_text
            current_headers["header_3"] = None
        elif level == 3:
            current_headers["header_3"] = header_text

        last_end = match.end()

    remaining = content[last_end:].strip()
    if remaining:
        sections.append((dict(current_headers), remaining))

    return sections


def _split_if_too_long(text: str) -> List[str]:
    if len(text) <= STRUCTURE["max_chunk_chars"]:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=STRUCTURE["max_chunk_chars"],
        chunk_overlap=STRUCTURE["chunk_overlap_chars"],
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def _chunk_structure(doc: Document, start_index: int) -> List[Chunk]:
    sections = _extract_sections(doc.content)
    chunks = []
    i = 0
    for headers, section_text in sections:
        for sub_text in _split_if_too_long(section_text):
            if not sub_text.strip():
                continue
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc.source, start_index + i),
                content=sub_text.strip(),
                metadata={
                    "source": doc.source,
                    "title": doc.title,
                    "url": doc.url,
                    "strategy": "structure",
                    "header_1": headers.get("header_1"),
                    "header_2": headers.get("header_2"),
                    "header_3": headers.get("header_3"),
                    "chunk_index": start_index + i,
                    "char_count": len(sub_text.strip()),
                }
            ))
            i += 1
    return chunks


# ── Node ───────────────────────────────────────────────────────────────────────

def chunk_node(state: IngestionState) -> dict:
    strategy = CHUNKING_STRATEGY
    all_chunks: List[Chunk] = []
    global_index = 0

    for doc in state["cleaned_documents"]:
        if strategy == "fixed":
            doc_chunks = _chunk_fixed(doc, global_index)
        elif strategy == "structure":
            doc_chunks = _chunk_structure(doc, global_index)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Backfill total_chunks per document
        for chunk in doc_chunks:
            chunk.metadata["total_chunks"] = len(doc_chunks)

        print(f"[chunk/{strategy}] {doc.title} → {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)
        global_index += len(doc_chunks)

    print(f"[chunk] Total: {len(all_chunks)} chunks")
    return {"chunks": all_chunks}
