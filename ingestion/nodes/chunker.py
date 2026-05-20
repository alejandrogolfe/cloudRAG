"""
Chunk node: supports five strategies, selected via config.

fixed          → RecursiveCharacterTextSplitter with fixed size
structure      → splits by Markdown headers, recursive fallback if too long
semantic       → splits by semantic similarity between consecutive sentences
sentence_window → indexes sentences with surrounding context window
parent_child   → small child chunks for retrieval, large parent chunks for context

All strategies produce Chunk objects with full metadata.
"""

import hashlib
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.state import IngestionState
from ingestion.models import Document, Chunk
from config.chunking import (
    CHUNKING_STRATEGY,
    FIXED_SIZE,
    STRUCTURE,
    SEMANTIC,
    SENTENCE_WINDOW,
    PARENT_CHILD,
    EMBEDDING,
)

HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')

openai_client = OpenAI()


def _make_chunk_id(source: str, index: int) -> str:
    return f"{hashlib.md5(source.encode()).hexdigest()[:8]}_{index:04d}"


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch embed texts using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING["model_id"],
        input=texts,
    )
    return [item.embedding for item in response.data]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


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


# ── Semantic strategy ──────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering very short ones."""
    raw = SENTENCE_PATTERN.split(text)
    return [s.strip() for s in raw if len(s.strip()) >= SEMANTIC["min_chunk_chars"] // 3]


def _chunk_semantic(doc: Document, start_index: int) -> List[Chunk]:
    """
    Splits by semantic similarity between consecutive sentences.
    When similarity drops below threshold, starts a new chunk.
    Embeddings are computed in batches to avoid too many API calls.
    """
    sentences = _split_sentences(doc.content)
    if not sentences:
        return []

    # Embed all sentences in one batch
    embeddings = _embed_texts(sentences)

    # Group sentences into chunks based on semantic similarity
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]
    chunks = []
    i = 0

    for j in range(1, len(sentences)):
        sim = _cosine_similarity(current_embedding, embeddings[j])

        # Start new chunk if similarity drops OR current chunk is too long
        current_text = " ".join(current_sentences)
        if sim < SEMANTIC["similarity_threshold"] or len(current_text) > SEMANTIC["max_chunk_chars"]:
            if len(current_text) >= SEMANTIC["min_chunk_chars"]:
                chunks.append(Chunk(
                    chunk_id=_make_chunk_id(doc.source, start_index + i),
                    content=current_text.strip(),
                    metadata={
                        "source": doc.source,
                        "title": doc.title,
                        "url": doc.url,
                        "strategy": "semantic",
                        "similarity_threshold": SEMANTIC["similarity_threshold"],
                        "chunk_index": start_index + i,
                        "char_count": len(current_text.strip()),
                    }
                ))
                i += 1
            current_sentences = [sentences[j]]
            current_embedding = embeddings[j]
        else:
            current_sentences.append(sentences[j])
            # Update running average embedding for the current chunk
            current_embedding = list(np.mean([current_embedding, embeddings[j]], axis=0))

    # Don't forget the last chunk
    remaining = " ".join(current_sentences).strip()
    if remaining and len(remaining) >= SEMANTIC["min_chunk_chars"]:
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.source, start_index + i),
            content=remaining,
            metadata={
                "source": doc.source,
                "title": doc.title,
                "url": doc.url,
                "strategy": "semantic",
                "similarity_threshold": SEMANTIC["similarity_threshold"],
                "chunk_index": start_index + i,
                "char_count": len(remaining),
            }
        ))

    return chunks


# ── Sentence window strategy ───────────────────────────────────────────────────

def _chunk_sentence_window(doc: Document, start_index: int) -> List[Chunk]:
    """
    Each chunk is a single sentence but includes surrounding context.
    The 'content' field contains only the sentence (indexed for retrieval).
    The 'window_content' metadata field contains the full context window
    (passed to GPT-4o during augmentation).
    """
    sentences = [
        s.strip() for s in SENTENCE_PATTERN.split(doc.content)
        if len(s.strip()) >= SENTENCE_WINDOW["min_sentence_chars"]
    ]

    if not sentences:
        return []

    window = SENTENCE_WINDOW["window_size"]
    chunks = []

    for i, sentence in enumerate(sentences):
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])

        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.source, start_index + i),
            content=sentence,
            metadata={
                "source": doc.source,
                "title": doc.title,
                "url": doc.url,
                "strategy": "sentence_window",
                "window_content": context,   # full context for GPT-4o
                "window_size": window,
                "chunk_index": start_index + i,
                "char_count": len(sentence),
            }
        ))

    return chunks


# ── Parent-child strategy ──────────────────────────────────────────────────────

def _chunk_parent_child(doc: Document, start_index: int) -> List[Chunk]:
    """
    Creates two levels of chunks:
    - Parent: larger chunks for context (stored in metadata, not indexed)
    - Child: smaller chunks indexed for retrieval, reference their parent

    The retriever returns child chunks. The augmenter should use
    child.metadata['parent_content'] as context instead of child.content.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHILD["parent_chunk_size"],
        chunk_overlap=PARENT_CHILD["parent_chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHILD["child_chunk_size"],
        chunk_overlap=PARENT_CHILD["child_chunk_overlap"],
        separators=["\n\n", "\n", ". ", " "],
    )

    parent_texts = parent_splitter.split_text(doc.content)
    chunks = []
    i = 0

    for p_idx, parent_text in enumerate(parent_texts):
        parent_id = _make_chunk_id(doc.source, start_index + p_idx * 1000)
        child_texts = child_splitter.split_text(parent_text)

        for child_text in child_texts:
            if not child_text.strip():
                continue
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc.source, start_index + i),
                content=child_text.strip(),       # indexed for retrieval
                metadata={
                    "source": doc.source,
                    "title": doc.title,
                    "url": doc.url,
                    "strategy": "parent_child",
                    "parent_id": parent_id,
                    "parent_content": parent_text.strip(),  # context for GPT-4o
                    "chunk_index": start_index + i,
                    "char_count": len(child_text.strip()),
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
        elif strategy == "semantic":
            doc_chunks = _chunk_semantic(doc, global_index)
        elif strategy == "sentence_window":
            doc_chunks = _chunk_sentence_window(doc, global_index)
        elif strategy == "parent_child":
            doc_chunks = _chunk_parent_child(doc, global_index)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for chunk in doc_chunks:
            chunk.metadata["total_chunks"] = len(doc_chunks)

        print(f"[chunk/{strategy}] {doc.title} → {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)
        global_index += len(doc_chunks)

    print(f"[chunk] Total: {len(all_chunks)} chunks")
    return {"chunks": all_chunks}
