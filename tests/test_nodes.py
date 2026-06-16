from unittest.mock import patch
import pytest

from ingestion.models import Document
from ingestion.nodes.filter import filter_node, _is_noisy
from retrieval.models import RetrievedChunk
from retrieval.nodes.augmenter import augment_node
from retrieval.nodes.reranker import rerank_node
import config.retrieval as retrieval_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc(title="Valid Title", content="x" * 300):
    return Document(source="test.md", title=title, url="https://example.com", content=content)


def _chunk(content="chunk content", score=0.9, **kwargs):
    return RetrievedChunk(
        chunk_id="test_id",
        content=content,
        score=score,
        title="Test Title",
        source="test.md",
        url="https://example.com",
        **kwargs,
    )


# ── filter_node ───────────────────────────────────────────────────────────────

def test_filter_noisy_title():
    assert _is_noisy(_doc(title="Category: Something")) is True


def test_filter_noisy_content():
    assert _is_noisy(_doc(content="Jump to navigation " + "x" * 300)) is True


def test_filter_short_content():
    assert _is_noisy(_doc(content="Too short")) is True


def test_filter_keeps_valid_doc():
    assert _is_noisy(_doc()) is False


def test_filter_node_splits_correctly():
    good = _doc()
    bad = _doc(title="Category: Test")
    result = filter_node({"raw_documents": [good, bad]})
    assert len(result["filtered_documents"]) == 1
    assert len(result["filtered_out"]) == 1
    assert result["filtered_documents"][0] is good


# ── augment_node ──────────────────────────────────────────────────────────────

def test_augment_prompt_contains_question():
    result = augment_node({"question": "What is RAG?", "reranked_chunks": [_chunk()]})
    assert "What is RAG?" in result["prompt"]


def test_augment_prompt_contains_chunk_content():
    result = augment_node({"question": "test", "reranked_chunks": [_chunk(content="important content here")]})
    assert "important content here" in result["prompt"]


def test_augment_uses_parent_content():
    chunk = _chunk(metadata={"parent_content": "parent context text"})
    result = augment_node({"question": "test", "reranked_chunks": [chunk]})
    assert "parent context text" in result["prompt"]


def test_augment_uses_window_content():
    chunk = _chunk(metadata={"window_content": "window context text"})
    result = augment_node({"question": "test", "reranked_chunks": [chunk]})
    assert "window context text" in result["prompt"]


def test_augment_truncates_when_over_token_limit(monkeypatch):
    import retrieval.nodes.augmenter as aug
    monkeypatch.setattr(aug, "_MAX_PROMPT_TOKENS", 10)
    chunks = [_chunk(content=f"content for chunk {i}") for i in range(5)]
    result = augment_node({"question": "test", "reranked_chunks": chunks})
    # With a 10-token limit only 1 chunk should remain
    assert result["prompt"].count("[1]") == 1
    assert result["prompt"].count("[2]") == 0


# ── rerank_node ───────────────────────────────────────────────────────────────

def test_rerank_disabled_returns_top_k(monkeypatch):
    import retrieval.nodes.reranker as reranker_module
    monkeypatch.setattr(reranker_module, "RERANKING_ENABLED", False)
    monkeypatch.setattr(reranker_module, "RETRIEVAL_TOP_K_FINAL", 2)
    candidates = [_chunk(score=0.9), _chunk(score=0.8), _chunk(score=0.7)]
    result = rerank_node({"question": "test", "retrieved_chunks": candidates})
    assert len(result["reranked_chunks"]) == 2
    assert result["reranked_chunks"][0].score == 0.9


def test_rerank_cohere_fallback(monkeypatch):
    import retrieval.nodes.reranker as reranker_module
    monkeypatch.setattr(reranker_module, "RERANKING_ENABLED", True)
    monkeypatch.setattr(reranker_module, "RETRIEVAL_TOP_K_FINAL", 2)
    with patch("retrieval.nodes.reranker._call_cohere_rerank", side_effect=Exception("Cohere down")):
        candidates = [_chunk(score=0.9), _chunk(score=0.8), _chunk(score=0.7)]
        result = rerank_node({"question": "test", "retrieved_chunks": candidates})
    assert len(result["reranked_chunks"]) == 2
    assert result["reranked_chunks"][0].score == 0.9
