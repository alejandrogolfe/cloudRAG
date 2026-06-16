import os
from unittest.mock import patch, MagicMock

# Set env vars before any app imports
os.environ.update({
    "API_KEY": "test-api-key-12345",
    "RATE_LIMIT": "1000/minute",
    "AWS_REGION": "eu-west-1",
    "OPENSEARCH_ENDPOINT": "test.aoss.amazonaws.com",
    "OPENSEARCH_INDEX": "test-index",
    "OPENAI_API_KEY": "sk-" + "a" * 48,
    "COHERE_API_KEY": "test-cohere-fake",
    "LANGCHAIN_TRACING_V2": "false",
})

import pytest
from fastapi.testclient import TestClient

_MOCK_GRAPH = MagicMock()
_MOCK_GRAPH.invoke.return_value = {
    "answer": "This is a test answer.",
    "reranked_chunks": [],
    "retrieved_chunks": [],
    "query_embedding": [],
    "prompt": "",
}

# Patch before retrieval.api is imported so graph = build_retrieval_graph() uses the mock
_patcher = patch("retrieval.graph.build_retrieval_graph", return_value=_MOCK_GRAPH)
_patcher.start()

from retrieval.api import app as _app


@pytest.fixture(scope="session")
def client():
    return TestClient(_app)


@pytest.fixture(scope="session")
def mock_graph():
    return _MOCK_GRAPH
