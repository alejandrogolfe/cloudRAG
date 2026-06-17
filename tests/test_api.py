from retrieval.models import RetrievedChunk

API_KEY = "test-api-key-12345"
AUTH = {"X-API-Key": API_KEY, "X-User-Id": "test-user"}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_no_api_key(client):
    response = client.post("/query", json={"question": "What is RAG?"}, headers={"X-User-Id": "test-user"})
    assert response.status_code == 403


def test_query_wrong_api_key(client):
    response = client.post("/query", json={"question": "What is RAG?"}, headers={"X-API-Key": "wrong-key", "X-User-Id": "test-user"})
    assert response.status_code == 403


def test_query_no_user_id(client):
    response = client.post("/query", json={"question": "What is RAG?"}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 422


def test_query_empty_question(client):
    response = client.post("/query", json={"question": "   "}, headers=AUTH)
    assert response.status_code == 400


def test_query_success(client, mock_graph):
    mock_graph.invoke.return_value = {
        "answer": "RAG stands for Retrieval-Augmented Generation.",
        "reranked_chunks": [],
        "retrieved_chunks": [],
        "query_embedding": [],
        "prompt": "",
    }
    response = client.post("/query", json={"question": "What is RAG?"}, headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is RAG?"
    assert data["answer"] == "RAG stands for Retrieval-Augmented Generation."
    assert "sources" in data


def test_query_response_includes_sources(client, mock_graph):
    chunk = RetrievedChunk(
        chunk_id="abc123",
        content="RAG combines retrieval with generation.",
        score=0.95,
        title="RAG Overview",
        source="docs/rag.md",
        url="https://example.com/rag",
    )
    mock_graph.invoke.return_value = {
        "answer": "RAG is a technique.",
        "reranked_chunks": [chunk],
        "retrieved_chunks": [chunk],
        "query_embedding": [],
        "prompt": "",
    }
    response = client.post("/query", json={"question": "What is RAG?"}, headers=AUTH)
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "abc123"
    assert sources[0]["score"] == 0.95
