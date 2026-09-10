"""
Tests for the /query and /health endpoints.

Heavy dependencies (the embedding model, the Chroma vector store, and the
Ollama LLM call) are mocked out rather than loaded for real -- these tests
check that the API wiring, request validation, and response schema are
correct, not that the RAG pipeline itself produces good answers (that's
what notebooks/rag_pipeline.ipynb Phase 2.6 evaluates).
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Patch the embedder/collection getters BEFORE the app's lifespan runs,
    # so startup loading never touches the real model or a real vector store.
    from app.services import retrieval

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = MagicMock(tolist=lambda: [[0.0] * 384])

    fake_collection = MagicMock()
    fake_collection.count.return_value = 3
    fake_collection.query.return_value = {
        "ids": [["crewai_agents.md::0", "crewai_agents.md::1"]],
        "documents": [["Agents are created with the Agent class.", "Pass role, goal, and backstory."]],
        "metadatas": [[{"source": "crewai_agents.md"}, {"source": "crewai_agents.md"}]],
        "distances": [[0.1, 0.2]],
    }

    monkeypatch.setattr(retrieval, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(retrieval, "get_collection", lambda: fake_collection)

    # Patch the Ollama call so tests don't need a running local Ollama server.
    from app.services import generation

    fake_ollama_response = {"response": "You can create an agent using the Agent class. [1]"}
    monkeypatch.setattr(generation.ollama, "generate", lambda model, prompt: fake_ollama_response)

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_happy_path(client):
    response = client.post("/query", json={"question": "How do I create an agent in CrewAI?"})
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert data["answer"] == "You can create an agent using the Agent class. [1]"
    assert data["sources"] == ["crewai_agents.md", "crewai_agents.md"]


def test_query_invalid_input(client):
    # Missing the required "question" field entirely
    response = client.post("/query", json={})
    assert response.status_code == 422

    # Empty string also violates min_length=1
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
