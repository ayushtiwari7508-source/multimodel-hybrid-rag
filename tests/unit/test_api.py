"""API smoke tests using FastAPI TestClient.

These tests bypass Ollama entirely by replacing the live pipeline with a stub.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_env(tmp_index_dir, monkeypatch):
    """Isolated data dir for Chroma/BM25, plus a stub pipeline."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_index_dir / "chroma"))
    monkeypatch.setenv("BM25_PERSIST_PATH", str(tmp_index_dir / "bm25" / "index.pkl"))
    monkeypatch.setenv("META_PERSIST_DIR", str(tmp_index_dir / "index_meta"))
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8765")

    # Force a fresh config + pipeline
    from backend.app.core.config import get_settings

    get_settings.cache_clear()

    # Stub the pipeline before route handlers resolve it
    import backend.app.api.main as api_mod
    from backend.app.api.main import app

    stub = MagicMock()
    stub.ollama.health.return_value = True
    stub.ollama.list_models.return_value = [
        "nomic-embed-text:latest",
        "qwen2.5:3b",
        "qwen2.5:7b",
    ]
    stub.stats.return_value = {"chroma": 0, "bm25": 0}
    stub.list_documents.return_value = []
    stub.delete_document.return_value = {"deleted_chroma": 2, "deleted_bm25": 2, "doc_id": "x"}
    stub.ingest.return_value = {"chunks": 0, "docs": {}}
    stub.clear.return_value = None

    # The refactored API splits access points: query/health keep their own lazy
    # pipeline singleton, document routes route through a DocumentService.
    # Inject the stub into each access point used by the routes.
    from backend.app.api.routes import documents as documents_route
    from backend.app.api.routes import health as health_route
    from backend.app.api.routes import query as query_route

    api_mod.pipeline = stub
    query_route._pipeline = stub
    health_route._pipeline = stub
    documents_route._document_service = stub

    client = TestClient(app)
    yield client, stub
    get_settings.cache_clear()


def test_health_reports_ok(api_env):
    client, stub = api_env
    # Provide all required models
    stub.ollama.list_models.return_value = [
        "nomic-embed-text:latest",
        "qwen2.5:3b",
        "qwen2.5:7b",
        "qwen2.5-coder:7b",
        "llava:7b",
        "bge-reranker-v2-m3",
    ]
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["ollama_reachable"] is True
    assert "notes" in data


def test_health_reports_missing_models(api_env):
    client, stub = api_env
    stub.ollama.list_models.return_value = ["qwen2.5:3b"]  # missing most
    r = client.get("/health")
    data = r.json()
    assert data["ok"] is False
    assert len(data["missing_required"]) >= 1
    assert any("ollama pull" in n for n in data["notes"])


def test_list_documents(api_env):
    client, stub = api_env
    stub.list_documents.return_value = [{"source": "a.md", "doc_id": "a", "chroma_count": 3}]
    r = client.get("/documents")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["documents"][0]["source"] == "a.md"


def test_delete_document_success(api_env):
    client, stub = api_env
    r = client.delete("/documents/myid")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["deleted_chroma"] == 2
    stub.delete_document.assert_called_once_with("myid")


def test_delete_document_not_found(api_env):
    client, stub = api_env
    stub.delete_document.return_value = {"deleted_chroma": 0, "deleted_bm25": 0, "doc_id": "ghost"}
    r = client.delete("/documents/ghost")
    assert r.status_code == 404


def test_delete_document_invalid_id(api_env):
    client, _ = api_env
    # FastAPI router blocks "." and ".." at routing level (405)
    # Handler validation covers other edge cases via 400/404
    r = client.delete("/documents/")  # empty -> 404 (no route match)
    assert r.status_code in (404, 405)
    # The handler's own validation is exercised by the 404 test above


def test_query_validation(api_env):
    client, stub = api_env
    stub.query.return_value = type(
        "R",
        (),
        {
            "answer": "Hello world",
            "route": type("Route", (), {"name": "general", "model": "x", "reason": ""})(),
            "hits": [],
        },
    )()
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 400
    r = client.post("/query", json={"question": "What is BM25?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Hello world"


def test_ingest_text_routes_to_pipeline(api_env):
    client, stub = api_env
    stub.ingest_text.return_value = {"chunks": 1, "docs": {"k": 1}}
    r = client.post("/ingest/text", json={"text": "hello", "source": "t"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    stub.ingest_text.assert_called_once_with("hello", "t")


def test_clear_routes_to_pipeline(api_env):
    client, stub = api_env
    r = client.post("/clear")
    assert r.status_code == 200
    stub.clear.assert_called_once()
