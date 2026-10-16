"""Tests for API key auth, rate limiting, metrics, and request IDs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_env_auth(tmp_index_dir, monkeypatch):
    """Set up API with auth enabled."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_index_dir / "chroma"))
    monkeypatch.setenv("BM25_PERSIST_PATH", str(tmp_index_dir / "bm25" / "index.pkl"))
    monkeypatch.setenv("META_PERSIST_DIR", str(tmp_index_dir / "index_meta"))
    monkeypatch.setenv("API_KEY", "secret-key-123")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "5")

    from app.config import get_settings

    get_settings.cache_clear()

    import app.api as api_mod
    from app.api import app

    stub = MagicMock()
    stub.ollama.health.return_value = True
    stub.ollama.list_models.return_value = [
        "nomic-embed-text:latest",
        "qwen2.5:3b",
        "qwen2.5:7b",
        "qwen2.5-coder:7b",
        "llava:7b",
        "bge-reranker-v2-m3",
    ]
    stub.stats.return_value = {"chroma": 0, "bm25": 0}
    stub.list_documents.return_value = []
    stub.delete_document.return_value = {"deleted_chroma": 2, "deleted_bm25": 2, "doc_id": "x"}
    stub.ingest.return_value = {"chunks": 0, "docs": {}}
    stub.clear.return_value = None

    api_mod.pipeline = stub
    api_mod._rate_limiter = api_mod.RateLimiter(5)

    # Reset metrics
    api_mod.metrics.reset()

    client = TestClient(app)
    yield client, stub
    get_settings.cache_clear()


def test_health_does_not_require_auth(api_env_auth):
    """Health endpoint is public for monitoring."""
    client, _ = api_env_auth
    r = client.get("/health")
    assert r.status_code == 200


def test_metrics_does_not_require_auth(api_env_auth):
    """Metrics endpoint is public for Prometheus scraping."""
    client, _ = api_env_auth
    r = client.get("/metrics")
    assert r.status_code == 200


def test_query_requires_api_key(api_env_auth):
    """Protected endpoints reject missing API key."""
    client, _ = api_env_auth
    r = client.post("/query", json={"question": "test"})
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


def test_query_rejects_invalid_api_key(api_env_auth):
    client, _ = api_env_auth
    r = client.post("/query", json={"question": "test"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_query_accepts_valid_api_key(api_env_auth):
    client, stub = api_env_auth
    stub.query.return_value = MagicMock(
        answer="Hello", route=type("Route", (), {"name": "general", "model": "x", "reason": ""})(), hits=[]
    )
    r = client.post("/query", json={"question": "test"}, headers={"X-API-Key": "secret-key-123"})
    assert r.status_code == 200


def test_rate_limit_blocks_excess_requests(api_env_auth):
    """After N requests, additional requests get 429."""
    client, stub = api_env_auth
    stub.query.return_value = MagicMock(
        answer="x", route=type("Route", (), {"name": "general", "model": "x", "reason": ""})(), hits=[]
    )
    headers = {"X-API-Key": "secret-key-123"}
    # Allow 5 requests
    for i in range(5):
        r = client.post("/query", json={"question": f"q{i}"}, headers=headers)
        assert r.status_code == 200, f"request {i} got {r.status_code}"
    # 6th should be blocked
    r = client.post("/query", json={"question": "blocked"}, headers=headers)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_request_id_in_response_headers(api_env_auth):
    client, _ = api_env_auth
    r = client.get("/health")
    assert "x-request-id" in r.headers
    rid = r.headers["x-request-id"]
    assert len(rid) >= 8


def test_request_id_echoes_client_header(api_env_auth):
    client, _ = api_env_auth
    r = client.get("/health", headers={"X-Request-ID": "client-supplied-id"})
    assert r.headers["x-request-id"] == "client-supplied-id"


def test_metrics_endpoint_returns_counters(api_env_auth):
    client, _ = api_env_auth
    # Make a few requests
    client.get("/health")
    client.get("/health")
    r = client.get("/metrics")
    data = r.json()
    assert "metrics" in data
    assert "counters" in data["metrics"]
    assert "timings" in data["metrics"]
    # http.requests should have been incremented
    http_reqs = [k for k in data["metrics"]["counters"] if "http.requests" in k]
    assert len(http_reqs) > 0


def test_health_endpoint_includes_request_id():
    """Verify request_id is in error responses too."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./data/chroma")
    monkeypatch.setenv("BM25_PERSIST_PATH", "./data/bm25/index.pkl")
    monkeypatch.setenv("META_PERSIST_DIR", "./data/index_meta")
    monkeypatch.setenv("API_KEY", "")

    from app.config import get_settings

    get_settings.cache_clear()

    import app.api as api_mod
    from app.api import app

    stub = MagicMock()
    stub.ollama.health.return_value = True
    stub.ollama.list_models.return_value = ["nomic-embed-text:latest", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5-coder:7b"]
    stub.stats.return_value = {"chroma": 0, "bm25": 0}
    api_mod.pipeline = stub
    api_mod.metrics.reset()

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    monkeypatch.undo()
    get_settings.cache_clear()
