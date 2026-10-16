"""Tests for app/reranker.py and retriever rerank dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings
from app.reranker import CrossEncoderReranker, OllamaReranker
from app.retriever import Hit, HybridRetriever


@pytest.fixture(autouse=True)
def _reset_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_cross_encoder_returns_empty_for_no_documents():
    """score() with no documents returns [] and never touches the model."""
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    assert reranker.score("query", []) == []
    assert reranker._model is None  # lazy: never loaded


def test_cross_encoder_lazy_loads_model():
    """First score() call invokes CrossEncoder() exactly once."""
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    fake_model = MagicMock()
    fake_model.predict.side_effect = [[0.9, 0.1, 0.5], [0.5]]

    with patch("sentence_transformers.CrossEncoder", return_value=fake_model) as ce_cls:
        scores1 = reranker.score("q", ["d1", "d2", "d3"])
        scores2 = reranker.score("q", ["d4"])

    assert ce_cls.call_count == 1, "model must be loaded once and cached"
    assert scores1 == [0.9, 0.1, 0.5]
    assert scores2 == [0.5]  # single-doc call


def test_ollama_reranker_delegates_to_client():
    fake_client = MagicMock()
    fake_client.rerank.return_value = [0.7, 0.3]
    reranker = OllamaReranker(fake_client, "bge-reranker-v2-m3")
    out = reranker.score("q", ["a", "b"])
    assert out == [0.7, 0.3]
    fake_client.rerank.assert_called_once_with(query="q", documents=["a", "b"], model="bge-reranker-v2-m3")


def test_ollama_reranker_empty_input():
    fake_client = MagicMock()
    reranker = OllamaReranker(fake_client, "x")
    assert reranker.score("q", []) == []
    fake_client.rerank.assert_not_called()


def test_retriever_dispatches_to_cross_encoder(monkeypatch, tmp_index_dir):
    """When RERANK_BACKEND=cross-encoder, retriever uses CrossEncoderReranker."""
    monkeypatch.setenv("RERANK_BACKEND", "cross-encoder")
    monkeypatch.setenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    get_settings.cache_clear()

    from app.ollama_client import OllamaClient
    from app.storage import BM25Store, ChromaStore

    ce = MagicMock()
    ce.score.return_value = [0.9, 0.5, 0.2]
    with patch("app.retriever.CrossEncoderReranker", return_value=ce) as ce_cls:
        hr = HybridRetriever(
            chroma=ChromaStore(),
            bm25=BM25Store(),
            ollama=OllamaClient(),
        )
    ce_cls.assert_called_once_with("BAAI/bge-reranker-base")
    assert hr.reranker is ce

    hits = [
        Hit(id="a", content="apple", meta={}, rrf_score=0.01),
        Hit(id="b", content="banana", meta={}, rrf_score=0.02),
        Hit(id="c", content="cherry", meta={}, rrf_score=0.03),
    ]
    # RRF alone would sort c > b > a, but rerank overrides to apple > cherry > banana
    ce.score.return_value = [0.9, 0.1, 0.5]
    out, _ = hr._rerank("fruit", hits, top_k=3)
    ce.score.assert_called_once_with("fruit", ["apple", "banana", "cherry"])
    # Final blend: 0.6*rrf + 0.4*normalized_rerank
    assert [h.id for h in out[:3]] == ["a", "c", "b"]


def test_retriever_falls_back_when_reranker_raises(monkeypatch, tmp_index_dir):
    """Reranker errors fall back to weighted blend using RRF scores."""
    monkeypatch.setenv("RERANK_BACKEND", "cross-encoder")
    get_settings.cache_clear()

    from app.ollama_client import OllamaClient
    from app.storage import BM25Store, ChromaStore

    bad = MagicMock()
    bad.score.side_effect = RuntimeError("model crashed")
    with patch("app.retriever.CrossEncoderReranker", return_value=bad):
        hr = HybridRetriever(
            chroma=ChromaStore(),
            bm25=BM25Store(),
            ollama=OllamaClient(),
        )

    hits = [
        Hit(id="a", content="a", meta={}, rrf_score=0.5),
        Hit(id="b", content="b", meta={}, rrf_score=0.9),
    ]
    out, _ = hr._rerank("q", hits, top_k=3)
    # Falls back to rrf_score ordering -> b first
    assert out[0].id == "b"
    assert out[1].id == "a"


def test_retriever_dispatch_to_ollama(monkeypatch, tmp_index_dir):
    monkeypatch.setenv("RERANK_BACKEND", "ollama")
    monkeypatch.setenv("RERANK_MODEL", "bge-reranker-v2-m3")
    get_settings.cache_clear()

    from app.ollama_client import OllamaClient
    from app.storage import BM25Store, ChromaStore

    ollama_rerank = MagicMock()
    ollama_rerank.score.return_value = [0.1, 0.8]
    with patch("app.retriever.OllamaReranker", return_value=ollama_rerank) as or_cls:
        hr = HybridRetriever(
            chroma=ChromaStore(),
            bm25=BM25Store(),
            ollama=OllamaClient(),
        )
    or_cls.assert_called_once()
    assert hr.reranker is ollama_rerank

    hits = [Hit(id="a", content="x", meta={}), Hit(id="b", content="y", meta={})]
    hr._rerank("q", hits, top_k=2)
    ollama_rerank.score.assert_called_once_with("q", ["x", "y"])


def test_pipeline_skips_ollama_rerank_check_when_cross_encoder(monkeypatch, tmp_index_dir):
    """Pipeline health shouldn't complain about an Ollama rerank model tag."""
    monkeypatch.setenv("RERANK_BACKEND", "cross-encoder")
    monkeypatch.setenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    get_settings.cache_clear()

    from app.pipeline import RAGPipeline

    p = RAGPipeline()
    p._model_status = {}  # force recheck
    p._models_checked = False
    _ok, missing = p._check_models()
    # The cross-encoder HF tag is intentionally NOT checked against Ollama
    assert "rerank" not in p._model_status
    # Rerank backend is independent of required/optional Ollama models
    assert all(m not in {"rerank"} for m in missing)
    p.close()
