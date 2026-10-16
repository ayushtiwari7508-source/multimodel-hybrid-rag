"""Integration test: full pipeline against a real Ollama instance.

Skipped automatically if Ollama is not reachable or required models aren't pulled.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from app.pipeline import RAGPipeline

REQUIRED = {"nomic-embed-text", "qwen2.5:3b", "qwen2.5:7b"}


@pytest.fixture(scope="module")
def live_pipeline():
    """Build a pipeline if Ollama + minimum models are available."""
    p = RAGPipeline()
    if not p.ollama.health():
        pytest.skip("Ollama not reachable")
    have = set(p.ollama.list_models())
    if not any(m == t or m.startswith(t + ":") for t in REQUIRED for m in have):
        pytest.skip(f"Missing one of required models: {REQUIRED}")
    # Use a clean tmp data dir for isolation
    tmp = Path(tempfile.mkdtemp())
    p.cfg.chroma_persist_dir = str(tmp / "chroma")
    p.cfg.bm25_persist_path = str(tmp / "bm25" / "index.pkl")
    p.cfg.meta_persist_dir = str(tmp / "index_meta")
    for d in [p.cfg.chroma_persist_dir, str(Path(p.cfg.bm25_persist_path).parent), p.cfg.meta_persist_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    # Recreate stores with new paths
    from app.storage import BM25Store, ChromaStore

    p.chroma = ChromaStore(p.cfg.chroma_persist_dir)
    p.bm25 = BM25Store(p.cfg.bm25_persist_path)
    p.retriever.chroma = p.chroma
    p.retriever.bm25 = p.bm25
    p.ingestor.chroma = p.chroma
    p.ingestor.bm25 = p.bm25
    p.clear()
    yield p
    p.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_ingest_and_retrieve(live_pipeline):
    doc = Path(tempfile.mkdtemp()) / "kb.md"
    doc.write_text(
        "# Knowledge Base\n\n"
        "RAG systems use retrievers and generators. "
        "BM25 is a sparse retrieval algorithm, "
        "while dense embeddings use neural encoders. "
        "Cross-encoders rerank for higher precision.\n",
        encoding="utf-8",
    )
    res = live_pipeline.ingest(paths=[str(doc)])
    assert res["chunks"] >= 1
    assert live_pipeline.chroma.count() >= 1


def test_router_classifies_short_query(live_pipeline):
    route = live_pipeline.router.route("hi")
    assert route.name == "fast"


def test_router_classifies_code_query(live_pipeline):
    route = live_pipeline.router.route("write a python function to compute fibonacci")
    assert route.name == "code"


def test_query_returns_citations_and_route(live_pipeline):
    # Ingest test data first
    doc = Path(tempfile.mkdtemp()) / "kb.md"
    doc.write_text(
        "# Knowledge Base\n\n"
        "RAG systems use retrievers and generators. "
        "BM25 is a sparse retrieval algorithm. "
        "Dense embeddings use neural encoders. "
        "Cross-encoders rerank for higher precision.\n",
        encoding="utf-8",
    )
    live_pipeline.ingest(paths=[str(doc)])

    res = live_pipeline.query("Explain how BM25 works in detail")
    assert res.route.name in {"general", "multimodal", "fast", "code", "math"}
    assert res.answer
    assert "[1]" in res.answer or "Sources" in res.answer
    assert res.hits


def test_query_returns_no_hits_on_empty_index(live_pipeline):
    live_pipeline.clear()
    res = live_pipeline.query("anything")
    # Empty index => hits may be empty, answer should still be returned
    assert res.answer
    assert res.route is not None
