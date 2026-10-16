"""Tests for per-document deletion and listing."""

from __future__ import annotations

import pytest

from app.storage import BM25Store


def test_bm25_delete_where_removes_matching_chunks(tmp_index_dir):
    store = BM25Store(str(tmp_index_dir / "bm25" / "index.pkl"))
    store.add(
        [
            {"id": "1", "content": "alpha", "meta": {"doc_id": "A"}},
            {"id": "2", "content": "beta", "meta": {"doc_id": "A"}},
            {"id": "3", "content": "gamma", "meta": {"doc_id": "B"}},
        ]
    )
    removed = store.delete_where({"doc_id": "A"})
    assert removed == 2
    remaining = [c["id"] for c in store.corpus]
    assert remaining == ["3"]
    # bm25 should still be queryable
    assert any(rec["id"] == "3" for rec, _ in store.search("gamma", n=5))


def test_bm25_delete_where_no_match_returns_zero(tmp_index_dir):
    store = BM25Store(str(tmp_index_dir / "bm25" / "index.pkl"))
    store.add([{"id": "1", "content": "x", "meta": {"doc_id": "A"}}])
    assert store.delete_where({"doc_id": "ghost"}) == 0


def test_bm25_delete_where_persists(tmp_index_dir):
    p = str(tmp_index_dir / "bm25" / "index.pkl")
    store = BM25Store(p)
    store.add(
        [
            {"id": "1", "content": "alpha", "meta": {"doc_id": "A"}},
            {"id": "2", "content": "beta", "meta": {"doc_id": "B"}},
        ]
    )
    store.delete_where({"doc_id": "A"})
    store2 = BM25Store(p)
    assert [c["id"] for c in store2.corpus] == ["2"]


def test_bm25_list_distinct(tmp_index_dir):
    store = BM25Store(str(tmp_index_dir / "bm25" / "index.pkl"))
    store.add(
        [
            {"id": "1", "content": "x", "meta": {"doc_id": "A"}},
            {"id": "2", "content": "y", "meta": {"doc_id": "A"}},
            {"id": "3", "content": "z", "meta": {"doc_id": "B"}},
            {"id": "4", "content": "w", "meta": {"source": "/path/c.md"}},
        ]
    )
    assert store.list_distinct("doc_id") == ["A", "B"]
    assert store.list_distinct("source") == ["/path/c.md"]


def test_chroma_delete_where_removes_matching(tmp_index_dir):
    """Real Chroma round-trip for delete_where."""
    pytest.importorskip("chromadb")
    from app.storage import ChromaStore

    store = ChromaStore(str(tmp_index_dir / "chroma"))
    embs_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    embs_b = [[0.0, 0.0, 1.0]]
    store.add(
        ids=["a1", "a2"],
        embeddings=embs_a,
        documents=["doc a one", "doc a two"],
        metadatas=[{"doc_id": "A", "source": "a.md"}, {"doc_id": "A", "source": "a.md"}],
    )
    store.add(
        ids=["b1"],
        embeddings=embs_b,
        documents=["doc b one"],
        metadatas=[{"doc_id": "B", "source": "b.md"}],
    )
    before = store.count()
    removed = store.delete_where({"doc_id": "A"})
    after = store.count()
    assert removed == before - after == 2
    assert after == 1


def test_chroma_list_distinct(tmp_index_dir):
    pytest.importorskip("chromadb")
    from app.storage import ChromaStore

    store = ChromaStore(str(tmp_index_dir / "chroma"))
    store.add(
        ids=["1", "2", "3"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        documents=["a", "b", "c"],
        metadatas=[{"source": "x.md"}, {"source": "x.md"}, {"source": "y.md"}],
    )
    assert set(store.list_distinct("source")) == {"x.md", "y.md"}
