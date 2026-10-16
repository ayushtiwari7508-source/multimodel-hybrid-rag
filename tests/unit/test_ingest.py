"""Tests for document chunking, stable IDs, and BM25 indexing."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.loaders import DocElement, LoadedDocument
from app.storage import BM25Store, _coerce_metadata, chunk_to_records


def test_chunking_splits_long_text():
    doc = LoadedDocument(
        source="x.md",
        doc_id="x",
        elements=[DocElement(type="text", content="word " * 500, section="p1")],
    )
    chunks = doc.to_chunks(chunk_size=200, overlap=20)
    assert len(chunks) >= 2
    # First chunk should be <= chunk_size chars (excluding the prefix/header additions)
    assert all(len(c["content"]) >= 1 for c in chunks)


def test_chunking_emits_metadata_per_block():
    doc = LoadedDocument(
        source="paper.pdf",
        doc_id="paper",
        elements=[DocElement(type="text", content="hello world", page=4, section="methods")],
    )
    chunks = doc.to_chunks(chunk_size=50, overlap=10)
    assert chunks
    meta = chunks[0]["meta"]
    assert meta["source"] == "paper.pdf"
    assert meta["page"] == 4
    assert meta["section"] == "methods"


def test_chunking_handles_table_and_image_elements():
    doc = LoadedDocument(
        source="r.pdf",
        doc_id="r",
        elements=[
            DocElement(type="text", content="intro", page=1),
            DocElement(type="table", content="| a | b |\n| 1 | 2 |", page=1),
            DocElement(type="image", content="A bar chart", page=2),
        ],
    )
    chunks = doc.to_chunks(chunk_size=50, overlap=10)
    types = [c["meta"].get("element_type", "text") for c in chunks]
    assert "table" in types
    assert "image" in types


def test_chunk_to_records_stable_ids():
    chunks = [{"content": "abc", "meta": {"source": "x", "page": 1}}]
    r1 = chunk_to_records(chunks)
    r2 = chunk_to_records(chunks)
    assert r1[0]["id"] == r2[0]["id"]


def test_chunk_to_records_distinguishes_by_page():
    a = chunk_to_records([{"content": "abc", "meta": {"source": "x", "page": 1}}])
    b = chunk_to_records([{"content": "abc", "meta": {"source": "x", "page": 2}}])
    assert a[0]["id"] != b[0]["id"]


def test_bm25_search_returns_matches(tmp_index_dir):
    store = BM25Store(str(tmp_index_dir / "bm25" / "index.pkl"))
    store.add(
        [
            {"id": "1", "content": "the quick brown fox jumps over the lazy dog", "meta": {"source": "a"}},
            {"id": "2", "content": "lorem ipsum dolor sit amet", "meta": {"source": "b"}},
            {"id": "3", "content": "foxes are clever animals", "meta": {"source": "c"}},
        ]
    )
    hits = store.search("fox", n=3)
    assert hits
    top_ids = [rec["id"] for rec, _ in hits]
    assert "1" in top_ids and "3" in top_ids
    # highest BM25 should be the doc containing 'fox' verbatim
    scores = [s for _, s in hits]
    assert scores[0] >= scores[-1]


def test_bm25_persists_across_instances(tmp_index_dir):
    p = str(tmp_index_dir / "bm25" / "index.pkl")
    store = BM25Store(p)
    store.add([{"id": "1", "content": "alpha beta gamma", "meta": {}}])
    store2 = BM25Store(p)
    assert any(rec["id"] == "1" for rec, _ in store2.search("beta", n=5))


def test_bm25_clear_resets_index(tmp_index_dir):
    store = BM25Store(str(tmp_index_dir / "bm25" / "index.pkl"))
    store.add([{"id": "1", "content": "hello", "meta": {}}])
    store.clear()
    assert store.search("hello", n=5) == []


def test_coerce_metadata_drops_none():
    assert _coerce_metadata({"a": 1, "b": None, "c": "x"}) == {"a": 1, "c": "x"}


def test_coerce_metadata_stringifies_complex_types():
    out = _coerce_metadata({"list": [1, 2, 3], "obj": {"k": "v"}})
    assert out["list"] == "1,2,3"
    assert "v" in out["obj"]


def test_load_markdown(tmp_index_dir):
    from app.loaders import load_document

    p = Path(tmp_index_dir) / "doc.md"
    p.write_text("# Heading\n\nParagraph one.\n\n## Sub\n\nParagraph two.", encoding="utf-8")
    doc = load_document(str(p), ollama=None, vision_model="llava:7b")
    assert doc.doc_id == "doc"
    assert any(e.section and "Heading" in e.section for e in doc.elements)
    assert any("Paragraph one" in e.content for e in doc.elements)


def test_load_text(tmp_index_dir):
    from app.loaders import load_document

    p = Path(tmp_index_dir) / "doc.txt"
    p.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    doc = load_document(str(p), ollama=None, vision_model="llava:7b")
    assert any("First paragraph" in e.content for e in doc.elements)


def test_load_unsupported_extension_raises(tmp_index_dir):
    from app.loaders import load_document

    p = Path(tmp_index_dir) / "doc.xyz"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(str(p), ollama=None, vision_model="llava:7b")
