"""Tests for the citation-aware context formatter and source renderer."""

from __future__ import annotations

from app.generator import Citation, _format_context, _format_sources
from app.retriever import Hit


def _hit(cid: str, content: str, **meta) -> Hit:
    return Hit(id=cid, content=content, meta=meta)


def test_format_context_respects_token_budget():
    hits = [_hit(f"id{i}", "x" * 400, source=f"doc{i}.md", page=i + 1, section="s") for i in range(10)]
    ctx, cites = _format_context(hits, max_tokens=200)
    assert ctx, "context should not be empty"
    # 1 token ~= 4 chars; budget 200 tokens ~= 800 chars.
    # We should fit at least the first 1-2 hits but not all 10.
    assert 1 <= len(cites) <= 5
    assert all(isinstance(c, Citation) for c in cites)
    assert cites[0].index == 1
    assert cites[0].source == "doc0.md"
    assert cites[0].page == 1


def test_format_context_empty_returns_empty():
    ctx, cites = _format_context([], max_tokens=1000)
    assert ctx == ""
    assert cites == []


def test_format_context_drops_oversized_block():
    huge = _hit("huge", "y" * 100_000, source="big.md")
    ctx, cites = _format_context([huge], max_tokens=50)
    # 50 tokens ~= 200 chars; 100k-char block can't fit -> dropped
    assert ctx == ""
    assert cites == []


def test_format_sources_empty():
    assert _format_sources([]) == ""


def test_format_sources_includes_page_and_section():
    cites = [
        Citation(index=1, source="a.pdf", page=3, section="intro", element_type="text", snippet="..."),
        Citation(index=2, source="b.md", page=None, section=None, element_type="table", snippet="..."),
    ]
    out = _format_sources(cites)
    assert "**Sources:**" in out
    assert "p.3" in out
    assert "/ intro" in out
    assert "a.pdf" in out
    assert "b.md" in out
