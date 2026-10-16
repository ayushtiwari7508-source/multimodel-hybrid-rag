"""Tests for the hybrid retriever's score fusion + RRF math."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from app.retrieval import Hit, HybridRetriever, _normalize_scores, _rrf


def test_normalize_basic():
    out = _normalize_scores([0.0, 0.5, 1.0])
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)


def test_normalize_constant_returns_ones():
    assert _normalize_scores([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


def test_normalize_empty():
    assert _normalize_scores([]) == []


def test_rrf_single_ranking():
    rrf = _rrf([["a", "b", "c"]], c=60)
    assert math.isclose(rrf["a"], 1 / 61, rel_tol=1e-6)
    assert math.isclose(rrf["b"], 1 / 62, rel_tol=1e-6)
    assert math.isclose(rrf["c"], 1 / 63, rel_tol=1e-6)


def test_rrf_combines_rankings():
    # 'a' is rank 0 in dense, rank 1 in bm25 -> highest total
    # 'b' is rank 1 in dense, rank 0 in bm25 -> second highest
    # 'c' only appears in dense
    rrf = _rrf([["a", "b", "c"], ["b", "a"]], c=60)
    assert rrf["a"] > rrf["c"]
    assert rrf["b"] > rrf["c"]
    assert math.isclose(rrf["a"], rrf["b"], rel_tol=1e-3)  # symmetric rank 0/1 vs 1/0


def test_fuse_merges_overlapping_hits():
    # Exercise _fuse directly with fake hit dicts
    hr = HybridRetriever.__new__(HybridRetriever)
    hr.cfg = type("Cfg", (), {"rrf_c": 60})()
    dense = {
        "d1": Hit(id="d1", content="dense doc 1", meta={"source": "a"}, dense_score=0.9),
        "d2": Hit(id="d2", content="dense doc 2", meta={"source": "b"}, dense_score=0.5),
    }
    bm = {
        "d1": Hit(id="d1", content="dense doc 1", meta={"source": "a"}, bm25_score=2.0),
        "d3": Hit(id="d3", content="bm25 only", meta={"source": "c"}, bm25_score=1.5),
    }
    fused, _ = hr._fuse(dense, bm)
    ids = [h.id for h in fused]
    assert "d1" in ids and "d2" in ids and "d3" in ids
    top = fused[0]
    assert top.id == "d1"
    assert top.rrf_score > 0
    assert top.final_score == top.rrf_score
    # d1 appears in both, should have the highest rrf
    assert fused[0].id == "d1"


def test_rerank_fallback_uses_rrf_when_rerank_returns_zeros():
    hr = HybridRetriever.__new__(HybridRetriever)
    hr.cfg = type("Cfg", (), {"top_k_rerank": 3, "rrf_c": 60})()
    hr.ollama = MagicMock()
    # The new dispatch goes through self.reranker.score() (CrossEncoderReranker by default).
    # Stub the reranker to mimic the Ollama /api/rerank contract of returning zeros.
    hr.reranker = MagicMock()
    hr.reranker.score.return_value = [0.0, 0.0, 0.0]

    cand = [
        Hit(id="a", content="alpha", meta={}, rrf_score=0.030),
        Hit(id="b", content="bravo", meta={}, rrf_score=0.020),
        Hit(id="c", content="charlie", meta={}, rrf_score=0.010),
    ]
    out, _ = hr._rerank("query", cand, top_k=3)
    assert len(out) == 3
    # Order preserved by rrf when rerank returns all zeros (fallback path)
    assert [h.id for h in out] == ["a", "b", "c"]
