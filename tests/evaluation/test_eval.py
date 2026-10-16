"""Tests for evaluation harness and evaluation dataset structure."""

from __future__ import annotations

from scripts.eval import EvalResult
from scripts.eval_dataset import QAPair, load_dataset


def test_eval_dataset_loads():
    ds = load_dataset()
    assert len(ds) >= 1
    assert all(isinstance(q, QAPair) for q in ds)
    for q in ds:
        assert q.question
        assert isinstance(q.expected_keywords, list)


def test_eval_result_dataclass():
    res = EvalResult(
        question="What is RAG?",
        route="general",
        route_correct=True,
        answer="RAG stands for retrieval augmented generation.",
        hits=3,
        latency_ms=120.5,
        keyword_recall=1.0,
        keyword_matched=["RAG"],
        keyword_missed=[],
    )
    assert res.question == "What is RAG?"
    assert res.route_correct is True
    assert res.hits == 3
    assert res.latency_ms > 0
    assert res.keyword_recall == 1.0
