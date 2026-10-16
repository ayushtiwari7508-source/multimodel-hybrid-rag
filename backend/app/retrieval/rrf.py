"""Reciprocal Rank Fusion utilities."""
from __future__ import annotations

import numpy as np


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return [1.0 for _ in scores]
    return ((arr - mn) / (mx - mn)).tolist()


def rrf(rankings: list[list[str]], c: int) -> dict[str, float]:
    rrf_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (c + rank + 1)
    return rrf_scores
