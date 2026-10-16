"""Backward-compatibility shim for app.retriever. Use backend.app.retrieval instead."""

from backend.app.models.domain import Hit
from backend.app.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    OllamaReranker,
    normalize_scores,
    rrf,
)

__all__ = [
    "CrossEncoderReranker",
    "Hit",
    "HybridRetriever",
    "OllamaReranker",
    "normalize_scores",
    "rrf",
]
