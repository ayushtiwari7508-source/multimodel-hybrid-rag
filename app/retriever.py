"""Backward-compatibility shim for app.retriever. Use backend.app.retrieval instead."""
from backend.app.models.domain import Hit
from backend.app.retrieval import (
    HybridRetriever,
    CrossEncoderReranker,
    OllamaReranker,
    rrf,
    normalize_scores,
)

__all__ = [
    "Hit",
    "HybridRetriever",
    "CrossEncoderReranker",
    "OllamaReranker",
    "rrf",
    "normalize_scores",
]
