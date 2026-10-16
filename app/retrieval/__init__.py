"""Backward-compatibility shim for app.retrieval. Use backend.app.retrieval instead."""

from backend.app.retrieval import (
    CrossEncoderReranker,
    Hit,
    HybridRetriever,
    OllamaReranker,
    PipelineRoute,
    Reranker,
    normalize_scores,
    rrf,
)

__all__ = [
    "CrossEncoderReranker",
    "Hit",
    "HybridRetriever",
    "OllamaReranker",
    "PipelineRoute",
    "Reranker",
    "_normalize_scores",
    "_rrf",
    "normalize_scores",
    "rrf",
]

# Backward compatibility aliases
_rrf = rrf
_normalize_scores = normalize_scores
