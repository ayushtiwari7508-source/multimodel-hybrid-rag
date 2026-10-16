"""Backward-compatibility shim for app.retrieval. Use backend.app.retrieval instead."""
from backend.app.retrieval import (
    HybridRetriever,
    CrossEncoderReranker,
    OllamaReranker,
    Reranker,
    rrf,
    normalize_scores,
    Hit,
    PipelineRoute,
)

__all__ = [
    "Hit",
    "HybridRetriever",
    "CrossEncoderReranker",
    "OllamaReranker",
    "Reranker",
    "rrf",
    "normalize_scores",
    "_rrf",
    "_normalize_scores",
    "PipelineRoute",
]

# Backward compatibility aliases
_rrf = rrf
_normalize_scores = normalize_scores
