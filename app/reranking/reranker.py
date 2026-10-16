"""Backward-compatibility shim for app.reranking.reranker. Use backend.app.retrieval instead."""
from backend.app.retrieval import CrossEncoderReranker, OllamaReranker, Reranker

__all__ = ["Reranker", "CrossEncoderReranker", "OllamaReranker"]
