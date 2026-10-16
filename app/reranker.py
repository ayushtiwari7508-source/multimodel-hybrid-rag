"""Backward-compatibility shim for app.reranker. Use app.reranking instead."""

from app.reranking.reranker import CrossEncoderReranker, OllamaReranker, Reranker

__all__ = ["Reranker", "CrossEncoderReranker", "OllamaReranker"]
