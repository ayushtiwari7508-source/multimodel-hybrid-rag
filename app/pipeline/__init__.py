"""Backward-compatibility shim for app.pipeline. Use backend.app.services instead."""

from backend.app.services import QueryService as RAGPipeline
from backend.app.services import RAGResponse

__all__ = ["RAGPipeline", "RAGResponse"]
