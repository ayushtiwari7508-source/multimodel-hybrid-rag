"""Backward-compatibility shim for app.pipeline. Use backend.app.services instead."""
from backend.app.services import QueryService as RAGPipeline, RAGResponse

__all__ = ["RAGPipeline", "RAGResponse"]
