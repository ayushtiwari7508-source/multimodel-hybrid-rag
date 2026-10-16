"""Backward-compatibility shim for app.pipeline. Use backend.app.services instead."""
from backend.app.services import RAGPipeline, RAGResponse

__all__ = ["RAGPipeline", "RAGResponse"]
