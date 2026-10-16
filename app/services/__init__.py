"""Backward compatibility shim for app.services."""

from backend.app.services import (
    DocumentService,
    QueryService,
    RAGResponse,
)

__all__ = ["DocumentService", "QueryService", "RAGResponse"]
