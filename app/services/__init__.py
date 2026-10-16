"""Backward compatibility shim for app.services."""
from backend.app.services import (
    QueryService,
    RAGResponse,
    DocumentService,
)

__all__ = ["QueryService", "RAGResponse", "DocumentService"]
