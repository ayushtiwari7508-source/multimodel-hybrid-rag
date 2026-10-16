"""Backward compatibility shim for app.schemas."""
from backend.app.schemas import (
    CitationModel,
    HitModel,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RouteModel,
)

__all__ = ["CitationModel", "HitModel", "IngestTextRequest", "QueryRequest", "QueryResponse", "RouteModel"]
