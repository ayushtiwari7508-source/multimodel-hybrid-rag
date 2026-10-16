"""Backward-compatibility shim for app.storage. Use backend.app.retrieval instead."""

from backend.app.retrieval import BM25Store, ChromaStore, chunk_to_records, coerce_metadata

__all__ = ["BM25Store", "ChromaStore", "_coerce_metadata", "chunk_to_records", "coerce_metadata"]

# Backward compatibility alias
_coerce_metadata = coerce_metadata
