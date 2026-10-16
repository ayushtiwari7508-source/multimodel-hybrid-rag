"""Backward-compatibility shim for app.ingest. Use app.pipeline.ingest instead."""

from app.pipeline.ingest import Ingestor

__all__ = ["Ingestor"]
