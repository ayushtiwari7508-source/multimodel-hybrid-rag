"""Backward-compatibility shim for app.pipeline.ingest. Use backend.app.ingestion instead."""

from backend.app.ingestion import Ingestor

__all__ = ["Ingestor"]
