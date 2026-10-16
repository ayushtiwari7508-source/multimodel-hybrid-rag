"""Backward-compatibility shim for app.retrieval.classifier. Use backend.app.routing instead."""
from backend.app.routing import PipelineRoute, QueryClassifier

__all__ = ["PipelineRoute", "QueryClassifier"]
