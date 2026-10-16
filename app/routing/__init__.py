"""Backward compatibility shim for app.routing."""

from backend.app.routing import (
    ModelRouter,
    PipelineRoute,
    QueryClassifier,
)

__all__ = ["ModelRouter", "PipelineRoute", "QueryClassifier"]
