"""Backward compatibility shim for app.routing."""
from backend.app.routing import (
    PipelineRoute,
    QueryClassifier,
    ModelRouter,
)

__all__ = ["PipelineRoute", "QueryClassifier", "ModelRouter"]
