"""API package - backward compatibility."""

from backend.app.api.main import app, get_pipeline, pipeline

__all__ = ["app", "get_pipeline", "pipeline"]
