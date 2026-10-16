"""API package - backward compatibility."""
from backend.app.api.main import app, pipeline, get_pipeline

__all__ = ["app", "pipeline", "get_pipeline"]
