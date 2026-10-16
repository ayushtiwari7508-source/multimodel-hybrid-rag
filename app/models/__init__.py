"""Backward compatibility shim for app.models."""

from backend.app.models import (
    Answer,
    Citation,
    DocElement,
    Hit,
    LoadedDocument,
    Route,
    RouteName,
)

__all__ = ["Answer", "Citation", "DocElement", "Hit", "LoadedDocument", "Route", "RouteName"]
