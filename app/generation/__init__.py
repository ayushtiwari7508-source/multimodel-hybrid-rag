"""Backward compatibility shim for app.generation."""

from backend.app.generation import (
    SYSTEM_PROMPT,
    Generator,
    OllamaClient,
)

__all__ = ["SYSTEM_PROMPT", "Generator", "OllamaClient"]
