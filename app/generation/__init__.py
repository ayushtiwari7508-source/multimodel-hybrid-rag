"""Backward compatibility shim for app.generation."""
from backend.app.generation import (
    OllamaClient,
    Generator,
    SYSTEM_PROMPT,
)

__all__ = ["OllamaClient", "Generator", "SYSTEM_PROMPT"]
