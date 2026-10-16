"""Backward-compatibility shim for app.generation.ollama_client. Use backend.app.generation instead."""
from backend.app.generation import OllamaClient

__all__ = ["OllamaClient"]
