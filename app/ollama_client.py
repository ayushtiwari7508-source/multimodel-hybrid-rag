"""Backward-compatibility shim for app.ollama_client. Use app.generation instead."""

from app.generation.ollama_client import OllamaClient

__all__ = ["OllamaClient"]
