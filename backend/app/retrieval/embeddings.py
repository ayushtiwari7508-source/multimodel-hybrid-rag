"""Embedding client for Ollama."""
from __future__ import annotations

from backend.app.generation.llm import OllamaClient


class EmbeddingClient:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return self.client.embed(texts, model)
