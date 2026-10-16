"""Rerankers: cross-encoder (default) or Ollama /api/rerank."""
from __future__ import annotations

import torch
from threading import Lock
from typing import Protocol

from backend.app.core.observability import get_logger

log = get_logger(__name__)


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = Lock()
        self._device = get_device()

    def _load(self):
        from sentence_transformers import CrossEncoder

        log.info(f"loading cross-encoder model: {self.model_name} on {self._device}")
        return CrossEncoder(self.model_name, device=self._device)

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._ensure_model()
        pairs = [[query, doc] for doc in documents]
        raw = model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(s) for s in raw]


class OllamaReranker:
    def __init__(self, client, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return self.client.rerank(query=query, documents=documents, model=self.model_name)
