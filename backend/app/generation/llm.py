"""Ollama client for LLM and embedding operations."""
from __future__ import annotations

import base64
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger

log = get_logger(__name__)


def get_client() -> httpx.Client:
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30.0)
    return httpx.Client(timeout=180.0, limits=limits)


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0) -> None:
        self.base_url = (base_url or get_settings().ollama_base_url).rstrip("/")
        self.timeout = timeout
        self._client = get_client()
        self._embed_cache: dict[str, list[float]] = {}
        self._cache_max_size = 512
        self._query_embed_cache = lru_cache(maxsize=128)(self._embed_single_cached)

    def _embed_single_cached(self, model: str, text: str) -> list[float]:
        r = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]
        elif isinstance(data, list) and data:
            return data[0] if isinstance(data[0], list) else data
        return []

    def list_models(self) -> list[str]:
        r = self._client.get(f"{self.base_url}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or get_settings().embed_model
        embeddings: list[list[float]] = []
        for text in texts:
            cache_key = f"{model}:{text}"
            if cache_key in self._embed_cache:
                embeddings.append(self._embed_cache[cache_key])
                continue
            emb = self._query_embed_cache(model, text)
            embeddings.append(emb)
            if len(self._embed_cache) < self._cache_max_size:
                self._embed_cache[cache_key] = emb
        return embeddings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(
        self,
        prompt: str,
        model: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
        images: list[bytes] | None = None,
    ) -> str | Iterator[str]:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        if images:
            payload["images"] = [base64.b64encode(b).decode("ascii") for b in images]
        r = self._client.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        if stream:
            return self._stream_response(r)
        return r.json().get("response", "")

    def _stream_response(self, response: httpx.Response) -> Iterator[str]:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                import json as _json

                chunk = _json.loads(line)
                if "response" in chunk:
                    yield chunk["response"]
                if chunk.get("done"):
                    break
            except Exception:
                continue

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        options: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if options:
            payload["options"] = options
        r = self._client.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        if stream:
            return self._stream_response(r)
        return r.json().get("message", {}).get("content", "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def rerank(self, query: str, documents: list[str], model: str | None = None) -> list[float]:
        model = model or get_settings().rerank_model
        if not documents:
            return []
        r = self._client.post(
            f"{self.base_url}/api/rerank",
            json={"model": model, "query": query, "documents": documents},
        )
        if r.status_code != 200:
            return [0.0] * len(documents)
        data = r.json()
        scores: list[float] = []
        if isinstance(data, list):
            for item in data:
                scores.append(float(item.get("score", 0.0)))
        elif isinstance(data, dict) and data.get("scores"):
            scores = [float(s) for s in data["scores"]]
        else:
            scores = [0.0] * len(documents)
        return scores

    def health(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
