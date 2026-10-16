"""Dense vector search using Chroma with Ollama embeddings."""
from __future__ import annotations

from chromadb import PersistentClient
from chromadb.api.types import EmbeddingFunction

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger
from backend.app.generation.llm import OllamaClient

log = get_logger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts, model=self.model)


class ChromaStore:
    def __init__(self, persist_dir: str | None = None) -> None:
        self.cfg = get_settings()
        self.persist_dir = persist_dir or self.cfg.chroma_persist_dir
        self.client = PersistentClient(path=self.persist_dir)
        self.ollama = OllamaClient()
        self.ef = OllamaEmbeddingFunction(self.ollama, self.cfg.embed_model)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.collection = self.client.get_collection("documents", embedding_function=self.ef)
        except Exception:
            self.collection = self.client.create_collection("documents", embedding_function=self.ef)

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        self._ensure_collection()
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], n_results: int = 10):
        self._ensure_collection()
        res = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return (
            res.get("documents", [[]])[0],
            res.get("metadatas", [[]])[0],
            res.get("distances", [[]])[0],
            res.get("ids", [[]])[0],
        )

    def count(self) -> int:
        self._ensure_collection()
        return self.collection.count()

    def delete_where(self, where: dict) -> int:
        self._ensure_collection()
        try:
            res = self.collection.delete(where=where)
            return len(res.get("ids", [])) if res else 0
        except Exception:
            return 0

    def list_distinct(self, field: str) -> list[str]:
        self._ensure_collection()
        try:
            res = self.collection.get()
            metadatas = res.get("metadatas", [])
            seen = set()
            for m in metadatas:
                v = m.get(field)
                if v is not None:
                    seen.add(str(v))
            return sorted(seen)
        except Exception:
            return []

    def clear(self) -> None:
        try:
            self.client.delete_collection("documents")
        except Exception:
            pass
        self._ensure_collection()
