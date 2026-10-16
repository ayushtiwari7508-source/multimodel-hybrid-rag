"""Document service for managing documents."""
from __future__ import annotations

from backend.app.ingestion.pipeline import Ingestor
from backend.app.retrieval.keyword_search import BM25Store
from backend.app.retrieval.vector_search import ChromaStore
from backend.app.generation.llm import OllamaClient


class DocumentService:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.chroma = ChromaStore()
        self.bm25 = BM25Store()
        self.ingestor = Ingestor(self.ollama, self.chroma, self.bm25)

    def ingest_paths(self, paths: list[str]) -> dict:
        return self.ingestor.ingest_paths(paths)

    def ingest_text(self, text: str, source: str = "user-text") -> dict:
        return self.ingestor.ingest_text(text, source)

    def list_documents(self) -> list[dict]:
        sources = self.chroma.list_distinct("source")
        doc_ids = self.chroma.list_distinct("doc_id")
        out: dict[str, dict] = {}
        for s in sources:
            out[s] = {"source": s, "doc_id": None, "chroma_count": 0}
        for d in doc_ids:
            if d not in out:
                out[d] = {"source": None, "doc_id": d, "chroma_count": 0}
        return sorted(out.values(), key=lambda r: r["source"] or r["doc_id"] or "")

    def delete_document(self, doc_id: str) -> dict:
        if not doc_id:
            return {"deleted_chroma": 0, "deleted_bm25": 0}
        removed_chroma = self.chroma.delete_where({"doc_id": doc_id})
        removed_bm25 = self.bm25.delete_where({"doc_id": doc_id})
        return {"deleted_chroma": removed_chroma, "deleted_bm25": removed_bm25, "doc_id": doc_id}

    def clear(self) -> None:
        self.chroma.clear()
        self.bm25.clear()

    def stats(self) -> dict:
        return {"chroma": self.chroma.count(), "bm25": len(self.bm25.corpus)}
