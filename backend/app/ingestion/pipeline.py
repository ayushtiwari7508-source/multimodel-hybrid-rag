"""Document ingestion pipeline."""
from __future__ import annotations

from collections.abc import Iterable

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger, stable_hash
from backend.app.generation.llm import OllamaClient
from backend.app.ingestion.loaders.loaders import load_paths
from backend.app.retrieval.keyword_search import BM25Store, chunk_to_records
from backend.app.retrieval.vector_search import ChromaStore

log = get_logger(__name__)


class Ingestor:
    def __init__(self, ollama: OllamaClient, chroma: ChromaStore, bm25: BM25Store) -> None:
        self.ollama = ollama
        self.chroma = chroma
        self.bm25 = bm25
        self.cfg = get_settings()

    def ingest_paths(self, paths: Iterable[str]) -> dict:
        cfg = self.cfg
        docs = load_paths(paths, self.ollama, cfg.vision_model)
        all_chunks: list[dict] = []
        per_doc_counts: dict[str, int] = {}
        for d in docs:
            chunks = d.to_chunks(cfg.chunk_size, cfg.chunk_overlap)
            for ch in chunks:
                meta = ch.get("meta", {})
                ch["meta"]["doc_id"] = d.doc_id
                if "source" not in meta:
                    meta["source"] = d.source
            all_chunks.extend(chunks)
            per_doc_counts[d.doc_id] = len(chunks)

        records = chunk_to_records(all_chunks)
        if not records:
            return {"chunks": 0, "docs": per_doc_counts}

        texts = [r["content"] for r in records]
        log.info(f"Embedding {len(texts)} chunks with {cfg.embed_model}")
        embs: list[list[float]] = []
        batch = cfg.embed_batch_size
        for i in range(0, len(texts), batch):
            embs.extend(self.ollama.embed(texts[i : i + batch], model=cfg.embed_model))

        ids = [r["id"] for r in records]
        metas = [r["meta"] for r in records]
        self.chroma.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        self.bm25.add(records)
        log.info(f"Indexed {len(records)} chunks")
        return {"chunks": len(records), "docs": per_doc_counts}

    def ingest_text(self, text: str, source: str = "user-text") -> dict:
        cfg = self.cfg
        doc_id = stable_hash({"s": source, "t": text[:200]})
        chunks = []
        for i in range(0, len(text), cfg.chunk_size - cfg.chunk_overlap):
            piece = text[i : i + cfg.chunk_size]
            chunks.append(
                {
                    "content": piece,
                    "meta": {"source": source, "doc_id": doc_id, "section": f"part {len(chunks) + 1}"},
                }
            )
        records = chunk_to_records(chunks)
        texts = [r["content"] for r in records]
        embs = self.ollama.embed(texts, model=cfg.embed_model)
        ids = [r["id"] for r in records]
        metas = [r["meta"] for r in records]
        self.chroma.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        self.bm25.add(records)
        return {"chunks": len(records), "docs": {doc_id: len(records)}}
