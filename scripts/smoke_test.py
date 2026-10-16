"""End-to-end smoke test. Requires Ollama running with the configured models pulled.

Run:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings
from backend.app.generation.llm import OllamaClient
from backend.app.ingestion.pipeline import Ingestor
from backend.app.retrieval.keyword_search import BM25Store
from backend.app.retrieval.vector_search import ChromaStore
from backend.app.services.query_service import QueryService


def main() -> int:
    settings = get_settings()

    # Create components
    ollama = OllamaClient()
    chroma = ChromaStore(settings.chroma_persist_dir)
    bm25 = BM25Store(settings.bm25_persist_path)
    ingestor = Ingestor(ollama, chroma, bm25)

    pipe = QueryService()
    if not pipe.ollama.health():
        print("Ollama not reachable at", settings.ollama_base_url)
        return 1
    available = set(pipe.ollama.list_models())

    def _have(tag: str) -> bool:
        return any(tag == a or a == f"{tag}:latest" or a.startswith(tag + ":") for a in available)

    needed = {
        "embed": settings.embed_model,
        "vision": settings.vision_model,
        "router": settings.router_model,
        "general": settings.general_model,
        "code": settings.code_model,
    }
    missing = [label for label, tag in needed.items() if not _have(tag)]
    if missing:
        print("Missing models in Ollama:", missing)
        print("Run: ollama pull", " ".join(t for t in needed.values() if not _have(t)))
        return 1

    if not _have(settings.rerank_model):
        print(
            f"NOTE: rerank model '{settings.rerank_model}' not installed — retriever will use the "
            "weighted RRF + dense + BM25 fallback blend."
        )

    print("Clearing index for clean test...")
    pipe.clear()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample.md"
        p.write_text(
            "# RAG Overview\n\n"
            "Hybrid RAG combines BM25 sparse retrieval with dense vector search, "
            "and a cross-encoder reranker. The router picks the best model per query.\n\n"
            "## Equations\n\n"
            "Cosine similarity is defined as: $cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$.\n\n"
            "## Tables\n\n"
            "| Route | Model |\n|---|---|\n| code | qwen-coder |\n| general | qwen |\n",
            encoding="utf-8",
        )
        print("Ingesting sample.md...")
        res = ingestor.ingest_paths([str(p)])
        print("Indexed:", res)

    questions = [
        "What is hybrid RAG?",
        "Write a Python function to compute cosine similarity.",
        "What does the table show?",
        "hi",
    ]
    for q in questions:
        print("\nQ:", q)
        r = pipe.query(q)
        print("Route:", r.route.name, r.route.model, "->", r.route.reason)
        print("Hits:", len(r.hits))
        print("Answer:\n", r.answer[:600], "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
