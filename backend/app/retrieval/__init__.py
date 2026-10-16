"""Retrieval package."""
from backend.app.models.domain import Hit
from backend.app.routing.classifier import PipelineRoute
from backend.app.retrieval.hybrid_search import HybridRetriever, RetrievalMetrics
from backend.app.retrieval.keyword_search import BM25Store, chunk_to_records, coerce_metadata
from backend.app.retrieval.vector_search import ChromaStore
from backend.app.retrieval.reranker import CrossEncoderReranker, OllamaReranker, Reranker
from backend.app.retrieval.embeddings import EmbeddingClient
from backend.app.retrieval.rrf import rrf, normalize_scores

__all__ = [
    "Hit",
    "PipelineRoute",
    "HybridRetriever",
    "RetrievalMetrics",
    "BM25Store",
    "ChromaStore",
    "CrossEncoderReranker",
    "OllamaReranker",
    "Reranker",
    "EmbeddingClient",
    "rrf",
    "normalize_scores",
    "chunk_to_records",
    "coerce_metadata",
]
