from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Root is 2 levels up from app/core/ -> hybrid-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), ".env"),
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"

    embed_model: str = "nomic-embed-text"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_backend: str = "cross-encoder"  # or "ollama"
    vision_model: str = "llava:7b"
    enable_vision: bool = False  # skip LLaVA for faster ingestion
    vision_only_when_no_text: bool = True  # only vision if page has no text
    router_model: str = "qwen2.5:3b"
    general_model: str = "qwen2.5:7b"
    code_model: str = "qwen2.5-coder:7b"
    math_model: str = "qwen2.5:7b"
    fast_model: str = "qwen2.5:3b"

    chroma_persist_dir: str = "./data/chroma"
    bm25_persist_path: str = "./data/bm25/index.pkl"
    meta_persist_dir: str = "./data/index_meta"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k_bm25: int = 5
    top_k_dense: int = 5
    top_k_rerank: int = 3
    rrf_c: int = 60
    max_context_tokens: int = 6000
    embed_batch_size: int = 64
    skip_router: bool = False

    # Query classifier / pipeline routing
    enable_query_classifier: bool = True
    simple_query_max_tokens: int = 10
    simple_query_keywords: list[str] = ["what is", "define", "who is", "when did", "where is", "capital of"]
    document_query_keywords: list[str] = ["according to", "in the document", "from the document", "summarize", "find", "section", "page", "document", "say", "text", "extract"]
    multimodal_query_keywords: list[str] = ["chart", "figure", "diagram", "image", "table", "graph", "plot", "visual"]
    complex_query_keywords: list[str] = ["compare", "contrast", "analyze", "explain why", "how does", "relationship between", "implications"]

    # Retrieval tuning per pipeline
    fast_top_k_bm25: int = 0
    fast_top_k_dense: int = 0
    fast_top_k_rerank: int = 0
    rag_top_k_bm25: int = 10
    rag_top_k_dense: int = 10
    rag_top_k_rerank: int = 5
    complex_top_k_bm25: int = 20
    complex_top_k_dense: int = 20
    complex_top_k_rerank: int = 8

    api_host: str = "0.0.0.0"  # noqa: S104 -- server intended to bind all interfaces
    api_port: int = 8000
    streamlit_port: int = 8501

    # Auth & rate limiting
    api_key: str = ""  # if empty, auth is disabled
    rate_limit_per_min: int = 60  # requests per minute per IP

    def ensure_dirs(self) -> None:
        for d in [
            self.chroma_persist_dir,
            Path(self.bm25_persist_path).parent,
            self.meta_persist_dir,
            "./data/raw",
            "./data/processed",
            "./data/indexes",
            "./data/uploads",
            "./logs",
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
