"""Query service for handling search and generation."""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger, metrics
from backend.app.generation.llm import OllamaClient
from backend.app.generation.prompts import Generator
from backend.app.models.domain import Hit, Route
from backend.app.ingestion.pipeline import Ingestor
from backend.app.retrieval.keyword_search import BM25Store
from backend.app.routing.classifier import PipelineRoute, QueryClassifier
from backend.app.retrieval.vector_search import ChromaStore
from backend.app.retrieval.hybrid_search import HybridRetriever, RetrievalMetrics
from backend.app.routing.router import ModelRouter

log = get_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    route: Route
    hits: list[Hit]
    pipeline_route: PipelineRoute
    classification: str
    retrieval_metrics: RetrievalMetrics | None = None
    total_latency_ms: float = 0.0


class QueryService:
    def __init__(self) -> None:
        self.cfg = get_settings()
        self.ollama = OllamaClient()
        self.chroma = ChromaStore(self.cfg.chroma_persist_dir)
        self.bm25 = BM25Store(self.cfg.bm25_persist_path)
        self.retriever = HybridRetriever(self.chroma, self.bm25, self.ollama)
        self.router = ModelRouter(self.ollama)
        self.classifier = QueryClassifier()
        self.generator = Generator(self.ollama)
        self.ingestor = Ingestor(self.ollama, self.chroma, self.bm25)
        self._models_checked = False
        self._model_status: dict = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._last_classification: dict = {}
        # Pre-warm cross-encoder if enabled (synchronous to ensure first query is fast)
        if (self.cfg.rerank_backend or "").lower() == "cross-encoder":
            try:
                _ = self.retriever.reranker._ensure_model()
                log.info("Cross-encoder model warmed up")
            except Exception as e:
                log.warning(f"Cross-encoder warmup failed: {e}")

    def _check_models(self) -> tuple[bool, list[str]]:
        """Check Ollama reachability and model availability. Returns (ok, missing_required)."""
        if self._models_checked:
            required_labels = {"embed", "router", "general", "code"}
            missing = [label for label in required_labels if self._model_status.get(label) == "missing"]
            return (len(missing) == 0, missing)
        if not self.ollama.health():
            return (False, ["ollama_unreachable"])
        available = set(self.ollama.list_models())
        required = {
            "embed": self.cfg.embed_model,
            "router": self.cfg.router_model,
            "general": self.cfg.general_model,
            "code": self.cfg.code_model,
        }
        missing = []
        for label, tag in required.items():
            has = any(tag == a or a == f"{tag}:latest" or a.startswith(tag + ":") for a in available)
            self._model_status[label] = "ok" if has else "missing"
            if not has:
                missing.append(label)
        optional = {"vision": self.cfg.vision_model}
        if (self.cfg.rerank_backend or "none").lower() == "ollama":
            optional["rerank"] = self.cfg.rerank_model
        for label, tag in optional.items():
            has = any(tag == a or a == f"{tag}:latest" or a.startswith(tag + ":") for a in available)
            self._model_status[label] = "ok" if has else "missing"
        self._models_checked = True
        return (len(missing) == 0, missing)

    def _require_models(self, missing_ok: bool = False) -> None:
        ok, missing = self._check_models()
        if not ok:
            missing_str = ", ".join(missing) if missing else "ollama_unreachable"
            extra = " (run `ollama serve`)" if missing_str == "ollama_unreachable" else ""
            raise RuntimeError(
                f"Ollama/models not ready: {missing_str}. "
                f"Start Ollama and pull missing models with `ollama pull ...`{extra}."
            )

    def _select_model_for_pipeline_route(self, pipeline_route: PipelineRoute, question: str) -> Route:
        """Select appropriate model based on pipeline route and question."""
        if pipeline_route == PipelineRoute.FAST:
            return Route(name="fast", model=self.cfg.fast_model, reason="fast path - no retrieval needed")
        if pipeline_route == PipelineRoute.MULTIMODAL:
            return Route(name="multimodal", model=self.cfg.general_model, reason="multimodal query")
        if pipeline_route == PipelineRoute.COMPLEX:
            return Route(name="general", model=self.cfg.general_model, reason="complex query")
        return self.router.route(question)

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        self._require_models()
        total_start = time.perf_counter()

        classification = self.classifier.classify(question)
        pipeline_route = classification.route
        log.info(f"Query classified as {pipeline_route.value}: {classification.reason} (confidence={classification.confidence:.2f})")

        metrics.observe("query.classification.ms", classification.latency_ms, route=pipeline_route.value)
        metrics.incr("query.classification.count", route=pipeline_route.value)

        route = self._select_model_for_pipeline_route(pipeline_route, question)

        hits, retrieval_metrics = self.retriever.retrieve(question, route=pipeline_route)

        if retrieval_metrics:
            metrics.observe("retrieval.bm25.ms", retrieval_metrics.bm25_ms, route=pipeline_route.value)
            metrics.observe("retrieval.dense.ms", retrieval_metrics.dense_ms, route=pipeline_route.value)
            metrics.observe("retrieval.rrf.ms", retrieval_metrics.rrf_ms, route=pipeline_route.value)
            metrics.observe("retrieval.rerank.ms", retrieval_metrics.rerank_ms, route=pipeline_route.value)
            metrics.observe("retrieval.total.ms", retrieval_metrics.total_ms, route=pipeline_route.value)
            metrics.incr("retrieval.bm25.count", retrieval_metrics.bm25_count, route=pipeline_route.value)
            metrics.incr("retrieval.dense.count", retrieval_metrics.dense_count, route=pipeline_route.value)
            metrics.incr("retrieval.reranked.count", retrieval_metrics.reranked_count, route=pipeline_route.value)

        if top_k:
            hits = hits[:top_k]

        gen_start = time.perf_counter()
        ans = self.generator.answer(question, hits, route)
        gen_latency = (time.perf_counter() - gen_start) * 1000

        metrics.observe("generation.latency.ms", gen_latency, model=route.model, route=pipeline_route.value)

        total_latency = (time.perf_counter() - total_start) * 1000
        metrics.observe("query.total.latency.ms", total_latency, route=pipeline_route.value, model=route.model)

        retrieval_ms = f"{retrieval_metrics.total_ms:.1f}" if retrieval_metrics else "0"
        log.info(
            f"Route={route.name} model={route.model} pipeline={pipeline_route.value} "
            f"reason={route.reason} classification={classification.reason} "
            f"retrieval_ms={retrieval_ms} gen_ms={gen_latency:.1f} total_ms={total_latency:.1f}"
        )

        return RAGResponse(
            answer=ans.text,
            route=route,
            hits=hits,
            pipeline_route=pipeline_route,
            classification=classification.reason,
            retrieval_metrics=retrieval_metrics,
            total_latency_ms=total_latency,
        )

    def stream(self, question: str, top_k: int | None = None) -> tuple[Route, list[Hit], Iterator[str]]:
        self._require_models()
        total_start = time.perf_counter()

        classification = self.classifier.classify(question)
        pipeline_route = classification.route
        log.info(f"Query classified as {pipeline_route.value}: {classification.reason}")

        self._last_classification = {
            "pipeline_route": pipeline_route.value,
            "reason": classification.reason,
        }

        route = self._select_model_for_pipeline_route(pipeline_route, question)

        hits, retrieval_metrics = self.retriever.retrieve(question, route=pipeline_route)

        if top_k:
            hits = hits[:top_k]

        if retrieval_metrics:
            metrics.observe("retrieval.bm25.ms", retrieval_metrics.bm25_ms, route=pipeline_route.value)
            metrics.observe("retrieval.dense.ms", retrieval_metrics.dense_ms, route=pipeline_route.value)
            metrics.observe("retrieval.rrf.ms", retrieval_metrics.rrf_ms, route=pipeline_route.value)
            metrics.observe("retrieval.rerank.ms", retrieval_metrics.rerank_ms, route=pipeline_route.value)
            metrics.observe("retrieval.total.ms", retrieval_metrics.total_ms, route=pipeline_route.value)

        metrics.observe("query.classification.ms", classification.latency_ms, route=pipeline_route.value)
        metrics.incr("query.classification.count", route=pipeline_route.value)

        total_latency = (time.perf_counter() - total_start) * 1000
        metrics.observe("query.total.latency.ms", total_latency, route=pipeline_route.value, model=route.model)

        return route, hits, self.generator.stream(question, hits, route)

    def stats(self) -> dict:
        return {"chroma": self.chroma.count(), "bm25": len(self.bm25.corpus)}

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

    def close(self) -> None:
        self.ollama.close()
        self._executor.shutdown(wait=False)
