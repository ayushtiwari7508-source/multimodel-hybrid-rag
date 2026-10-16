"""Hybrid retrieval combining BM25 and dense search with RRF and reranking."""
from __future__ import annotations

import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger, metrics
from backend.app.generation.llm import OllamaClient
from backend.app.models.domain import Hit
from backend.app.retrieval.reranker import CrossEncoderReranker, OllamaReranker
from backend.app.retrieval.keyword_search import BM25Store
from backend.app.routing.classifier import PipelineRoute
from backend.app.retrieval.vector_search import ChromaStore
from backend.app.retrieval.rrf import rrf, normalize_scores

log = get_logger(__name__)

_search_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class RetrievalMetrics:
    bm25_ms: float = 0.0
    dense_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    bm25_count: int = 0
    dense_count: int = 0
    reranked_count: int = 0


def get_top_k_for_route(cfg, route: PipelineRoute) -> tuple[int, int, int]:
    if route == PipelineRoute.FAST:
        return (cfg.fast_top_k_bm25, cfg.fast_top_k_dense, cfg.fast_top_k_rerank)
    elif route == PipelineRoute.COMPLEX:
        return (cfg.complex_top_k_bm25, cfg.complex_top_k_dense, cfg.complex_top_k_rerank)
    else:  # RAG or MULTIMODAL
        return (cfg.rag_top_k_bm25, cfg.rag_top_k_dense, cfg.rag_top_k_rerank)


def should_rerank(cfg, route: PipelineRoute) -> bool:
    _, _, top_k_rerank = get_top_k_for_route(cfg, route)
    return top_k_rerank > 0


class HybridRetriever:
    def __init__(
        self,
        chroma: ChromaStore,
        bm25: BM25Store,
        ollama: OllamaClient,
        reranker: object | None = None,
    ) -> None:
        self.chroma = chroma
        self.bm25 = bm25
        self.ollama = ollama
        self.cfg = get_settings()
        self.reranker = reranker or self._build_default_reranker()

    def _dense_search(self, query: str, top_k: int) -> tuple[dict[str, Hit], float]:
        t0 = time.perf_counter()
        q_emb = self.ollama.embed([query], model=self.cfg.embed_model)[0]
        docs, metas, sims, ids = self.chroma.query(q_emb, n_results=top_k)
        hits: dict[str, Hit] = {}
        for i, doc in enumerate(docs):
            hid = ids[i] if i < len(ids) else f"dense-{i}"
            meta = metas[i] if i < len(metas) else {}
            sim = sims[i] if i < len(sims) else 0.0
            hits[hid] = Hit(
                id=hid, content=doc, meta=meta, dense_score=float(sim), source=meta.get("source", "")
            )
        elapsed = (time.perf_counter() - t0) * 1000
        return hits, elapsed

    def _bm25_search(self, query: str, top_k: int) -> tuple[dict[str, Hit], float]:
        t0 = time.perf_counter()
        results = self.bm25.search(query, n=top_k)
        hits: dict[str, Hit] = {}
        for rec, score in results:
            hits[rec["id"]] = Hit(
                id=rec["id"],
                content=rec["content"],
                meta=rec.get("meta", {}),
                bm25_score=float(score),
                source=rec.get("meta", {}).get("source", ""),
            )
        elapsed = (time.perf_counter() - t0) * 1000
        return hits, elapsed

    def _fuse(self, dense: dict[str, Hit], bm25: dict[str, Hit]) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        dense_ids = list(dense.keys())
        bm25_ids = list(bm25.keys())
        rrf_scores = rrf([dense_ids, bm25_ids], c=self.cfg.rrf_c)

        dense_norm = {k: v.dense_score for k, v in dense.items()}
        bm25_norm_map = {k: v.bm25_score for k, v in bm25.items()}
        for arr in (dense_norm, bm25_norm_map):
            keys = list(arr.keys())
            norm_vals = normalize_scores(list(arr.values()))
            for k, v in zip(keys, norm_vals, strict=True):
                arr[k] = v

        merged: dict[str, Hit] = {}
        for hid in set(dense_ids) | set(bm25_ids):
            h = dense.get(hid) or bm25.get(hid)
            if h is None:
                continue
            h.rrf_score = rrf_scores.get(hid, 0.0)
            h.dense_score = dense_norm.get(hid, 0.0)
            h.bm25_score = bm25_norm_map.get(hid, 0.0)
            h.final_score = h.rrf_score
            merged[hid] = h
        ranked = sorted(merged.values(), key=lambda h: h.final_score, reverse=True)
        elapsed = (time.perf_counter() - t0) * 1000
        return ranked, elapsed

    def _build_default_reranker(self) -> object:
        backend = (self.cfg.rerank_backend or "none").lower()
        ret_mod = sys.modules.get("app.retriever")
        ce_cls = getattr(ret_mod, "CrossEncoderReranker", CrossEncoderReranker) if ret_mod else CrossEncoderReranker
        ollama_cls = getattr(ret_mod, "OllamaReranker", OllamaReranker) if ret_mod else OllamaReranker

        if backend == "cross-encoder":
            return ce_cls(self.cfg.rerank_model)
        if backend == "ollama":
            return ollama_cls(self.ollama, self.cfg.rerank_model)
        return None

    def _rerank(self, query: str, hits: list[Hit], top_k: int) -> tuple[list[Hit], float]:
        if not hits:
            return [], 0.0
        t0 = time.perf_counter()
        cand = hits[: max(top_k * 2, top_k)]
        docs = [h.content for h in cand]
        if self.reranker is not None:
            try:
                scores = self.reranker.score(query, docs)
                if not scores or all(s == 0.0 for s in scores):
                    raise RuntimeError("empty rerank scores")
            except Exception as e:
                log.warning(f"Rerank fallback (RRF only): {e}")
                scores = [h.rrf_score for h in cand]
        else:
            scores = [h.rrf_score for h in cand]
        norm = normalize_scores(scores)
        for h, s in zip(cand, norm, strict=True):
            h.rerank_score = float(s)
            h.final_score = 0.6 * h.rrf_score + 0.4 * h.rerank_score
        cand.sort(key=lambda h: h.final_score, reverse=True)
        elapsed = (time.perf_counter() - t0) * 1000
        return cand[:top_k], elapsed

    def retrieve(
        self,
        query: str,
        route: PipelineRoute = PipelineRoute.RAG,
    ) -> tuple[list[Hit], RetrievalMetrics]:
        metrics_obj = RetrievalMetrics()
        top_k_bm25, top_k_dense, top_k_rerank = get_top_k_for_route(self.cfg, route)

        if route == PipelineRoute.FAST:
            metrics_obj.total_ms = 0.0
            return [], metrics_obj

        fut_dense = _search_executor.submit(self._dense_search, query, top_k_dense)
        fut_bm25 = _search_executor.submit(self._bm25_search, query, top_k_bm25)
        dense, dense_ms = fut_dense.result()
        bm25, bm25_ms = fut_bm25.result()

        metrics_obj.dense_ms = dense_ms
        metrics_obj.bm25_ms = bm25_ms
        metrics_obj.dense_count = len(dense)
        metrics_obj.bm25_count = len(bm25)

        fused, rrf_ms = self._fuse(dense, bm25)
        metrics_obj.rrf_ms = rrf_ms

        if not fused:
            metrics_obj.total_ms = dense_ms + bm25_ms + rrf_ms
            return [], metrics_obj

        if should_rerank(self.cfg, route):
            reranked, rerank_ms = self._rerank(query, fused, top_k_rerank)
            metrics_obj.rerank_ms = rerank_ms
            metrics_obj.reranked_count = len(reranked)
            metrics_obj.total_ms = dense_ms + bm25_ms + rrf_ms + rerank_ms
            return reranked, metrics_obj
        else:
            limit = top_k_rerank if top_k_rerank > 0 else 5
            for h in fused[:limit]:
                h.final_score = h.rrf_score
            metrics_obj.reranked_count = len(fused[:limit])
            metrics_obj.total_ms = dense_ms + bm25_ms + rrf_ms
            return fused[:limit], metrics_obj
