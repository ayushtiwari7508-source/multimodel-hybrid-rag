"""Query classifier for pipeline routing."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger

log = get_logger(__name__)


class PipelineRoute(str, Enum):
    """Pipeline execution path."""
    FAST = "fast"           # No retrieval, direct LLM
    RAG = "rag"             # Standard BM25 + Dense + RRF + Rerank
    COMPLEX = "complex"     # Extended retrieval + strong model
    MULTIMODAL = "multimodal"  # Multimodal retrieval + vision model


@dataclass
class ClassificationResult:
    route: PipelineRoute
    confidence: float
    reason: str
    latency_ms: float
    matched_keywords: list[str]


class QueryClassifier:
    """Lightweight query classifier that routes to appropriate pipeline."""

    def __init__(self) -> None:
        self.cfg = get_settings()
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for keyword matching."""
        # Code queries - check BEFORE document patterns
        self.code_patterns = [
            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            for kw in ["code", "function", "bug", "error", "python", "javascript", "sql", "class", "api", "regex", "compile", "syntax", "def ", "import ", "from ", "query", "database", "script", "shell", "bash", "command"]
        ]
        # Math queries - check BEFORE document patterns
        self.math_patterns = [
            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            for kw in ["equation", "derivative", "integral", "matrix", "proof", "theorem", "formula", "cosine", "similarity", "bm25", "tf-idf", "idf", "tf ", "rrf", "cross-encoder", "rerank", "entropy", "kl divergence", "probability", "variance", "expectation", "gradient", "optimization", "sVD", "eigenvalue", "eigenvector"]
        ]
        # Simple queries: factual, definitional, short
        self.simple_patterns = [
            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            for kw in self.cfg.simple_query_keywords
        ]
        # Document-specific queries - more specific phrases
        self.document_patterns = [
            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            for kw in self.cfg.document_query_keywords
        ]
        # Multimodal queries
        self.multimodal_patterns = [
            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            for kw in self.cfg.multimodal_query_keywords
        ]
        # Complex queries
        self.complex_patterns = [
            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            for kw in self.cfg.complex_query_keywords
        ]

    def _count_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return len(text.split())

    def _match_keywords(self, query: str, patterns: list[re.Pattern]) -> list[str]:
        """Return list of matched keywords."""
        matched = []
        for pattern in patterns:
            if pattern.search(query):
                # Extract the keyword from pattern
                kw = pattern.pattern.replace(r"\b", "").replace(r"\\", "")
                matched.append(kw)
        return matched

    def classify(self, query: str) -> ClassificationResult:
        """Classify query into pipeline route."""
        start = time.perf_counter()

        if not self.cfg.enable_query_classifier:
            latency = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                route=PipelineRoute.RAG,
                confidence=1.0,
                reason="classifier disabled",
                latency_ms=latency,
                matched_keywords=[],
            )

        q = query.strip().lower()
        token_count = self._count_tokens(q)

        # Check multimodal first (highest priority for specialized handling)
        multimodal_matches = self._match_keywords(q, self.multimodal_patterns)
        if multimodal_matches:
            latency = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                route=PipelineRoute.MULTIMODAL,
                confidence=0.9,
                reason=f"multimodal keywords: {multimodal_matches}",
                latency_ms=latency,
                matched_keywords=multimodal_matches,
            )

        # Check code queries (before document patterns to avoid false RAG routing)
        code_matches = self._match_keywords(q, self.code_patterns)
        if code_matches:
            latency = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                route=PipelineRoute.RAG,
                confidence=0.9,
                reason=f"code keywords: {code_matches}",
                latency_ms=latency,
                matched_keywords=code_matches,
            )

        # Check math queries (before document patterns)
        math_matches = self._match_keywords(q, self.math_patterns)
        if math_matches:
            latency = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                route=PipelineRoute.RAG,
                confidence=0.9,
                reason=f"math keywords: {math_matches}",
                latency_ms=latency,
                matched_keywords=math_matches,
            )

        # Check complex queries
        complex_matches = self._match_keywords(q, self.complex_patterns)
        if complex_matches:
            latency = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                route=PipelineRoute.COMPLEX,
                confidence=0.85,
                reason=f"complex keywords: {complex_matches}",
                latency_ms=latency,
                matched_keywords=complex_matches,
            )

        # Check document-specific queries
        document_matches = self._match_keywords(q, self.document_patterns)
        if document_matches:
            # Only route to RAG if query is substantial (not just "text" or "find")
            # Require at least 2 document keywords or a longer query
            if len(document_matches) >= 2 or token_count > 8:
                latency = (time.perf_counter() - start) * 1000
                return ClassificationResult(
                    route=PipelineRoute.RAG,
                    confidence=0.9,
                    reason=f"document keywords: {document_matches}",
                    latency_ms=latency,
                    matched_keywords=document_matches,
                )

        # Check simple queries (short, factual) - but only if no other patterns matched
        simple_matches = self._match_keywords(q, self.simple_patterns)
        # Only route to FAST if we have explicit simple keyword matches
        # or if it's a very short query (<= 5 tokens) that looks like a simple question
        is_very_short = token_count <= 5
        if simple_matches or (is_very_short and any(q.startswith(w) for w in ["what", "who", "when", "where", "define", "hi", "hello", "thanks", "ok"])):
            # BUT: if the query contains technical terms that need retrieval, don't use FAST
            technical_terms = ["rrf", "bm25", "cross-encoder", "rerank", "hnsw", "ivf", "pq", "quantization", "embedding", "vector", "chroma", "ollama"]
            has_technical = any(term in q for term in technical_terms)
            if has_technical:
                # Route to RAG instead for technical definitional queries
                latency = (time.perf_counter() - start) * 1000
                return ClassificationResult(
                    route=PipelineRoute.RAG,
                    confidence=0.8,
                    reason=f"technical definitional query (tokens={token_count})",
                    latency_ms=latency,
                    matched_keywords=[],
                )
            latency = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                route=PipelineRoute.FAST,
                confidence=0.8 if simple_matches else 0.7,
                reason=f"simple query (tokens={token_count}, matches={simple_matches})",
                latency_ms=latency,
                matched_keywords=simple_matches,
            )

        # Default to RAG for uncertain cases (safe fallback)
        latency = (time.perf_counter() - start) * 1000
        return ClassificationResult(
            route=PipelineRoute.RAG,
            confidence=0.6,
            reason="default fallback to RAG",
            latency_ms=latency,
            matched_keywords=[],
        )
