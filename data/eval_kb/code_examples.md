# Code Generation and Technical Documentation

## Python RAG Implementation

### Core Pipeline Class

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from app.core.config import get_settings
from app.generation.generator import Answer, Generator
from app.generation.ollama_client import OllamaClient
from app.generation.router import ModelRouter
from app.models.domain import Hit, Route
from app.pipeline.ingest import Ingestor
from app.retrieval.bm25 import BM25Store
from app.retrieval.classifier import PipelineRoute, QueryClassifier
from app.retrieval.dense import ChromaStore
from app.retrieval.hybrid import HybridRetriever, RetrievalMetrics

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

class RAGPipeline:
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
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        self._require_models()
        total_start = time.perf_counter()
        
        # Classify query
        classification = self.classifier.classify(question)
        pipeline_route = classification.route
        
        # Select model based on pipeline route
        route = self._select_model_for_pipeline_route(pipeline_route, question)
        
        # Retrieve
        hits, retrieval_metrics = self.retriever.retrieve(question, route=pipeline_route)
        
        if top_k:
            hits = hits[:top_k]
        
        # Generate answer
        gen_start = time.perf_counter()
        ans: Answer = self.generator.answer(question, hits, route)
        gen_latency = (time.perf_counter() - gen_start) * 1000
        
        total_latency = (time.perf_counter() - total_start) * 1000
        
        return RAGResponse(
            answer=ans.text,
            route=route,
            hits=hits,
            pipeline_route=pipeline_route,
            classification=classification.reason,
            retrieval_metrics=retrieval_metrics,
            total_latency_ms=total_latency,
        )
```

### Hybrid Retriever with RRF

```python
def _rrf(rankings: list[list[str]], c: int) -> dict[str, float]:
    """Reciprocal Rank Fusion implementation."""
    rrf: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (c + rank + 1)
    return rrf

def _fuse(self, dense: dict[str, Hit], bm25: dict[str, Hit]) -> tuple[list[Hit], float]:
    """Fuse dense and sparse results using RRF."""
    t0 = time.perf_counter()
    dense_ids = list(dense.keys())
    bm25_ids = list(bm25.keys())
    rrf = _rrf([dense_ids, bm25_ids], c=self.cfg.rrf_c)
    
    # Normalize scores
    dense_norm = {k: v.dense_score for k, v in dense.items()}
    bm25_norm_map = {k: v.bm25_score for k, v in bm25.items()}
    for arr in (dense_norm, bm25_norm_map):
        keys = list(arr.keys())
        norm_vals = _normalize_scores(list(arr.values()))
        for k, v in zip(keys, norm_vals, strict=True):
            arr[k] = v
    
    # Merge and rank
    merged: dict[str, Hit] = {}
    for hid in set(dense_ids) | set(bm25_ids):
        h = dense.get(hid) or bm25.get(hid)
        if h is None:
            continue
        h.rrf_score = rrf.get(hid, 0.0)
        h.dense_score = dense_norm.get(hid, 0.0)
        h.bm25_score = bm25_norm_map.get(hid, 0.0)
        h.final_score = h.rrf_score
        merged[hid] = h
    
    ranked = sorted(merged.values(), key=lambda h: h.final_score, reverse=True)
    elapsed = (time.perf_counter() - t0) * 1000
    return ranked, elapsed
```

### Cross-Encoder Reranking

```python
def _rerank(self, query: str, hits: list[Hit], top_k: int) -> tuple[list[Hit], float]:
    """Rerank candidates using cross-encoder."""
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
    
    norm = _normalize_scores(scores)
    for h, s in zip(cand, norm, strict=True):
        h.rerank_score = float(s)
        h.final_score = 0.6 * h.rrf_score + 0.4 * h.rerank_score
    
    cand.sort(key=lambda h: h.final_score, reverse=True)
    elapsed = (time.perf_counter() - t0) * 1000
    return cand[:top_k], elapsed
```

## SQL Examples for RAG

```sql
-- Find documents containing RAG-related terms
SELECT doc_id, source, COUNT(*) as chunk_count
FROM chroma_embeddings
WHERE document LIKE '%RAG%' OR document LIKE '%retrieval%'
GROUP BY doc_id, source
ORDER BY chunk_count DESC;

-- BM25-style keyword search on stored text
SELECT id, content, source
FROM documents
WHERE content LIKE '%hybrid%' AND content LIKE '%retrieval%'
ORDER BY LENGTH(content) ASC
LIMIT 10;

-- Get chunks for a specific document with scores
SELECT c.id, c.content, c.meta->>'page' as page, 
       c.meta->>'section' as section
FROM chunks c
WHERE c.meta->>'doc_id' = 'rag_basics'
ORDER BY c.meta->>'char_start' ASC;
```

## Regex Patterns for Text Processing

```python
import re

# Extract LaTeX equations
INLINE_EQ = re.compile(r'\$(.+?)\$')
DISPLAY_EQ = re.compile(r'\$\$(.+?)\$\$')
LATEX_INLINE = re.compile(r'\\\((.+?)\\\)')
LATEX_DISPLAY = re.compile(r'\\\[(.+?)\\\]')

# Extract code blocks
CODE_BLOCK = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
INLINE_CODE = re.compile(r'`([^`]+)`')

# Extract markdown tables
TABLE_PATTERN = re.compile(r'(\|.+\|\n)+(\|[-:]+\|.+\n)+(\|.+\|\n)*')

# Tokenize for BM25
TOKENIZE_RE = re.compile(r'\w+', re.UNICODE)
```

## Shell Commands for RAG Operations

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull llava:7b

# Run API server
python scripts/serve_api.py

# Run Streamlit UI
python scripts/serve_ui.py

# Run evaluation
python scripts/eval.py

# Run parameter tuning
python scripts/tune.py

# Backup index
python scripts/backup.py backup ./backups

# Restore index
python scripts/backup.py restore ./backups/latest.tar.gz
```

## Docker Commands

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f api

# Stop
docker compose down

# Clean volumes
docker compose down -v
```

## Configuration via Environment Variables

```bash
# .env file
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
RERANK_MODEL=BAAI/bge-reranker-base
RERANK_BACKEND=cross-encoder
VISION_MODEL=llava:7b
ROUTER_MODEL=qwen2.5:3b
GENERAL_MODEL=qwen2.5:7b
CODE_MODEL=qwen2.5-coder:7b
MATH_MODEL=qwen2.5:7b
FAST_MODEL=qwen2.5:3b
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K_BM25=25
TOP_K_DENSE=25
TOP_K_RERANK=6
RRF_C=60
MAX_CONTEXT_TOKENS=6000
API_KEY=your-secret-key
RATE_LIMIT_PER_MIN=60
```