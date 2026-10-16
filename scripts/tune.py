"""Parameter grid search for chunk_size, overlap, top_k values."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.pipeline import RAGPipeline
from scripts.eval_dataset import load_dataset


def build_kb() -> Path:
    """Create a more substantial knowledge base for realistic testing."""
    td = tempfile.mkdtemp()
    doc = Path(td) / "kb.md"
    doc.write_text(
        "# RAG System Documentation\n\n"
        "## Overview\n\n"
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. "
        "A hybrid RAG system uses both sparse (BM25) and dense (vector) retrieval, "
        "fused via Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking.\n\n"
        "## Architecture\n\n"
        "### Retrieval Layer\n"
        "The retrieval layer has two branches:\n"
        "1. **Sparse retrieval (BM25)**: Tokenizes queries and documents, scores using term frequency and inverse document frequency.\n"
        "2. **Dense retrieval**: Uses sentence transformers (e.g., nomic-embed-text) to embed queries and chunks into vector space.\n"
        "Results are fused using RRF with parameter C=60.\n\n"
        "### Reranking\n"
        "A cross-encoder (bge-reranker-v2-m3) re-scores the top candidates. "
        "If unavailable, falls back to weighted blend: 0.6 * RRF + 0.4 * (normalized dense + BM25).\n\n"
        "### Generation\n"
        "A router (qwen2.5:3b) classifies queries into routes:\n"
        "- **code**: qwen2.5-coder:7b (Python, SQL, regex, debugging)\n"
        "- **math**: qwen2.5:7b (equations, proofs, formulas)\n"
        "- **multimodal**: qwen2.5:7b (tables, charts, images)\n"
        "- **fast**: qwen2.5:3b (short, chit-chat)\n"
        "- **general**: qwen2.5:7b (default)\n\n"
        "### Equations\n\n"
        "Cosine similarity: $cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$.\n"
        "RRF score: $RRF(d) = \\sum \\frac{1}{C + rank(d)}$ where $C=60$.\n\n"
        "### Tables\n\n"
        "| Parameter | Default | Description |\n"
        "|-----------|---------|-------------|\n"
        "| chunk_size | 900 | Token window for chunking |\n"
        "| chunk_overlap | 150 | Overlap between adjacent chunks |\n"
        "| top_k_bm25 | 25 | Sparse retrieval fan-out |\n"
        "| top_k_dense | 25 | Dense retrieval fan-out |\n"
        "| top_k_rerank | 6 | Final context size |\n"
        "| rrf_c | 60 | RRF smoothing constant |\n\n"
        "## Code Examples\n\n"
        "```python\n"
        "def cosine_similarity(a, b):\n"
        "    import numpy as np\n"
        "    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\n"
        "```\n\n"
        "```sql\n"
        "SELECT * FROM documents WHERE content LIKE '%RAG%';\n"
        "```\n\n"
        "## References\n\n"
        '- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020\n'
        '- Gao et al., "Rerankers for Hybrid Retrieval", 2023\n'
        'Rao et al., "Reciprocal Rank Fusion for Hybrid Search", 2009\n',
        encoding="utf-8",
    )
    return doc


def run_config(
    chunk_size: int,
    chunk_overlap: int,
    top_k_bm25: int,
    top_k_dense: int,
    top_k_rerank: int,
) -> dict[str, Any]:
    """Run eval with a specific config."""
    # Monkey-patch settings before pipeline init
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    os.environ["TOP_K_BM25"] = str(top_k_bm25)
    os.environ["TOP_K_DENSE"] = str(top_k_dense)
    os.environ["TOP_K_RERANK"] = str(top_k_rerank)

    get_settings.cache_clear()

    pipeline = RAGPipeline()
    if not pipeline.ollama.health():
        return {"error": "ollama_unreachable"}

    # Build KB
    kb_path = build_kb()
    try:
        pipeline.clear()
        res = pipeline.ingest(paths=[str(kb_path)])
        chunks = res.get("chunks", 0)
        if chunks == 0:
            return {"error": "no_chunks_indexed"}

        dataset = load_dataset()
        results = []
        total_kw_recall = 0.0
        total_latency = 0.0
        total_hits = 0
        route_correct = 0
        route_total = 0

        for qa in dataset:
            t0 = time.perf_counter()
            try:
                r = pipeline.query(qa.question)
            except Exception as e:
                results.append({"question": qa.question, "error": str(e)})
                continue
            latency = (time.perf_counter() - t0) * 1000

            # Keyword recall
            lower = r.answer.lower()
            matched = sum(1 for k in qa.expected_keywords if k.lower() in lower)
            kw_recall = matched / len(qa.expected_keywords) if qa.expected_keywords else 1.0

            route_ok = (r.route.name == qa.route_hint) if qa.route_hint else None
            if route_ok is not None:
                route_total += 1
                if route_ok:
                    route_correct += 1

            total_kw_recall += kw_recall
            total_latency += latency
            total_hits += len(r.hits)

            results.append(
                {
                    "question": qa.question,
                    "route": r.route.name,
                    "route_ok": route_ok,
                    "kw_recall": kw_recall,
                    "hits": len(r.hits),
                    "latency_ms": latency,
                }
            )

        n = len(dataset)
        return {
            "config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k_bm25": top_k_bm25,
                "top_k_dense": top_k_dense,
                "top_k_rerank": top_k_rerank,
            },
            "chunks_indexed": chunks,
            "avg_kw_recall": total_kw_recall / n,
            "avg_latency_ms": total_latency / n,
            "avg_hits": total_hits / n,
            "route_accuracy": route_correct / route_total if route_total else None,
            "results": results,
        }
    finally:
        pipeline.close()


def main():
    # Grid: (chunk_size, overlap, top_k_bm25, top_k_dense, top_k_rerank)
    grid = [
        # Baseline
        (900, 150, 25, 25, 6),
        # Smaller chunks
        (500, 100, 25, 25, 6),
        (700, 100, 25, 25, 6),
        # Larger chunks
        (1200, 200, 25, 25, 6),
        # More retrieval
        (900, 150, 50, 50, 10),
        (900, 150, 25, 25, 10),
        # Fewer retrieval
        (900, 150, 10, 10, 4),
    ]

    all_results = []
    for cfg in grid:
        print(f"\n=== Testing {cfg} ===")
        r = run_config(*cfg)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(
            f"  chunks={r['chunks_indexed']} kw_recall={r['avg_kw_recall']:.3f} "
            f"latency={r['avg_latency_ms']:.0f}ms route_acc={r['route_accuracy']}"
        )
        all_results.append(r)

    # Save
    out = ROOT / "eval_tuning.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out}")

    # Find best
    if all_results:
        best = max(all_results, key=lambda x: x["avg_kw_recall"] / (x["avg_latency_ms"] / 10000))
        print(f"\nBest balance (kw_recall / normalized_latency): {best['config']}")
        print(f"  kw_recall={best['avg_kw_recall']:.3f} latency={best['avg_latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
