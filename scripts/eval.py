"""Evaluation harness: runs queries, measures hit rate, citation precision, route accuracy."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.query_service import QueryService
from scripts.eval_dataset import QAPair, load_dataset


@dataclass
class EvalResult:
    question: str
    route: str
    route_correct: bool | None
    answer: str
    hits: int
    latency_ms: float
    keyword_recall: float
    keyword_matched: list[str]
    keyword_missed: list[str]
    citation_precision: float | None = None
    source_relevant: bool | None = None


def keyword_recall(answer: str, expected: list[str]) -> tuple[float, list[str], list[str]]:
    if not expected:
        return 1.0, [], []
    lower = answer.lower()
    matched = [k for k in expected if k.lower() in lower]
    missed = [k for k in expected if k.lower() not in lower]
    return len(matched) / len(expected), matched, missed


def citation_precision(hits: list, expected_source: str | None) -> tuple[float | None, bool | None]:
    if expected_source is None:
        return None, None
    relevant = [h for h in hits if expected_source.lower() in (h.meta.get("source") or "").lower()]
    prec = len(relevant) / len(hits) if hits else 0.0
    return prec, len(relevant) > 0


def run_eval(pipeline: QueryService, dataset: list[QAPair]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for qa in dataset:
        t0 = time.perf_counter()
        try:
            res = pipeline.query(qa.question)
        except Exception as e:
            results.append(
                EvalResult(
                    question=qa.question,
                    route="ERROR",
                    route_correct=None,
                    answer=f"ERROR: {e}",
                    hits=0,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    keyword_recall=0.0,
                    keyword_matched=[],
                    keyword_missed=qa.expected_keywords,
                )
            )
            continue

        latency = (time.perf_counter() - t0) * 1000
        kw_recall, matched, missed = keyword_recall(res.answer, qa.expected_keywords)
        cit_prec, src_rel = citation_precision(res.hits, qa.expected_source_contains)
        route_ok = (res.route.name == qa.route_hint) if qa.route_hint else None

        results.append(
            EvalResult(
                question=qa.question,
                route=res.route.name,
                route_correct=route_ok,
                answer=res.answer,
                hits=len(res.hits),
                latency_ms=latency,
                keyword_recall=kw_recall,
                keyword_matched=matched,
                keyword_missed=missed,
                citation_precision=cit_prec,
                source_relevant=src_rel,
            )
        )
        print(f"  Q: {qa.question[:60]}...")
        print(
            f"    route={res.route.name} ({'✓' if route_ok else '✗' if route_ok is False else '?'}) "
            f"kw_recall={kw_recall:.2f} hits={len(res.hits)} latency={latency:.0f}ms"
        )
    return results


def summarize(results: list[EvalResult]) -> dict:
    n = len(results)
    return {
        "total_questions": n,
        "route_accuracy": sum(1 for r in results if r.route_correct is True)
        / sum(1 for r in results if r.route_correct is not None)
        if any(r.route_correct is not None for r in results)
        else None,
        "avg_keyword_recall": sum(r.keyword_recall for r in results) / n,
        "avg_latency_ms": sum(r.latency_ms for r in results) / n,
        "avg_hits": sum(r.hits for r in results) / n,
        "citation_precision_avg": sum(r.citation_precision for r in results if r.citation_precision is not None)
        / sum(1 for r in results if r.citation_precision is not None)
        if any(r.citation_precision is not None for r in results)
        else None,
        "source_relevant_rate": sum(1 for r in results if r.source_relevant is True)
        / sum(1 for r in results if r.source_relevant is not None)
        if any(r.source_relevant is not None for r in results)
        else None,
        "by_route": {
            route: {
                "count": sum(1 for r in results if r.route == route),
                "avg_kw_recall": sum(r.keyword_recall for r in results if r.route == route)
                / sum(1 for r in results if r.route == route),
                "avg_latency_ms": sum(r.latency_ms for r in results if r.route == route)
                / sum(1 for r in results if r.route == route),
            }
            for route in {r.route for r in results}
        },
    }


def main():
    print("Starting RAG evaluation...")
    pipeline = QueryService()

    if not pipeline.ollama.health():
        print("ERROR: Ollama not reachable", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} QA pairs")

    # Clear and re-ingest comprehensive evaluation KB
    pipeline.clear()
    eval_kb_dir = ROOT / "data" / "eval_kb"
    if eval_kb_dir.exists():
        md_files = list(eval_kb_dir.glob("*.md"))
        print(f"Ingesting {len(md_files)} evaluation documents...")
        res = pipeline.ingest(paths=[str(f) for f in md_files])
        print(f"Indexed {res['chunks']} chunks from {len(res['docs'])} documents")
    else:
        print("WARNING: eval_kb directory not found, using minimal KB")
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            doc = Path(td) / "eval_kb.md"
            doc.write_text(
                "# Knowledge Base\n\n"
                "Hybrid RAG combines BM25 sparse retrieval with dense vector search "
                "and a cross-encoder reranker. The router picks the best model per query.\n\n"
                "## Equations\n\n"
                "Cosine similarity is defined as: $cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$.\n\n"
                "## Tables\n\n"
                "| Route | Model |\n|---|---|\n| code | qwen-coder |\n| general | qwen |\n",
                encoding="utf-8",
            )
            pipeline.ingest(paths=[str(doc)])

    print("Running queries...")
    results = run_eval(pipeline, dataset)

    summary = summarize(results)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Save detailed results
    out_path = ROOT / "eval_results.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nDetailed results saved to {out_path}")

    pipeline.close()
    return 0 if summary["avg_keyword_recall"] > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
