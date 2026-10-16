"""Health check routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import require_api_key, rate_limit
from backend.app.core.config import get_settings
from backend.app.services.query_service import QueryService

router = APIRouter(tags=["health"])

# Global pipeline instance
_pipeline: QueryService | None = None


def get_pipeline() -> QueryService:
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryService()
    return _pipeline


@router.get("/health")
def health() -> dict:
    pipe = get_pipeline()
    cur_settings = get_settings()
    ollama_ok = pipe.ollama.health()
    available = set(pipe.ollama.list_models()) if ollama_ok else set()

    def _have(tag: str) -> bool:
        return any(tag == a or a == f"{tag}:latest" or a.startswith(tag + ":") for a in available)

    required = {
        "embed": cur_settings.embed_model,
        "router": cur_settings.router_model,
        "general": cur_settings.general_model,
        "code": cur_settings.code_model,
    }
    optional = {"vision": cur_settings.vision_model}
    if cur_settings.rerank_backend.lower() == "ollama":
        optional["rerank"] = cur_settings.rerank_model
    missing_required = [label for label, tag in required.items() if not _have(tag)]
    missing_optional = [label for label, tag in optional.items() if not _have(tag)]

    notes: list[str] = []
    if not ollama_ok:
        notes.append(f"Ollama is not reachable at {cur_settings.ollama_base_url}. Start it with: ollama serve")
    if missing_required:
        notes.append(
            "Missing required models: "
            + ", ".join(f"{label}={required[label]}" for label in missing_required)
            + ". Run: ollama pull "
            + " ".join(required[label] for label in missing_required)
        )
    if missing_optional:
        for label in missing_optional:
            tag = optional[label]
            if label == "rerank":
                notes.append(
                    f"NOTE: rerank model '{tag}' not installed — the retriever will use the "
                    "weighted RRF + dense + BM25 fallback blend automatically."
                )
            else:
                notes.append(f"NOTE: optional model '{tag}' not installed.")

    return {
        "ok": ollama_ok and not missing_required,
        "ollama_reachable": ollama_ok,
        "available_models": sorted(available),
        "stats": pipe.stats(),
        "missing_required": missing_required,
        "notes": notes,
    }


@router.get("/metrics")
def metrics_endpoint() -> dict:
    """Return current metrics counters and timing histograms."""
    pipe = get_pipeline()
    from backend.app.core.observability import metrics
    return {
        "ok": True,
        "stats": pipe.stats(),
        "metrics": metrics.snapshot(),
    }
