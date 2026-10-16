from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.dependencies import (
    rate_limit,
    require_api_key,
)
from app.core.config import get_settings
from app.core.observability import (
    get_logger,
    metrics,
    set_request_id,
)
from app.models.schemas import (
    CitationModel,
    HitModel,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RouteModel,
)
from app.pipeline.rag_pipeline import RAGPipeline

log = get_logger(__name__)

settings = get_settings()
pipeline = RAGPipeline()
app = FastAPI(title="Hybrid Multi-Model RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline():
    api_mod = sys.modules.get("app.api")
    if api_mod and hasattr(api_mod, "pipeline"):
        return api_mod.pipeline
    return pipeline


def _get_settings():
    return get_settings()


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Assign request ID, log request, time it, expose ID in response."""
    rid = request.headers.get("x-request-id") or set_request_id()
    set_request_id(rid)
    metrics.incr("http.requests", method=request.method, path=request.url.path)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        metrics.observe("http.latency.ms", elapsed, method=request.method, path=request.url.path)
        metrics.incr("http.errors", method=request.method, path=request.url.path)
        log.exception(f"unhandled error on {request.method} {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": str(exc), "request_id": rid},
            headers={"x-request-id": rid},
        )
    elapsed = (time.perf_counter() - t0) * 1000
    metrics.observe("http.latency.ms", elapsed, method=request.method, path=request.url.path)
    if response.status_code >= 400:
        metrics.incr("http.errors", method=request.method, path=request.url.path)
    response.headers["x-request-id"] = rid
    log.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.1f}ms)")
    return response


@app.get("/metrics")
def metrics_endpoint() -> dict:
    """Return current metrics counters and timing histograms."""
    pipe = _get_pipeline()
    return {
        "ok": True,
        "stats": pipe.stats(),
        "metrics": metrics.snapshot(),
    }


@app.get("/health")
def health() -> dict:
    pipe = _get_pipeline()
    cur_settings = _get_settings()
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


@app.get("/documents", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def list_documents() -> dict:
    pipe = _get_pipeline()
    return {"ok": True, "documents": pipe.list_documents(), "stats": pipe.stats()}


@app.delete("/documents/{doc_id}", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def delete_document(doc_id: str) -> dict:
    if not doc_id or doc_id.strip() in {".", "..", "/"}:
        raise HTTPException(status_code=400, detail="Invalid doc_id")
    pipe = _get_pipeline()
    res = pipe.delete_document(doc_id)
    if res["deleted_chroma"] == 0 and res["deleted_bm25"] == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in index")
    return {"ok": True, **res, "stats": pipe.stats()}


@app.post("/ingest/text", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def ingest_text(req: IngestTextRequest) -> dict:
    pipe = _get_pipeline()
    res = pipe.ingest(text=req.text, source=req.source)
    return {"ok": True, **res, "stats": pipe.stats()}


@app.post("/ingest/upload", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def ingest_upload(files: list[UploadFile] = File(...)) -> dict:
    save_dir = Path("./data/uploads")
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for f in files:
        if not f.filename:
            continue
        dest = save_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        paths.append(str(dest))
    if not paths:
        raise HTTPException(status_code=400, detail="No files uploaded")
    pipe = _get_pipeline()
    res = pipe.ingest(paths=paths)
    return {"ok": True, "files": paths, **res, "stats": pipe.stats()}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    pipe = _get_pipeline()
    res = pipe.query(req.question, top_k=req.top_k)
    hits = [
        HitModel(
            id=str(h.id),
            content=str(h.content),
            meta=h.meta or {},
            rrf_score=float(getattr(h, "rrf_score", 0.0) or 0.0),
            rerank_score=float(getattr(h, "rerank_score", 0.0) or 0.0),
            final_score=float(getattr(h, "final_score", 0.0) or 0.0),
            source=str(getattr(h, "source", "") or ""),
        )
        for h in res.hits
    ]
    citations = [
        CitationModel(
            index=i,
            source=str(h.meta.get("source", "?")),
            page=h.meta.get("page"),
            section=h.meta.get("section"),
            element_type=str(h.meta.get("element_type", "text")),
            snippet=str(h.content)[:240] + ("..." if len(str(h.content)) > 240 else ""),
        )
        for i, h in enumerate(res.hits, start=1)
    ]
    answer = str(res.answer)
    if "\n\n**Sources:**" in answer:
        answer = answer.split("\n\n**Sources:**", 1)[0]
    route_name = str(getattr(res.route, "name", "general"))
    route_model = str(getattr(res.route, "model", "default"))
    route_reason = str(getattr(res.route, "reason", ""))
    pipeline_route = str(getattr(res, "pipeline_route", "rag"))
    classification = str(getattr(res, "classification", ""))
    latency_ms = float(getattr(res, "total_latency_ms", 0.0))
    return QueryResponse(
        answer=answer.strip(),
        route=RouteModel(name=route_name, model=route_model, reason=route_reason),
        pipeline_route=pipeline_route,
        classification=classification,
        hits=hits,
        citations=citations,
        latency_ms=latency_ms,
    )


@app.post("/clear", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def clear() -> dict:
    pipe = _get_pipeline()
    pipe.clear()
    return {"ok": True, "stats": pipe.stats()}


@app.get("/query/stream", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def query_stream(question: str, top_k: int | None = None):
    """Server-Sent Events streaming endpoint for real-time answer generation."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    async def event_generator():
        try:
            pipe = _get_pipeline()
            route, hits, stream = pipe.stream(question, top_k=top_k)
            import json

            # Get pipeline route from the last query (if available)
            pipeline_route = "rag"
            classification = ""
            if hasattr(pipe, "_last_classification"):
                pipeline_route = pipe._last_classification.get("pipeline_route", "rag")
                classification = pipe._last_classification.get("reason", "")

            meta = {
                "type": "meta",
                "route": str(getattr(route, "name", "general")),
                "model": str(getattr(route, "model", "default")),
                "reason": str(getattr(route, "reason", "")),
                "pipeline_route": pipeline_route,
                "classification": classification,
                "hits": [
                    {
                        "id": str(h.id),
                        "source": h.meta.get("source"),
                        "page": h.meta.get("page"),
                        "section": h.meta.get("section"),
                        "score": float(getattr(h, "final_score", 0.0) or 0.0),
                    }
                    for h in hits
                ],
            }
            yield f"data: {json.dumps(meta)}\n\n"

            for token in stream:
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            citations = [
                {
                    "index": i,
                    "source": str(h.meta.get("source", "?")),
                    "page": h.meta.get("page"),
                    "section": h.meta.get("section"),
                    "element_type": str(h.meta.get("element_type", "text")),
                    "snippet": str(h.content)[:240] + ("..." if len(str(h.content)) > 240 else ""),
                }
                for i, h in enumerate(hits, start=1)
            ]
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception as e:
            import json

            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
