"""FastAPI application entry point."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.app.api.dependencies import (
    RateLimiter,
    rate_limit,
    require_api_key,
)
from backend.app.core.config import get_settings
from backend.app.core.observability import (
    get_logger,
    metrics,
    set_request_id,
)
from backend.app.schemas import (
    CitationModel,
    HitModel,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RouteModel,
)
from backend.app.services.query_service import QueryService

log = get_logger(__name__)

settings = get_settings()

# Global pipeline instance for backward compatibility
pipeline = QueryService()

app = FastAPI(title="Hybrid Multi-Model RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
from backend.app.api.routes import health, documents, query

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


def get_pipeline():
    return pipeline


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


# Keep the old main for backward compatibility in scripts
def serve():
    import uvicorn
    uvicorn.run("backend.app.api.main:app", host=settings.api_host, port=settings.api_port, reload=False)


if __name__ == "__main__":
    serve()
