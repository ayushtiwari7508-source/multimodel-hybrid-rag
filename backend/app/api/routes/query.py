"""Query routes."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.app.api.dependencies import rate_limit, require_api_key
from backend.app.schemas import (
    CitationModel,
    HitModel,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RouteModel,
)
from backend.app.services.query_service import QueryService

router = APIRouter(tags=["query"])

# Global pipeline instance
_pipeline: QueryService | None = None


def get_pipeline() -> QueryService:
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryService()
    return _pipeline


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    pipe = get_pipeline()
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


@router.get("/query/stream", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def query_stream(question: str, top_k: Optional[int] = None):
    """Server-Sent Events streaming endpoint for real-time answer generation."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    async def event_generator():
        try:
            pipe = get_pipeline()
            route, hits, stream = pipe.stream(question, top_k=top_k)

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
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
