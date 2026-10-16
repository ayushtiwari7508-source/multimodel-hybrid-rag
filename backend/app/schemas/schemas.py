from __future__ import annotations

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str
    source: str | None = "user-text"


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class CitationModel(BaseModel):
    index: int
    source: str
    page: int | None = None
    section: str | None = None
    element_type: str | None = None
    snippet: str


class HitModel(BaseModel):
    id: str
    content: str
    meta: dict = Field(default_factory=dict)
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    source: str = ""


class RouteModel(BaseModel):
    name: str
    model: str
    reason: str = ""


class QueryResponse(BaseModel):
    answer: str
    route: RouteModel
    pipeline_route: str = "rag"
    classification: str = ""
    hits: list[HitModel]
    citations: list[CitationModel]
    latency_ms: float = 0.0
