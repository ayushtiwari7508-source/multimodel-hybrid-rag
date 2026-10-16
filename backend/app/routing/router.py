"""Query routing and classification."""
from __future__ import annotations

import json
import re

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger
from backend.app.generation.llm import OllamaClient
from backend.app.models.domain import Route, RouteName

log = get_logger(__name__)

ROUTER_SYSTEM = """You are a query router for a multi-model RAG system.
Given a user question, decide which model should answer it.
Return ONLY valid JSON with keys: route, reason.

Routes:
- "code": programming, code generation, debugging, refactoring, regex, shell, SQL.
- "math": equations, derivations, formal math, proofs, scientific formulas.
- "multimodal": questions that reference or depend on figures, charts, diagrams, tables,
  images, equations, or non-textual content.
- "fast": very short, simple, or chit-chat style questions, or when speed matters.
- "general": anything else.

Pick exactly one route. Be decisive. The 'reason' should be one short sentence."""


class ModelRouter:
    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama
        self.cfg = get_settings()

    def _model_for(self, route: RouteName) -> str:
        return {
            "general": self.cfg.general_model,
            "code": self.cfg.code_model,
            "math": self.cfg.math_model,
            "fast": self.cfg.fast_model,
            "multimodal": self.cfg.general_model,
        }[route]

    def _heuristic(self, query: str) -> RouteName:
        q = query.lower().strip()
        if len(q.split()) <= 4:
            return "fast"
        if re.search(r"\b(code|function|bug|error|python|javascript|sql|class|api|regex|compile|syntax)\b", q):
            return "code"
        if re.search(r"[=∫∑√π]|\\$|equation|derivative|integral|matrix|proof|theorem|formula", q):
            return "math"
        if re.search(r"\b(chart|figure|diagram|image|table|graph|plot|axis|legend)\b", q):
            return "multimodal"
        return "general"

    def route(self, query: str) -> Route:
        if get_settings().skip_router:
            primary = self._heuristic(query)
            return Route(name=primary, model=self._model_for(primary), reason="router skipped (heuristic only)")
        primary = self._heuristic(query)
        try:
            raw = self.ollama.generate(
                prompt=f"Query:\n{query}\n\nReturn JSON only.",
                model=self.cfg.router_model,
                system=ROUTER_SYSTEM,
                options={"temperature": 0.0, "num_predict": 200},
            )
            text = raw.strip() if isinstance(raw, str) else ""
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                name = str(data.get("route", primary)).lower()
                if name in {"general", "code", "math", "fast", "multimodal"}:
                    return Route(name=name, model=self._model_for(name), reason=str(data.get("reason", ""))[:200])
        except Exception as e:
            log.warning(f"LLM router failed, using heuristic: {e}")
        return Route(name=primary, model=self._model_for(primary), reason="heuristic fallback")
