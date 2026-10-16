"""Prompt building and answer generation."""
from __future__ import annotations

from collections.abc import Iterator

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger
from backend.app.generation.llm import OllamaClient
from backend.app.models.domain import Answer, Citation, Hit, Route

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a careful, citation-grounded assistant for a hybrid RAG system.

Rules:
- Answer ONLY using the provided context. If the context does not contain the answer, say so explicitly.
- Cite sources inline as [1], [2], ... corresponding to the numbered CONTEXT blocks.
- At the end, output a "Sources:" list with the numbered entries (path, page/section, type).
- For multimodal content (tables, images, equations), preserve and reference the data exactly as given.
- Be concise but complete. Use markdown formatting.
- If a calculation is needed, show the steps.
- Do not fabricate citations or content.
"""


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def format_context(hits: list[Hit], max_tokens: int) -> tuple[str, list[Citation]]:
    blocks: list[str] = []
    citations: list[Citation] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        meta = h.meta or {}
        header = (
            f"[{i}] source={meta.get('source', '?')} page={meta.get('page', '-')} "
            f"section={meta.get('section', '-')} type={meta.get('element_type', 'text')}"
        )
        body = h.content.strip()
        block = f"{header}\n{body}"
        t = approx_tokens(block)
        if used + t > max_tokens:
            continue
        blocks.append(block)
        citations.append(
            Citation(
                index=i,
                source=str(meta.get("source", "?")),
                page=meta.get("page"),
                section=meta.get("section"),
                element_type=str(meta.get("element_type", "text")),
                snippet=body[:240] + ("..." if len(body) > 240 else ""),
            )
        )
        used += t
    return "\n\n---\n\n".join(blocks), citations


def format_sources(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["\n\n**Sources:**"]
    for c in citations:
        page = f" p.{c.page}" if c.page is not None else ""
        sec = f" / {c.section}" if c.section else ""
        lines.append(f"- [{c.index}] `{c.source}`{page}{sec} ({c.element_type})")
    return "\n".join(lines)


class Generator:
    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama
        self.cfg = get_settings()

    def _get_num_predict(self, route: Route) -> int:
        if route.name == "code":
            return 512
        if route.name == "fast":
            return 256
        if route.name == "math":
            return 768
        return 900

    def build_prompt(self, query: str, hits: list[Hit]) -> tuple[str, list[Citation]]:
        context, citations = format_context(hits, self.cfg.max_context_tokens)
        if not context:
            context = "(no relevant context found)"
        user = (
            "CONTEXT:\n"
            f"{context}\n\nQUESTION:\n{query}\n\n"
            "Answer with inline [n] citations and end with a Sources list."
        )
        return user, citations

    def answer(self, query: str, hits: list[Hit], route: Route) -> Answer:
        user, citations = self.build_prompt(query, hits)
        out = self.ollama.generate(
            prompt=user,
            model=route.model,
            system=SYSTEM_PROMPT,
            options={"temperature": 0.2, "num_predict": self._get_num_predict(route)},
        )
        text = out if isinstance(out, str) else ""
        text = text.rstrip() + format_sources(citations)
        return Answer(text=text, citations=citations, route=route, hits=hits)

    def stream(self, query: str, hits: list[Hit], route: Route) -> Iterator[str]:
        user, citations = self.build_prompt(query, hits)
        try:
            stream = self.ollama.generate(
                prompt=user,
                model=route.model,
                system=SYSTEM_PROMPT,
                options={"temperature": 0.2, "num_predict": self._get_num_predict(route)},
                stream=True,
            )
            if hasattr(stream, "__iter__") and not isinstance(stream, str):
                yield from stream
                yield format_sources(citations)
                return
        except Exception as e:
            log.warning(f"streaming failed, falling back: {e}")
        ans = self.answer(query, hits, route)
        yield ans.text
