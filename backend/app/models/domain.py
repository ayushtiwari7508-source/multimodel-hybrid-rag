from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RouteName = Literal["general", "code", "math", "fast", "multimodal"]


@dataclass
class Route:
    name: RouteName
    model: str
    reason: str = ""


@dataclass
class Hit:
    id: str
    content: str
    meta: dict
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    source: str = ""


@dataclass
class Citation:
    index: int
    source: str
    page: int | None
    section: str | None
    element_type: str | None
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    route: Route
    hits: list[Hit]


@dataclass
class DocElement:
    type: str  # "text" | "table" | "image" | "equation" | "ocr"
    content: str
    page: int | None = None
    section: str | None = None
    bbox: tuple | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    source: str
    doc_id: str
    elements: list[DocElement] = field(default_factory=list)

    def to_chunks(self, chunk_size: int, overlap: int) -> list[dict]:
        chunks: list[dict] = []
        text_buf: list[str] = []
        buf_meta: dict | None = None

        def flush() -> None:
            nonlocal text_buf, buf_meta
            if not text_buf:
                return
            text = "\n".join(text_buf).strip()
            if not text:
                text_buf = []
                buf_meta = None
                return
            start = 0
            while start < len(text):
                piece = text[start : start + chunk_size]
                chunks.append(
                    {
                        "content": piece,
                        "meta": {**(buf_meta or {}), "char_start": start, "char_end": start + len(piece)},
                    }
                )
                if start + chunk_size >= len(text):
                    break
                start += max(1, chunk_size - overlap)
            text_buf = []
            buf_meta = None

        for el in self.elements:
            if el.type == "text" and el.content.strip():
                text_buf.append(el.content)
                buf_meta = {
                    "source": self.source,
                    "doc_id": self.doc_id,
                    "page": el.page,
                    "section": el.section,
                }
                if len("\n".join(text_buf)) >= chunk_size:
                    flush()
            else:
                flush()
                prefix = f"[{el.type.upper()}] "
                if el.type == "image":
                    prefix = "[IMAGE DESCRIPTION] "
                elif el.type == "table":
                    prefix = "[TABLE]\n"
                elif el.type == "equation":
                    prefix = "[EQUATION] "
                elif el.type == "ocr":
                    prefix = "[OCR] "
                chunks.append(
                    {
                        "content": prefix + el.content,
                        "meta": {
                            "source": self.source,
                            "doc_id": self.doc_id,
                            "page": el.page,
                            "section": el.section,
                            "element_type": el.type,
                        },
                    }
                )
        flush()
        return chunks
