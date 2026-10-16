"""Backward-compatibility shim for app.loaders. Use backend.app.ingestion instead."""

from backend.app.ingestion.loaders import (
    load_document,
    load_docx,
    load_markdown,
    load_paths,
    load_pdf,
    load_pptx,
    load_text,
    load_xlsx,
)
from backend.app.models.domain import DocElement, LoadedDocument

__all__ = [
    "DocElement",
    "LoadedDocument",
    "load_document",
    "load_docx",
    "load_markdown",
    "load_paths",
    "load_pdf",
    "load_pptx",
    "load_text",
    "load_xlsx",
]
