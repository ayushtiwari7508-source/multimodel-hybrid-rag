"""Ingestion loaders package."""
from backend.app.ingestion.loaders.loaders import (
    load_document,
    load_docx,
    load_markdown,
    load_pdf,
    load_pptx,
    load_text,
    load_xlsx,
    load_paths,
)

__all__ = [
    "load_document",
    "load_docx",
    "load_markdown",
    "load_pdf",
    "load_pptx",
    "load_text",
    "load_xlsx",
    "load_paths",
]
