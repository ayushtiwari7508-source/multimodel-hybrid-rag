"""Multimodal document loaders, OCR, equation, table, and image processing."""

from app.models.domain import DocElement, LoadedDocument
from app.multimodal.loaders import (
    _describe_image_ollama,
    _extract_equations_heuristic,
    _extract_tables_camelot,
    _extract_tables_pdfplumber,
    _ocr_image,
    load_document,
    load_docx,
    load_markdown,
    load_paths,
    load_pdf,
    load_pptx,
    load_text,
    load_xlsx,
)

__all__ = [
    "DocElement",
    "LoadedDocument",
    "_describe_image_ollama",
    "_extract_equations_heuristic",
    "_extract_tables_camelot",
    "_extract_tables_pdfplumber",
    "_ocr_image",
    "load_document",
    "load_docx",
    "load_markdown",
    "load_paths",
    "load_pdf",
    "load_pptx",
    "load_text",
    "load_xlsx",
]
