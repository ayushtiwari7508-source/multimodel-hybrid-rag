"""Backward-compatibility shim for app.generator. Use backend.app.generation instead."""

from backend.app.generation import (
    SYSTEM_PROMPT,
    Generator,
    _approx_tokens,
    _format_context,
    _format_sources,
)
from backend.app.models.domain import Answer, Citation

__all__ = [
    "SYSTEM_PROMPT",
    "Answer",
    "Citation",
    "Generator",
    "_approx_tokens",
    "_format_context",
    "_format_sources",
]
