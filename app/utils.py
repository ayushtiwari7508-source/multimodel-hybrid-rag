"""Backward-compatibility shim for app.utils. Use app.utils or app.core.observability instead."""

from app.core.observability import get_logger, stable_hash
from app.utils.helpers import chunk_text

__all__ = ["chunk_text", "get_logger", "stable_hash"]
