from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from app.core.observability import get_logger, stable_hash


def chunk_text(s: str, n: int) -> Iterator[str]:
    for i in range(0, len(s), n):
        yield s[i : i + n]


__all__ = ["get_logger", "stable_hash", "chunk_text"]
