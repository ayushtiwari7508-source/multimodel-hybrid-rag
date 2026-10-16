"""Backward-compatibility shim for app.observability. Use backend.app.core.observability instead."""

from backend.app.core.observability import (
    LOG_DIR,
    JsonFormatter,
    Metrics,
    RequestIdFilter,
    Timer,
    get_logger,
    get_request_id,
    metrics,
    set_request_id,
    stable_hash,
)

__all__ = [
    "LOG_DIR",
    "JsonFormatter",
    "Metrics",
    "RequestIdFilter",
    "Timer",
    "get_logger",
    "get_request_id",
    "metrics",
    "set_request_id",
    "stable_hash",
]
