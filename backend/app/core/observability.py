"""Observability: structured logging, request IDs, metrics."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Context variable for request ID (thread-safe)
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(rid: str | None = None) -> str:
    """Set a new request ID for the current context. Returns the ID."""
    rid = rid or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Inject request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Extra fields
        for k, v in record.__dict__.items():
            if k not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "request_id",
                "taskName",
            }:
                try:
                    json.dumps(v)
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = str(v)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, json: bool = False) -> logging.Logger:
    """Get a logger with structured formatting and request ID tracking."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.addFilter(RequestIdFilter())

    # File handler (always JSON for parsing)
    fh = logging.FileHandler(LOG_DIR / "app.log")
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)

    if json:
        # Console JSON
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(JsonFormatter())
    else:
        # Rich console (human-readable)
        sh = RichHandler(rich_tracebacks=True, markup=False, show_path=False)
        sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# Simple in-memory metrics store
class Metrics:
    """Lightweight metrics counter/histogram. No external dependencies."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._timings: dict[str, list[float]] = {}
        self._max_samples = 1000

    def incr(self, name: str, value: int = 1, **labels: str) -> None:
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        samples = self._timings.setdefault(key, [])
        samples.append(value)
        if len(samples) > self._max_samples:
            samples.pop(0)

    @staticmethod
    def _key(name: str, labels: dict) -> str:
        if not labels:
            return name
        lbl = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{lbl}}}"

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {"counters": dict(self._counters), "timings": {}}
        for k, samples in self._timings.items():
            if not samples:
                continue
            sorted_s = sorted(samples)
            n = len(sorted_s)
            out["timings"][k] = {
                "count": n,
                "mean": sum(samples) / n,
                "p50": sorted_s[n // 2],
                "p95": sorted_s[min(n - 1, int(n * 0.95))],
                "p99": sorted_s[min(n - 1, int(n * 0.99))],
                "min": sorted_s[0],
                "max": sorted_s[-1],
            }
        return out

    def reset(self) -> None:
        self._counters.clear()
        self._timings.clear()


metrics = Metrics()


class Timer:
    """Context manager for timing an operation."""

    def __init__(self, name: str, **labels: str) -> None:
        self.name = name
        self.labels = labels
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
        metrics.observe(f"{self.name}.ms", self.elapsed_ms, **self.labels)


def stable_hash(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
