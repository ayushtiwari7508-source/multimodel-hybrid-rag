from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request

from app.core.config import get_settings
from app.core.observability import metrics


def require_api_key(request: Request) -> None:
    """Validate X-API-Key header if API_KEY is configured."""
    current = get_settings()
    if not current.api_key:
        return  # auth disabled
    supplied = request.headers.get("x-api-key", "")
    if supplied != current.api_key:
        metrics.incr("auth.failures", path=request.url.path)
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class RateLimiter:
    def __init__(self, per_min: int) -> None:
        self.per_min = per_min
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = __import__("threading").Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            hits = self._buckets[key]
            # drop old entries
            self._buckets[key] = [t for t in hits if t > cutoff]
            if len(self._buckets[key]) >= self.per_min:
                retry = max(1, int(60 - (now - self._buckets[key][0])))
                return False, retry
            self._buckets[key].append(now)
            return True, 0


_rate_limiter = RateLimiter(get_settings().rate_limit_per_min)


def rate_limit(request: Request) -> None:
    """Per-IP rate limit; returns 429 if exceeded."""
    import sys

    main_mod = sys.modules.get("backend.app.api.main")
    limiter = getattr(main_mod, "_rate_limiter", None)
    if limiter is None:
        compat_mod = sys.modules.get("app.api")  # legacy compat surface
        limiter = getattr(compat_mod, "_rate_limiter", None) or _rate_limiter
    current = get_settings()

    ip = request.client.host if request.client else "unknown"
    allowed, retry = limiter.check(ip)
    if not allowed:
        metrics.incr("rate_limit.blocks", path=request.url.path)
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded ({current.rate_limit_per_min}/min). Retry after {retry}s",
            headers={"Retry-After": str(retry)},
        )
