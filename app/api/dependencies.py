"""Backward-compatibility shim for app.api.dependencies. Use backend.app.api.dependencies instead."""

from backend.app.api.dependencies import (
    RateLimiter,
    _rate_limiter,
    rate_limit,
    require_api_key,
)

__all__ = ["RateLimiter", "_rate_limiter", "rate_limit", "require_api_key"]
