"""Backward-compatibility shim for app.core.config. Use backend.app.core.config instead."""
from backend.app.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
