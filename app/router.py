"""Backward-compatibility shim for app.router. Use backend.app.routing instead."""

from backend.app.models.domain import Route, RouteName
from backend.app.routing import ROUTER_SYSTEM, ModelRouter

__all__ = ["ROUTER_SYSTEM", "ModelRouter", "Route", "RouteName"]
