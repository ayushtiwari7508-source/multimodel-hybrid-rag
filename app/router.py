"""Backward-compatibility shim for app.router. Use backend.app.routing instead."""
from backend.app.routing import ROUTER_SYSTEM, ModelRouter
from backend.app.models.domain import Route, RouteName

__all__ = ["RouteName", "Route", "ROUTER_SYSTEM", "ModelRouter"]
