"""Convenience runner: uvicorn for the FastAPI app."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("backend.app.api.main:app", host=s.api_host, port=s.api_port, reload=False)
