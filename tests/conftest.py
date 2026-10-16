"""Shared pytest fixtures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    """Isolated data dir for Chroma/BM25/pickle so tests don't touch user state."""
    d = tmp_path / "data"
    (d / "chroma").mkdir(parents=True)
    (d / "bm25").mkdir(parents=True)
    (d / "index_meta").mkdir(parents=True)
    (d / "logs").mkdir(parents=True)
    os.environ["CHROMA_PERSIST_DIR"] = str(d / "chroma")
    os.environ["BM25_PERSIST_PATH"] = str(d / "bm25" / "index.pkl")
    os.environ["META_PERSIST_DIR"] = str(d / "index_meta")
    yield d
    shutil.rmtree(d, ignore_errors=True)
