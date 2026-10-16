"""Backward compatibility shim for app package."""
import sys
from pathlib import Path

# Add backend to path for imports
BACKEND_ROOT = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

# Re-export from backend.app
from backend.app import *

__all__ = ["__version__"]
