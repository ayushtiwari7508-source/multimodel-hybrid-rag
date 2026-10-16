"""Convenience runner: streamlit for the chat UI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "frontend" / "app.py"

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

    from backend.app.core.config import get_settings

    s = get_settings()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(UI),
        "--server.port",
        str(s.streamlit_port),
        "--server.address",
        "0.0.0.0",
    ]
    sys.exit(subprocess.call(cmd))
