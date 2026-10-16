"""Pull all required Ollama models. Idempotent."""

from __future__ import annotations

import subprocess
import sys

from app.config import get_settings

MODELS = [
    "nomic-embed-text",
    "llava:7b",
    "qwen2.5:3b",
    "qwen2.5:7b",
    "qwen2.5-coder:7b",
]


def main() -> int:
    s = get_settings()
    base = s.ollama_base_url
    print(f"Using Ollama at {base}")
    for m in MODELS:
        print(f"\n>>> ollama pull {m}")
        rc = subprocess.run(["ollama", "pull", m]).returncode
        if rc != 0:
            print(f"Failed to pull {m}")
            return rc
    print("\nAll models pulled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
