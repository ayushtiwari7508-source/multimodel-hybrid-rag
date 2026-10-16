#!/usr/bin/env bash
# Pull all 6 required models sequentially in the background.
set -u
cd "$(dirname "$0")/.."
LOG=./data/logs/pull.log
: > "$LOG"
MODELS=(
  "nomic-embed-text"
  "qwen2.5:3b"
  "qwen2.5:7b"
  "qwen2.5-coder:7b"
  "llava:7b"
)
{
  echo "Starting sequential pull of ${#MODELS[@]} models"
  for m in "${MODELS[@]}"; do
    echo ""
    echo "==================== $(date -u +%FT%TZ) PULL: $m ===================="
    if ollama pull "$m"; then
      echo "OK: $m"
    else
      echo "FAIL: $m"
    fi
  done
  echo ""
  echo "==================== DONE ===================="
} >> "$LOG" 2>&1
