"""Generation package."""
from backend.app.generation.llm import OllamaClient
from backend.app.generation.prompts import (
    Generator, 
    SYSTEM_PROMPT, 
    approx_tokens, 
    format_context, 
    format_sources,
    # Aliases for backward compatibility
    approx_tokens as _approx_tokens,
    format_context as _format_context,
    format_sources as _format_sources,
)

__all__ = [
    "OllamaClient", "Generator", "SYSTEM_PROMPT", 
    "approx_tokens", "format_context", "format_sources",
    "_approx_tokens", "_format_context", "_format_sources",
]
