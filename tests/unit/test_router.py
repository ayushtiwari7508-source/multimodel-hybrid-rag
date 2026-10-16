"""Tests for the rule-based router fallback."""

from __future__ import annotations

from app.router import ModelRouter


class _StubOllama:
    """Always raises so we exercise the heuristic path."""

    def generate(self, *args, **kwargs):
        raise RuntimeError("ollama down")


def _make_router() -> ModelRouter:
    r = ModelRouter.__new__(ModelRouter)
    r.ollama = _StubOllama()
    from app.config import get_settings

    r.cfg = get_settings()
    r.cfg.skip_router = False  # test router LLM behavior
    return r


def test_heuristic_short_queries_go_fast():
    r = _make_router()
    for q in ["hi", "ok", "thanks!", "what?"]:
        assert r._heuristic(q) == "fast"


def test_heuristic_code_keywords():
    r = _make_router()
    for q in [
        "write a python function to sort a list",
        "fix this bug in my javascript",
        "compile error in my code",
        "how does this SQL query work",
    ]:
        assert r._heuristic(q) == "code"


def test_heuristic_math_signals():
    r = _make_router()
    for q in [
        "Solve x^2 + 5x + 6 = 0",
        "Compute the integral of sin(x) dx",
        "Show me the proof of the Pythagorean theorem",
    ]:
        assert r._heuristic(q) == "math"


def test_heuristic_multimodal_signals():
    r = _make_router()
    for q in [
        "What does the chart on page 3 show?",
        "Describe the figure in the diagram",
        "Read the data from the table",
    ]:
        assert r._heuristic(q) == "multimodal"


def test_heuristic_general_fallback():
    r = _make_router()
    long = "Explain the differences between TCP and UDP and when each is preferable for streaming media"
    assert r._heuristic(long) == "general"


def test_route_falls_back_to_heuristic_when_ollama_fails():
    r = _make_router()
    route = r.route("hi")
    assert route.name == "fast"
    assert route.reason == "heuristic fallback"
    assert route.model == r.cfg.fast_model


def test_route_accepts_valid_json_from_ollama():
    class _GoodOllama:
        def generate(self, *args, **kwargs):
            return '{"route":"code","reason":"user asked for code"}'

    r = ModelRouter.__new__(ModelRouter)
    r.ollama = _GoodOllama()
    from app.config import get_settings

    r.cfg = get_settings()
    r.cfg.skip_router = False  # test router LLM behavior
    route = r.route("write a regex for emails")
    assert route.name == "code"
    assert route.model == r.cfg.code_model
    assert "code" in route.reason.lower()


def test_route_rejects_invalid_route_and_uses_heuristic():
    class _BadJsonOllama:
        def generate(self, *args, **kwargs):
            return '{"route":"unicorn","reason":"made up"}'

    r = ModelRouter.__new__(ModelRouter)
    r.ollama = _BadJsonOllama()
    from app.config import get_settings

    r.cfg = get_settings()
    # 'write a function' is a code heuristic
    route = r.route("write a function to reverse a string")
    assert route.name == "code"
