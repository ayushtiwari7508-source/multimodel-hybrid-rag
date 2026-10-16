"""Tests for the query classifier / pipeline router."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.retrieval.classifier import PipelineRoute, QueryClassifier


@pytest.fixture(autouse=True)
def _reset_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_classifier_simple_factual_queries():
    """Simple factual queries should route to FAST path."""
    clf = QueryClassifier()

    queries = [
        "What is the capital of France?",
        "Define RAG",
        "Who is Einstein?",
        "When did WW2 end?",
        "Where is Tokyo?",
        "What is 2+2?",
    ]

    for q in queries:
        result = clf.classify(q)
        assert result.route == PipelineRoute.FAST, f"Query '{q}' should be FAST, got {result.route}"
        assert result.confidence > 0.5
        assert result.latency_ms >= 0


def test_classifier_document_queries():
    """Document-specific queries should route to RAG path."""
    clf = QueryClassifier()

    queries = [
        "According to the document, what is X?",
        "In the uploaded PDF, find the section about Y",
        "Summarize the document about Z",
        "Find information about ABC in the files",
        "What does the document say about pricing?",
        "From the text, extract the key points",
    ]

    for q in queries:
        result = clf.classify(q)
        assert result.route == PipelineRoute.RAG, f"Query '{q}' should be RAG, got {result.route}"
        assert result.confidence > 0.5


def test_classifier_multimodal_queries():
    """Multimodal queries should route to MULTIMODAL path."""
    clf = QueryClassifier()

    queries = [
        "What does the chart on page 3 show?",
        "Describe the figure in the diagram",
        "Read the data from the table",
        "Explain the image in the report",
        "What does the graph illustrate?",
        "Analyze the visual chart",
    ]

    for q in queries:
        result = clf.classify(q)
        assert result.route == PipelineRoute.MULTIMODAL, f"Query '{q}' should be MULTIMODAL, got {result.route}"
        assert result.confidence > 0.5


def test_classifier_complex_queries():
    """Complex reasoning queries should route to COMPLEX path."""
    clf = QueryClassifier()

    queries = [
        "Compare the approaches in document A and B",
        "Contrast the two methodologies",
        "Analyze the relationship between X and Y",
        "Explain why the results differ",
        "How does the proposed method improve over baseline?",
        "What are the implications of these findings?",
    ]

    for q in queries:
        result = clf.classify(q)
        assert result.route == PipelineRoute.COMPLEX, f"Query '{q}' should be COMPLEX, got {result.route}"
        assert result.confidence > 0.5


def test_classifier_default_fallback():
    """Ambiguous queries should default to RAG (safe fallback)."""
    clf = QueryClassifier()

    queries = [
        "Tell me about machine learning in detail with examples",
        "Explain how neural networks work and their applications",
        "What are the benefits and drawbacks of this approach?",
        "Give me a comprehensive overview of the topic with citations",
    ]

    for q in queries:
        result = clf.classify(q)
        # Should default to RAG for uncertain cases
        assert result.route == PipelineRoute.RAG, f"Query '{q}' should default to RAG, got {result.route}"


def test_classifier_disabled():
    """When classifier is disabled, should default to RAG."""
    from app.config import get_settings

    get_settings.cache_clear()

    import os

    os.environ["ENABLE_QUERY_CLASSIFIER"] = "false"
    get_settings.cache_clear()

    clf = QueryClassifier()
    result = clf.classify("What is the capital of France?")

    assert result.route == PipelineRoute.RAG
    assert result.reason == "classifier disabled"

    del os.environ["ENABLE_QUERY_CLASSIFIER"]
    get_settings.cache_clear()


def test_classifier_short_queries():
    """Very short queries should route to FAST."""
    clf = QueryClassifier()

    queries = [
        "Hi",
        "Hello",
        "Thanks",
        "OK",
        "What?",
    ]

    for q in queries:
        result = clf.classify(q)
        assert result.route == PipelineRoute.FAST, f"Query '{q}' should be FAST, got {result.route}"


def test_classifier_latency_tracking():
    """Classifier should track latency."""
    clf = QueryClassifier()
    result = clf.classify("What is RAG?")

    assert result.latency_ms >= 0
    assert isinstance(result.latency_ms, float)
