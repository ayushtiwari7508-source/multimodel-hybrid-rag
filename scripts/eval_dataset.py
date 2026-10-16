"""Evaluation dataset: Q&A pairs for retrieval + generation quality."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    expected_keywords: list[str]
    expected_source_contains: str | None = None
    route_hint: str | None = None  # "general", "code", "math", "fast", "multimodal"
    description: str = ""


# Comprehensive evaluation dataset covering all capabilities
DATASET = [
    # === FAST ROUTE (short, simple queries) ===
    QAPair(
        question="hi",
        expected_keywords=["hello", "help", "assist"],
        expected_source_contains=None,
        route_hint="fast",
        description="Simple greeting - fast route",
    ),
    QAPair(
        question="what is RAG",
        expected_keywords=["retrieval", "generation", "augmented"],
        expected_source_contains="RAG",
        route_hint="fast",
        description="Short definitional query - fast route",
    ),
    QAPair(
        question="define BM25",
        expected_keywords=["ranking", "function", "retrieval", "sparse"],
        expected_source_contains="BM25",
        route_hint="fast",
        description="Definition query - fast route",
    ),
    QAPair(
        question="what does RRF stand for",
        expected_keywords=["reciprocal", "rank", "fusion"],
        expected_source_contains="RRF",
        route_hint="fast",
        description="Acronym expansion - fast route",
    ),
    QAPair(
        question="thanks",
        expected_keywords=["welcome", "help"],
        expected_source_contains=None,
        route_hint="fast",
        description="Polite closing - fast route",
    ),

    # === CODE ROUTE (programming, debugging, SQL, regex) ===
    QAPair(
        question="Write a Python function to compute cosine similarity",
        expected_keywords=["def cosine_similarity", "numpy", "dot", "norm", "linalg"],
        expected_source_contains="code_examples",
        route_hint="code",
        description="Code generation - should route to qwen-coder",
    ),
    QAPair(
        question="How to extract LaTeX equations from text using regex in Python?",
        expected_keywords=["re.compile", "findall", "latex", "equation", "pattern"],
        expected_source_contains="code_examples",
        route_hint="code",
        description="Regex code question - code route",
    ),
    QAPair(
        question="SQL query to find documents containing RAG terms",
        expected_keywords=["SELECT", "WHERE", "LIKE", "%RAG%", "documents"],
        expected_source_contains="code_examples",
        route_hint="code",
        description="SQL query generation - code route",
    ),
    QAPair(
        question="Write a shell command to pull all required Ollama models",
        expected_keywords=["ollama pull", "nomic-embed-text", "qwen2.5", "llava"],
        expected_source_contains="code_examples",
        route_hint="code",
        description="Shell commands - code route",
    ),
    QAPair(
        question="How to configure RAG pipeline via environment variables?",
        expected_keywords=["OLLAMA_BASE_URL", "CHUNK_SIZE", "TOP_K", "RRF_C", ".env"],
        expected_source_contains="code_examples",
        route_hint="code",
        description="Configuration code - code route",
    ),
    QAPair(
        question="Python code to initialize Chroma client with nomic-embed-text",
        expected_keywords=["chromadb", "PersistentClient", "SentenceTransformerEmbeddingFunction", "nomic-embed-text"],
        expected_source_contains="vector_databases",
        route_hint="code",
        description="Vector database initialization code - code route",
    ),

    # === MATH ROUTE (equations, proofs, formulas) ===
    QAPair(
        question="Explain the equation for cosine similarity",
        expected_keywords=["cosine", "A", "B", "dot", "norm", "formula", "theta"],
        expected_source_contains="math_fundamentals",
        route_hint="math",
        description="Equation explanation - math route",
    ),
    QAPair(
        question="What is the BM25 formula and what do k1 and b parameters control?",
        expected_keywords=["BM25", "k1", "b", "term frequency", "document length", "IDF", "TF"],
        expected_source_contains="math_fundamentals",
        route_hint="math",
        description="BM25 mathematical formulation - math route",
    ),
    QAPair(
        question="Derive the RRF formula and explain the C parameter",
        expected_keywords=["RRF", "reciprocal", "rank", "fusion", "C=60", "smoothing"],
        expected_source_contains="math_fundamentals",
        route_hint="math",
        description="RRF mathematical derivation - math route",
    ),
    QAPair(
        question="Explain cross-entropy loss and KL divergence",
        expected_keywords=["cross-entropy", "KL divergence", "entropy", "p(x)", "q(x)", "log"],
        expected_source_contains="math_fundamentals",
        route_hint="math",
        description="Information theory metrics - math route",
    ),
    QAPair(
        question="What is the relationship between Euclidean distance and cosine similarity for normalized vectors?",
        expected_keywords=["euclidean", "cosine", "normalized", "sqrt", "2(1-cos)"],
        expected_source_contains="math_fundamentals",
        route_hint="math",
        description="Vector space geometry - math route",
    ),

    # === MULTIMODAL ROUTE (tables, charts, images) ===
    QAPair(
        question="What does the model comparison table show?",
        expected_keywords=["nomic-embed-text", "bge-large", "e5-large", "dimensions", "context", "license"],
        expected_source_contains="multimodal",
        route_hint="multimodal",
        description="Table reference - multimodal route",
    ),
    QAPair(
        question="Which embedding model has 3072 dimensions according to the table?",
        expected_keywords=["text-embedding-3-large", "3072", "dimensions"],
        expected_source_contains="multimodal",
        route_hint="multimodal",
        description="Specific table lookup - multimodal route",
    ),
    QAPair(
        question="What are the RRF C values and their behaviors in the table?",
        expected_keywords=["C=10", "C=60", "C=100", "strong rank", "balanced", "score-like"],
        expected_source_contains="multimodal",
        route_hint="multimodal",
        description="Table parameter reference - multimodal route",
    ),
    QAPair(
        question="Describe the cosine similarity equation shown in the multimodal document",
        expected_keywords=["cosine", "theta", "A dot B", "norm", "equation", "formula"],
        expected_source_contains="multimodal",
        route_hint="multimodal",
        description="Equation in multimodal doc - multimodal route",
    ),
    QAPair(
        question="How are images processed in the multimodal pipeline?",
        expected_keywords=["detection", "cropping", "LLaVA", "vision", "OCR", "Tesseract", "200 DPI"],
        expected_source_contains="multimodal",
        route_hint="multimodal",
        description="Image processing pipeline - multimodal route",
    ),

    # === GENERAL/RAG ROUTE (complex queries needing retrieval) ===
    QAPair(
        question="How does hybrid RAG combine BM25 and dense retrieval?",
        expected_keywords=["hybrid", "BM25", "dense", "RRF", "fusion", "reciprocal rank", "rerank"],
        expected_source_contains="rag_basics",
        route_hint="general",
        description="Core architecture question - general RAG route",
    ),
    QAPair(
        question="What are the advantages of cross-encoder reranking over bi-encoder retrieval?",
        expected_keywords=["cross-encoder", "bi-encoder", "joint", "attention", "interaction", "quality", "speed"],
        expected_source_contains="reranking",
        route_hint="general",
        description="Reranker comparison - general RAG route",
    ),
    QAPair(
        question="Compare HNSW and IVF indexing algorithms for vector search",
        expected_keywords=["HNSW", "IVF", "graph", "clusters", "centroids", "nlist", "nprobe", "M", "ef"],
        expected_source_contains="vector_databases",
        route_hint="general",
        description="Vector index comparison - general RAG route",
    ),
    QAPair(
        question="What quantization methods are available for embeddings and their tradeoffs?",
        expected_keywords=["FP16", "INT8", "PQ", "binary", "product quantization", "memory", "quality loss"],
        expected_source_contains="vector_databases",
        route_hint="general",
        description="Quantization tradeoffs - general RAG route",
    ),
    QAPair(
        question="Explain the complete multimodal ingestion pipeline for PDFs",
        expected_keywords=["pdfplumber", "camelot", "tables", "images", "LLaVA", "OCR", "Tesseract", "equations"],
        expected_source_contains="multimodal",
        route_hint="general",
        description="Multimodal PDF processing - general RAG route",
    ),
    QAPair(
        question="What are the default configuration parameters for the RAG system?",
        expected_keywords=["chunk_size=900", "chunk_overlap=150", "top_k_bm25=25", "top_k_dense=25", "top_k_rerank=6", "rrf_c=60"],
        expected_source_contains="rag_basics",
        route_hint="general",
        description="Configuration defaults - general RAG route",
    ),
    QAPair(
        question="How does the query classifier route different types of questions?",
        expected_keywords=["classifier", "fast", "rag", "complex", "multimodal", "keywords", "patterns", "confidence"],
        expected_source_contains="code_examples",
        route_hint="general",
        description="Query classification logic - general RAG route",
    ),
    QAPair(
        question="What evaluation metrics are used to measure retrieval quality?",
        expected_keywords=["Recall@k", "NDCG@k", "MAP", "MRR", "Precision@k", "F1@k"],
        expected_source_contains="math_fundamentals",
        route_hint="general",
        description="Evaluation metrics - general RAG route",
    ),
    QAPair(
        question="Compare the latency of different model routes in the system",
        expected_keywords=["fast", "800ms", "RAG", "3650ms", "complex", "15000ms", "multimodal", "5000ms"],
        expected_source_contains="benchmarks",
        route_hint="general",
        description="Latency benchmark comparison - general RAG route",
    ),
]


def load_dataset() -> list[QAPair]:
    return DATASET


def add_custom_qa(question: str, expected_keywords: list[str], **kwargs) -> QAPair:
    return QAPair(question=question, expected_keywords=expected_keywords, **kwargs)
