# Hybrid Multi-Model RAG System
## Technology Stack Overview

---

## 1. Core Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Ingest     │────▶│  Retrieval   │────▶│   Rerank    │────▶│  Generate   │
│  (Load/     │     │  (BM25 +     │     │  (Cross-    │     │  (LLM       │
│   Chunk/    │     │   Dense)     │     │   Encoder)  │     │   Router)   │
│   Embed)    │     │              │     │             │     │             │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
```

---

## 2. Technology Categories

### 📄 Document Processing & Ingestion

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Parsing** | `pdfplumber`, `pypdf` | Extract text, tables, metadata from PDFs |
| **OCR** | `Tesseract` (via `pytesseract`) | Extract text from scanned/image PDFs |
| **Table Extraction** | `camelot-py`, `img2table` | Parse tables from PDFs to CSV |
| **Office Docs** | `python-docx`, `python-pptx`, `openpyxl` | DOCX, PPTX, XLSX support |
| **Images** | `Pillow` + `LLaVA` (via Ollama) | Vision model describes charts/diagrams |
| **Markdown/Text** | Native Python | Direct parsing |
| **Chunking** | Custom sliding window | 900 tokens, 150 overlap |

### 🔍 Retrieval (Hybrid Search)

| Component | Technology | Configuration |
|-----------|------------|---------------|
| **Sparse (BM25)** | `rank-bm25` (BM25Okapi) | Top-10, persisted as pickle |
| **Dense (Vector)** | `ChromaDB` + `nomic-embed-text` | Top-10, cosine similarity |
| **Fusion** | Reciprocal Rank Fusion (RRF) | c=60 |
| **Reranker** | `sentence-transformers` CrossEncoder | `BAAI/bge-reranker-base` (default) |
| **Fallback** | Weighted blend (0.6 RRF + 0.4 norm) | When reranker unavailable |

### 🤖 LLM & Routing

| Component | Technology | Models Used |
|-----------|------------|-------------|
| **LLM Runtime** | `Ollama` (local) | All models run locally |
| **Router** | LLM-based classifier | `qwen2.5:3b` → routes to specialized models |
| **Code** | Code-specialized | `qwen2.5-coder:7b` |
| **General** | General purpose | `qwen2.5:7b` |
| **Math** | Math/Reasoning | `qwen2.5:7b` |
| **Fast** | Quick/Chat | `qwen2.5:3b` |
| **Multimodal** | Vision-capable | `qwen2.5:7b` (or `llava:7b`) |
| **Embeddings** | Text embeddings | `nomic-embed-text` |

### 🌐 API & Web Layer

| Component | Technology | Features |
|-----------|------------|----------|
| **REST API** | `FastAPI` + `Uvicorn` | Async, OpenAPI docs, CORS |
| **Streaming** | SSE (Server-Sent Events) | Token-by-token responses |
| **Auth** | API Key header (`X-API-Key`) | Optional, per-request |
| **Rate Limiting** | Token bucket (in-memory) | 60 req/min per IP |
| **Metrics** | Custom counters/timers | Prometheus-compatible `/metrics` |
| **UI** | `Streamlit` | Chat, upload, citations, doc management |

### 📦 Infrastructure & Ops

| Component | Technology |
|-----------|------------|
| **Containerization** | `Docker` + `docker-compose` |
| **CI/CD** | GitHub Actions (ruff + pytest) |
| **Linting** | `ruff` (fast Python linter) |
| **Testing** | `pytest` (71 tests: unit + integration) |
| **Backup/Restore** | Custom Python script (tar.gz snapshots) |
| **Configuration** | `.env` + `pydantic-settings` |
| **Logging** | Structured JSON + request IDs |

---

## 3. Data Flow Summary

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Router (qwen2.5:3b)                │
│  Classifies: code / math / general  │
│  / multimodal / fast                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Hybrid Retrieval                   │
│  1. Dense search (Chroma + nomic)   │
│  2. Sparse search (BM25)            │
│  3. RRF fusion                      │
│  4. Cross-encoder rerank            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Generator (routed model)           │
│  - Inline [n] citations             │
│  - Sources section                  │
│  - Streaming support                │
└─────────────────────────────────────┘
```

---

## 4. Key Differentiators

| Feature | Implementation |
|---------|----------------|
| **Local-First** | Zero external API calls; all models on your machine |
| **Multi-Model** | 6 specialized Ollama models, smart routing |
| **Hybrid Retrieval** | BM25 + Dense + Cross-encoder (not just one) |
| **Citations** | Every answer includes source chunks with page/section |
| **Streaming** | Token-by-token via SSE, works with curl |
| **Production Ops** | Auth, rate limit, metrics, health, backup, Docker |
| **Multimodal** | Tables, images, equations, OCR all extracted |

---

## 5. Project Structure

```
hybrid-rag/
├── app/                    # Core library
│   ├── api.py              # FastAPI endpoints
│   ├── config.py           # Pydantic settings
│   ├── generator.py        # Answer generation + streaming
│   ├── ingest.py           # Document ingestion pipeline
│   ├── loaders.py          # PDF/DOCX/PPTX/XLSX/MD loaders
│   ├── ollama_client.py    # Ollama HTTP client
│   ├── pipeline.py         # RAGPipeline orchestrator
│   ├── retriever.py        # Hybrid retrieval + rerank
│   ├── reranker.py         # CrossEncoder + Ollama backends
│   ├── router.py           # LLM-based query classifier
│   ├── storage.py          # Chroma + BM25 stores
│   └── observability.py    # Logging, metrics, request IDs
├── ui/
│   └── streamlit_app.py    # Chat UI with citations
├── scripts/
│   ├── serve_api.py        # API server
│   ├── serve_ui.py         # UI server
│   ├── pull_models.py      # Download Ollama models
│   ├── tune.py             # Hyperparameter tuning
│   ├── eval.py             # Evaluation harness
│   └── backup.py           # Backup/restore tool
├── tests/                  # 71 tests (unit + integration)
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 6. Requirements Summary

**System Dependencies:**
- Ollama (llama.cpp backend)
- Tesseract OCR
- Poppler (pdfplumber)
- Ghostscript (camelot)

**Python Packages (key):**
```
fastapi, uvicorn, streamlit
chromadb, rank-bm25, sentence-transformers
pdfplumber, pypdf, camelot-py, img2table
pytesseract, pillow
python-docx, python-pptx, openpyxl
pydantic, pydantic-settings, requests, httpx
pytest, ruff
```

**Models (auto-pulled):**
- `nomic-embed-text` — embeddings
- `qwen2.5:3b` — router + fast
- `qwen2.5:7b` — general + math + multimodal
- `qwen2.5-coder:7b` — code
- `llava:7b` — vision
- `BAAI/bge-reranker-base` — cross-encoder (HF, not Ollama)

---

## 7. Quick Start

```bash
# 1. Clone & setup
cd hybrid-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start Ollama
ollama serve

# 3. Pull models
python scripts/pull_models.py

# 4. Run API + UI
python scripts/serve_api.py    # Terminal 1
python scripts/serve_ui.py     # Terminal 2

# 5. Open http://localhost:8501
```

---

## 8. Evaluation Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Keyword Recall** | **0.917** | Cross-encoder reranker |
| **Route Accuracy** | 0.60 | 5 test queries |
| **Avg Latency** | ~11.5s | First run (model load) |
| **Tests Passing** | 71/71 | Unit + Integration |

*Previous weighted-blend fallback: 0.800 keyword recall*

---

## 9. License & Credits

- **Code**: MIT-style (project-local)
- **Models**: Respect each model's license (Qwen, LLaVA, Nomic, BGE)
- **Libraries**: All open-source (Apache-2.0, MIT, BSD)

---

*Generated for college project presentation — Hybrid Multi-Model RAG System*