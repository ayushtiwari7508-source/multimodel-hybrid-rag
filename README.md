# Hybrid Multi-Model RAG

A local-first, hybrid retrieval-augmented generation system that combines sparse (BM25) and dense (Ollama embeddings) retrieval, fuses them with reciprocal rank fusion (RRF), reranks with a cross-encoder, and routes the answer generation across multiple Ollama models.

- **Loaders**: PDF, Markdown, TXT, DOCX, PPTX, XLSX
- **Multimodal extraction**: text, tables (pdfplumber + camelot + img2table), images (LLaVA descriptions via Ollama), OCR (Tesseract), equations (heuristic LaTeX detection)
- **Storage**: Chroma (dense) + persistent BM25 (sparse)
- **Retrieval**: dense (nomic-embed-text) + sparse (BM25) → RRF → cross-encoder rerank
- **Router**: LLM-based classifier (qwen2.5:3b) → `code | math | multimodal | fast | general`
- **Generation**: each route maps to a specialized Ollama model; answers carry inline `[n]` citations and a Sources block
- **APIs**: FastAPI REST (incl. SSE streaming + `/metrics`) + Streamlit chat UI with citations and source viewer
- **Ops**: API-key auth, per-IP rate limiting, structured logs with request IDs, Docker, CI, backup/restore

## Architecture

```
                       ┌──────────────┐
    PDF/MD/DOCX/... ──▶│  Ingestion   │── text + tables + image-descs + OCR + equations
                       └──────┬───────┘
                              ▼
               ┌──────────────┴──────────────┐
               ▼                             ▼
        ┌─────────────┐               ┌─────────────┐
        │  Chroma DB  │               │  BM25 Index │
        │ (nomic-emb) │               │  (rank-bm25)│
        └──────┬──────┘               └──────┬──────┘
               │   query embedding           │  tokenized query
               ▼                             ▼
               └─────────► RRF fusion ◄──────┘
                              ▼
                   cross-encoder reranker
                              ▼
                   top-N context blocks
                              ▼
              ┌──── LLM router (qwen2.5:3b) ───┐
              ▼                                 ▼
    code (qwen2.5:3b)        general (qwen2.5:3b)  / fast (qwen2.5:3b)
                                             math / multimodal (qwen2.5:7b)
                              ▼
                   Cited answer + Sources
```

## Project Structure

```
hybrid-rag/
├── frontend/
│   ├── app.py              # Streamlit chat UI
│   └── README.md
│
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI entry point
│   │   │
│   │   ├── api/
│   │   │   ├── routes/     # API endpoints
│   │   │   │   ├── query.py
│   │   │   │   ├── documents.py
│   │   │   │   └── health.py
│   │   │   └── dependencies.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── observability.py
│   │   │
│   │   ├── ingestion/
│   │   │   ├── loaders/    # PDF, DOCX, MD, etc.
│   │   │   └── pipeline.py
│   │   │
│   │   ├── retrieval/
│   │   │   ├── embeddings.py
│   │   │   ├── keyword_search.py
│   │   │   ├── vector_search.py
│   │   │   ├── hybrid_search.py
│   │   │   ├── rrf.py
│   │   │   └── reranker.py
│   │   │
│   │   ├── routing/
│   │   │   ├── router.py   # LLM-based router
│   │   │   └── classifier.py  # Query classifier
│   │   │
│   │   ├── generation/
│   │   │   ├── llm.py      # Ollama client
│   │   │   └── prompts.py  # Prompt building
│   │   │
│   │   ├── services/
│   │   │   ├── document_service.py
│   │   │   └── query_service.py
│   │   │
│   │   ├── models/
│   │   └── schemas/
│   │
│   ├── requirements.txt
│   └── README.md
│
├── database/
│   └── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── eval_kb/            # Evaluation knowledge base
│   └── README.md
│
├── models/
│   └── README.md
│
├── scripts/
│   ├── eval.py
│   ├── tune.py
│   ├── serve_api.py
│   ├── serve_ui.py
│   ├── smoke_test.py
│   └── backup.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── conftest.py
│
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   ├── api.md
│   └── evaluation.md
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── docker-compose.yml
```

## Setup

```bash
cd hybrid-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit if needed

# 1) start Ollama (https://ollama.com)
ollama serve

# 2) pull models
python scripts/pull_models.py

# 3) run the API
python scripts/serve_api.py     # http://localhost:8000

# 4) in another terminal, run the UI
python scripts/serve_ui.py      # http://localhost:8501
```

## Usage

### Streamlit UI
Open http://localhost:8501, upload documents, then chat. Each answer shows route, model, and clickable source excerpts with page/section.
If the API has auth enabled, set `RAG_API_KEY` in `.env` (or the environment) before starting the UI.

### REST API
```bash
# health
curl http://localhost:8000/health

# ingest text
curl -X POST http://localhost:8000/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"Your content here", "source":"docs"}'

# upload files
curl -X POST http://localhost:8000/ingest/upload \
  -F 'files=@paper.pdf' -F 'files=@notes.docx'

# query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Summarize the methods section."}'
```

When `API_KEY` is set, protected endpoints require the header:
```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"question":"hello"}'
```
`/health` and `/metrics` stay public for monitoring. Every response includes an `x-request-id` header.

### Streaming answers (SSE)
```bash
curl -N http://localhost:8000/query/stream \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Summarize the intro."}'
```
Emits `meta`, then `token` events, then `citations` and `done`.

### Metrics
```bash
curl http://localhost:8000/metrics
```
Returns request counters/timings, auth failures, rate-limit blocks, and pipeline stats for Prometheus scraping.

### Smoke test
```bash
python scripts/smoke_test.py
```

## Rerank backends

The retriever supports two rerank backends via `RERANK_BACKEND`:

| Backend          | Model (default)             | Notes                                  |
|------------------|-----------------------------|----------------------------------------|
| `cross-encoder`  | `cross-encoder/ms-marco-MiniLM-L-6-v2`    | **Default**. Local HF CrossEncoder; highest quality, no Ollama dependency. |
| `ollama`         | `bge-reranker-v2-m3`        | Falls back to weighted RRF blend if model missing or endpoint fails. |

## How the router picks a model

`backend/app/routing/router.py` runs `qwen2.5:3b` with a strict JSON-only prompt and falls back to a heuristic if it fails. Routes map to:

| Route       | Model              | When                                 |
|-------------|--------------------|--------------------------------------|
| `code`      | `qwen2.5:3b` | code, debugging, regex, SQL, shell   |
| `math`      | `qwen2.5:7b`       | equations, proofs, scientific math   |
| `multimodal`| `qwen2.5:7b`       | references to figures/tables/images  |
| `fast`      | `qwen2.5:3b`       | very short or chit-chat questions    |
| `general`   | `qwen2.5:3b`       | default                              |

Edit `.env` to point these at any Ollama tag you have.

## Configuration

See `.env.example` for all knobs. The most important ones:

- `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunking window
- `TOP_K_BM25`, `TOP_K_DENSE`, `TOP_K_RERANK` — retrieval fan-out
- `RRF_C` — RRF smoothing constant
- `MAX_CONTEXT_TOKENS` — context budget for the prompt
- `EMBED_MODEL`, `RERANK_MODEL`, `VISION_MODEL` — model tags (HF for cross-encoder, Ollama for others)
- `RERANK_BACKEND` — `cross-encoder` (default) or `ollama`
- `API_KEY` — if non-empty, all endpoints except `/health` and `/metrics` require `X-API-Key`
- `RATE_LIMIT_PER_MIN` — per-IP request cap (default 60)
- `RAG_API_KEY` — sent as `X-API-Key` by the Streamlit UI (leave empty if auth is disabled)

## Docker

```bash
docker compose up --build      # API on :8000, UI on :8501
```
The `api` service reaches Ollama on the host via `host.docker.internal:11434`, and persists index data in the `rag-data` volume. Set `API_KEY` / `RAG_API_KEY` in the environment to enable auth.

## Backup & restore

```bash
python scripts/backup.py backup [OUTPUT_DIR]      # tar.gz snapshot (chroma, bm25, meta, uploads)
python scripts/backup.py restore <archive.tar.gz> [--force]
```
`--force` overwrites existing data directories; without it, existing dirs are skipped.

## CI

`.github/workflows/ci.yml` runs ruff lint + format check and the test suite on every push/PR. Integration tests that need a live Ollama are excluded in CI; they auto-skip locally when Ollama is down.

## Storage layout

```
data/
├── raw/              # Optional source documents (ignored except .gitkeep)
├── processed/        # Derived documents (ignored except .gitkeep)
├── eval_kb/          # Evaluation knowledge base
├── chroma/           # Chroma persistent dir (runtime, ignored)
├── bm25/index.pkl    # Pickled BM25 corpus (runtime, ignored)
├── index_meta/       # Reserved per-document metadata (runtime, ignored)
├── uploads/          # API uploads (runtime, ignored)
├── evaluation/       # Generated evaluation and tuning output (ignored)
└── logs/             # Runtime logs (ignored)
```

## Tests

```bash
# Run all tests
python -m pytest

# Run only unit tests
python -m pytest tests/unit -v

# Run integration tests (requires Ollama)
python -m pytest tests/integration -v
```

## Evaluation

```bash
# Run evaluation
python scripts/eval.py

# Run parameter tuning
python scripts/tune.py
```

## Notes

- The rerank step uses `Ollama /api/rerank` if your Ollama build exposes it; otherwise it gracefully falls back to a weighted RRF + BM25 + dense score blend.
- For PDFs without a text layer, OCR (Tesseract) is applied to image crops. Install Tesseract via your system package manager (e.g. `brew install tesseract`).
- Vision descriptions use `llava:7b` by default; swap to any vision-capable Ollama tag in `.env`.
- Clearing the index (`POST /clear` or the sidebar button) wipes both Chroma and the BM25 pickle.
