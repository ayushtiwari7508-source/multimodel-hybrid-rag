# Models

This directory is reserved for **model configuration documentation only** — no model weights are stored in the repository (`.gitignore` excludes local weights/artifacts).

## Model configuration

All generation/embedding models are pulled from Ollama and referenced by tag in `.env`:

| Role | Default tag (see `.env`) |
|---|---|
| Embeddings (dense retrieval) | `nomic-embed-text` |
| Router + fast/code/general routes | `qwen2.5:3b` |
| Math + multimodal routes | `qwen2.5:7b` |
| Image descriptions (vision) | `llava:7b` |
| Rerank backend (optional, `RERANK_BACKEND=ollama`) | `bge-reranker-v2-m3` |

The cross-encoder reranker (`RERANK_BACKEND=cross-encoder`, default) downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face and caches it locally outside this repo.

## Pulling models

```bash
python scripts/pull_models.py     # pulls the models referenced in .env
```
