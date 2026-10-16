# Frontend

Streamlit-based chat UI for the Hybrid Multi-Model RAG system.

## Running

```bash
# From project root
python scripts/serve_ui.py
```

Or directly:
```bash
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

## Configuration

Set environment variables:
- `RAG_API_URL` - Backend API URL (default: http://localhost:8000)
- `RAG_API_KEY` - API key for authentication (if enabled)
