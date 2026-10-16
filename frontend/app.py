import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid Multi-Model RAG", page_icon="R", layout="wide")
st.title("Hybrid Multi-Model RAG")
st.caption("BM25 + dense embeddings + cross-encoder reranking, routed across Ollama models with citations.")


def _headers() -> dict:
    headers = {}
    api_key = os.environ.get("RAG_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def api_get(path: str, timeout: int = 30, retries: int = 3) -> dict:
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(f"{API_URL}{path}", headers=_headers(), timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise last_err


def api_post(path: str, json: dict | None = None, files: list | None = None) -> dict:
    if files:
        r = requests.post(f"{API_URL}{path}", files=files, headers=_headers(), timeout=600)
    else:
        r = requests.post(f"{API_URL}{path}", json=json or {}, headers=_headers(), timeout=600)
    r.raise_for_status()
    return r.json()


def api_post_upload_one(file_bytes: bytes, filename: str) -> dict:
    """Upload a single file and return result."""
    files = [("files", (filename, file_bytes))]
    return api_post("/ingest/upload", files=files)


def api_delete(path: str) -> dict:
    r = requests.delete(f"{API_URL}{path}", headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


with st.sidebar:
    st.header("Index")
    try:
        health = api_get("/health")
    except Exception as e:
        st.error(f"API not reachable at {API_URL}: {e}")
        st.stop()

    if health.get("ok"):
        st.success(f"API OK | chunks: {health['stats'].get('chroma', 0)}")
    else:
        st.error("API reachable but unhealthy")
    for note in health.get("notes", []):
        st.warning(note)

    if st.button("Clear index"):
        api_post("/clear")
        st.rerun()

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = api_get("/documents").get("documents", [])
    except Exception as e:
        docs = []
        st.error(f"Failed to list: {e}")
    if docs:
        for d in docs:
            label = d.get("source") or d.get("doc_id") or "?"
            doc_id = d.get("doc_id") or label
            cols = st.columns([4, 1])
            cols[0].markdown(f"`{label}`")
            if cols[1].button("Delete", key=f"del-{doc_id}"):
                try:
                    api_delete(f"/documents/{doc_id}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.subheader("Upload documents")
    uploads = st.file_uploader(
        "PDF, MD, TXT, DOCX, PPTX, XLSX",
        type=["pdf", "md", "markdown", "txt", "docx", "pptx", "xlsx"],
        accept_multiple_files=True,
    )
    if st.button("Ingest uploads") and uploads:
        total = len(uploads)
        overall_bar = st.progress(0.0, text=f"Processing {total} file(s)...")
        status_area = st.empty()
        total_chunks = 0
        for i, u in enumerate(uploads):
            status_area.info(f"📄 **{u.name}** ({i + 1}/{total}) — extracting text, tables, images, OCR...")
            file_bar = st.progress(0.0, text=f"Uploading {u.name}...")
            try:
                for p in range(1, 11):
                    file_bar.progress(p / 10, text=f"Uploading {u.name}... {p * 10}%")
                    time.sleep(0.02)
                res = api_post_upload_one(u.getvalue(), u.name)
                chunks = res.get("chunks", 0)
                total_chunks += chunks
                status_area.success(f"✅ **{u.name}** — {chunks} chunks indexed")
            except Exception as e:
                status_area.error(f"❌ **{u.name}** failed: {e}")
            overall_bar.progress((i + 1) / total, text=f"Processed {i + 1}/{total} files")
        overall_bar.progress(1.0, text=f"Done — {total_chunks} total chunks indexed in {total} file(s)")
        status_area.empty()
        st.rerun()

    st.divider()
    st.subheader("Add raw text")
    txt = st.text_area("Paste text", height=120)
    src = st.text_input("Source label", value="user-text")
    if st.button("Ingest text") and txt.strip():
        api_post("/ingest/text", {"text": txt, "source": src})
        st.success("Indexed")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_hits(hits: list[dict]) -> None:
    if not hits:
        st.info("No retrieval hits.")
        return
    for h in hits:
        meta = h.get("meta", {})
        with st.expander(
            f"[{h.get('id', '')[:8]}] {Path(meta.get('source', '?')).name} • p.{meta.get('page', '-')} • score {h.get('final_score', 0):.3f}"
        ):
            st.markdown(f"**Type:** `{meta.get('element_type', 'text')}` • **Section:** `{meta.get('section', '-')}`")
            st.code(h.get("content", "")[:1500], language="markdown")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            with st.expander(f"Sources ({len(m['hits'])})"):
                render_hits(m["hits"])
        if m.get("route"):
            r = m["route"]
            st.caption(f"Routed to `{r.get('model')}` via `{r.get('name')}` — {r.get('reason', '')}")

if q := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            try:
                res = api_post("/query", {"question": q})
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()
        st.markdown(res.get("answer", ""))
        hits = res.get("hits", [])
        route = res.get("route", {})
        with st.expander(f"Sources ({len(hits)})"):
            render_hits(hits)
        st.caption(f"Routed to `{route.get('model')}` via `{route.get('name')}` — {route.get('reason', '')}")
    st.session_state.messages.append(
        {"role": "assistant", "content": res.get("answer", ""), "hits": hits, "route": route}
    )
