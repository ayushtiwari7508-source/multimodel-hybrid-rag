"""Sparse keyword search using BM25."""
from __future__ import annotations

import pickle
import re
import threading
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.app.core.config import get_settings
from backend.app.core.observability import get_logger, stable_hash

log = get_logger(__name__)

_TOKENIZE_RE = re.compile(r"\w+", re.UNICODE)


def coerce_metadata(meta: dict) -> dict:
    out: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ",".join(str(x) for x in v if x is not None)
        else:
            out[k] = str(v)
    return out


def chunk_to_records(chunks: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ch in chunks:
        content = ch["content"]
        meta = ch.get("meta", {})
        cid = stable_hash(
            {
                "c": content[:200],
                "s": meta.get("source"),
                "p": meta.get("page"),
                "sec": meta.get("section"),
                "did": meta.get("doc_id"),
            }
        )
        out.append({"id": cid, "content": content, "meta": coerce_metadata(meta)})
    return out


class BM25Store:
    def __init__(self, persist_path: str | None = None) -> None:
        self.persist_path = Path(persist_path or get_settings().bm25_persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.corpus: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self.persist_path.exists():
                try:
                    with self.persist_path.open("rb") as f:
                        payload = pickle.load(f)  # noqa: S301 -- local BM25 index, not untrusted input
                    self.corpus = payload.get("corpus", [])
                    if self.corpus:
                        self.bm25 = BM25Okapi([self._tokenize(c["content"]) for c in self.corpus])
                except Exception as e:
                    log.warning(f"BM25 load failed: {e}")

    def _save(self) -> None:
        with self._lock, self.persist_path.open("wb") as f:
            pickle.dump({"corpus": self.corpus}, f)

    @staticmethod
    @lru_cache(maxsize=2048)
    def _tokenize(text: str) -> tuple[str, ...]:
        return tuple(t.lower() for t in _TOKENIZE_RE.findall(text) if t)

    def _tokenize_list(self, texts: list[str]) -> list[tuple[str, ...]]:
        return [self._tokenize(t) for t in texts]

    def add(self, docs: list[dict]) -> None:
        with self._lock:
            existing_ids = {c["id"] for c in self.corpus}
            new = [d for d in docs if d["id"] not in existing_ids]
            if not new:
                return
            self.corpus.extend(new)
            self.bm25 = BM25Okapi(self._tokenize_list([c["content"] for c in self.corpus]))
            self._save()

    def search(self, query: str, n: int) -> list[tuple[dict, float]]:
        with self._lock:
            if not self.bm25 or not self.corpus:
                return []
            tokens = self._tokenize(query)
            scores = self.bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
            return [(self.corpus[i], float(s)) for i, s in ranked]

    def clear(self) -> None:
        with self._lock:
            self.corpus = []
            self.bm25 = None
            if self.persist_path.exists():
                self.persist_path.unlink()
            self._tokenize.cache_clear()

    def delete_where(self, where: dict) -> int:
        with self._lock:
            before = len(self.corpus)
            self.corpus = [c for c in self.corpus if not all(c.get("meta", {}).get(k) == v for k, v in where.items())]
            removed = before - len(self.corpus)
            if removed:
                self.bm25 = BM25Okapi(self._tokenize_list([c["content"] for c in self.corpus])) if self.corpus else None
                self._save()
            return removed

    def list_distinct(self, field: str) -> list[str]:
        with self._lock:
            seen: set[str] = set()
            for c in self.corpus:
                v = c.get("meta", {}).get(field)
                if v is not None:
                    seen.add(str(v))
            return sorted(seen)
