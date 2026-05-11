"""Hybrid retrieval over the SHL catalog.

BM25 over tokenized name + type labels + description + job levels carries
exact-token matches ("OPQ32r", "Java", "HIPAA"). A sentence-transformer dense
index carries semantic intent ("leadership", "stakeholder management").

The two scores are min-max normalized and averaged with weight `alpha`
(default 0.5). Structured filters (test_types, languages, max_duration_minutes)
are applied as a hard mask after scoring — they exclude items, not down-rank
them.

Dense retrieval is optional: if the corpus embedding cache is missing AND
sentence-transformers can't load (e.g. tight memory on the deploy host),
search degrades to BM25-only without raising.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# Pin transformers to the PyTorch backend so a stray TensorFlow/Keras-3 install
# in the environment doesn't break model loading. Must be set *before* any
# transformers-backed module gets imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
from rank_bm25 import BM25Okapi

from app.catalog import ALL_ITEMS

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
CORPUS_EMB = EMB_DIR / "corpus.npy"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOK_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK_RE.findall(text.lower())


def _build_doc(item: dict[str, Any]) -> str:
    """Compose the searchable text for one assessment.

    Name is repeated so token-level signal weighs heavier in BM25 — exact
    product mentions ("OPQ32r", "Verify G+") should outrank prose hits.
    """
    name = item.get("name", "")
    type_codes = " ".join(item.get("test_types") or [])
    type_labels = ", ".join(item.get("test_type_labels") or [])
    description = (item.get("description") or "")[:800]
    job_levels = ", ".join(item.get("job_levels") or [])
    return (
        f"{name}. {name}. "
        f"Test type codes: {type_codes}. "
        f"Categories: {type_labels}. "
        f"{description} "
        f"Suitable for: {job_levels}."
    )


class Retriever:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items = items if items is not None else ALL_ITEMS
        self.docs: list[str] = [_build_doc(it) for it in self.items]
        tokenized = [tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)
        self._dense: np.ndarray | None = None
        self._encoder = None  # lazy

    # ----- dense -----------------------------------------------------------
    def _encode_query(self, query: str) -> "np.ndarray":
        """Encode a single query string, cached by text to avoid re-encoding."""
        if not hasattr(self, "_query_cache"):
            self._query_cache: dict[str, np.ndarray] = {}
        if query not in self._query_cache:
            self._query_cache[query] = self._encoder.encode(  # type: ignore[union-attr]
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0].astype(np.float32)
        return self._query_cache[query]

    def _ensure_dense(self) -> bool:
        """Load (or build) the dense corpus index. Returns True on success."""
        if self._dense is not None:
            return True
        # Load cached corpus embeddings
        if CORPUS_EMB.exists():
            try:
                arr = np.load(CORPUS_EMB)
                if arr.shape[0] == len(self.items):
                    self._dense = arr.astype(np.float32, copy=False)
                else:
                    log.warning(
                        "corpus embedding shape %s != %d items; ignoring",
                        arr.shape,
                        len(self.items),
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to load %s: %s", CORPUS_EMB, exc)
        # Build encoder lazily
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(EMBED_MODEL)
            except Exception as exc:  # noqa: BLE001
                log.warning("dense disabled — cannot load %s: %s", EMBED_MODEL, exc)
                self._encoder = False  # sentinel: don't retry
        if self._dense is None and self._encoder:
            log.info("encoding %d docs (no cache found)", len(self.docs))
            self._dense = self._encoder.encode(
                self.docs, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            EMB_DIR.mkdir(parents=True, exist_ok=True)
            np.save(CORPUS_EMB, self._dense)
        return self._dense is not None and bool(self._encoder)

    # ----- search ----------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        test_types: list[str] | None = None,
        max_duration_minutes: int | None = None,
        languages: list[str] | None = None,
        exclude_slugs: set[str] | None = None,
        top_k: int = 20,
        alpha: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]:
        if not query or not query.strip():
            return []

        n = len(self.items)
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        bm25_norm = _minmax(bm25_scores)

        if self._ensure_dense():
            q_emb = self._encode_query(query)
            dense_scores = self._dense @ q_emb  # type: ignore[operator]
            dense_norm = _minmax(dense_scores)
            fused = alpha * bm25_norm + (1.0 - alpha) * dense_norm
        else:
            fused = bm25_norm

        # Apply structured filters as a mask
        mask = np.ones(n, dtype=bool)
        if test_types:
            wanted = {c.upper() for c in test_types}
            for i, it in enumerate(self.items):
                codes = {c.upper() for c in (it.get("test_types") or [])}
                if not (codes & wanted):
                    mask[i] = False
        if max_duration_minutes is not None:
            for i, it in enumerate(self.items):
                d = it.get("duration_minutes")
                # Keep untimed items unless explicitly excluded; keep items with no duration data.
                if d is not None and not it.get("untimed") and d > max_duration_minutes:
                    mask[i] = False
        if languages:
            wanted_langs = [l.lower() for l in languages]
            for i, it in enumerate(self.items):
                langs = " ".join(it.get("languages") or []).lower()
                if not any(w in langs for w in wanted_langs):
                    mask[i] = False
        if exclude_slugs:
            for i, it in enumerate(self.items):
                if it.get("slug") in exclude_slugs:
                    mask[i] = False

        fused = np.where(mask, fused, -np.inf)
        order = np.argsort(-fused)[:top_k]
        results = []
        for idx in order:
            score = float(fused[idx])
            if not np.isfinite(score):
                break
            results.append((self.items[int(idx)], score))
        return results


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
