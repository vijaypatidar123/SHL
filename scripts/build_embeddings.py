"""Precompute corpus embeddings so the deployed service doesn't have to.

Run this offline whenever data/catalog.json changes:

    python -m scripts.build_embeddings

Output: data/embeddings/corpus.npy (float32, shape [N, 384] for MiniLM-L6).
"""
from __future__ import annotations

import time

from app.retrieval import Retriever, CORPUS_EMB


def main() -> int:
    if CORPUS_EMB.exists():
        CORPUS_EMB.unlink()
    r = Retriever()
    t0 = time.time()
    ok = r._ensure_dense()
    if not ok:
        print("FAILED: dense retriever could not be initialized")
        return 1
    elapsed = time.time() - t0
    arr = r._dense
    assert arr is not None
    print(f"Built embeddings: shape={arr.shape} dtype={arr.dtype} in {elapsed:.1f}s")
    print(f"Saved to {CORPUS_EMB} ({CORPUS_EMB.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
