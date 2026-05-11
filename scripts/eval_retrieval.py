"""Quick retrieval sanity probe across trace personas.

Each (label, query, expected_keywords) line is a "did the right items show up
in top-10?" check — not a Recall@10 metric (that needs the formal trace
expected-list). Useful to spot embarrassing failures fast.

Run: python -m scripts.eval_retrieval
"""
from __future__ import annotations

from app.retrieval import get_retriever


PROBES = [
    # (label, query, must-include-substrings-in-top10)
    (
        "C1 — senior leadership",
        "senior leadership CXO director-level 15+ years selection benchmark",
        ["OPQ32r", "Leadership"],
    ),
    (
        "C2 — Rust senior IC infra",
        "senior Rust engineer high-performance networking infrastructure systems",
        ["Linux Programming", "Networking", "Smart Interview Live Coding"],
    ),
    (
        "C9 — backend Java Spring SQL",
        "senior backend Java Spring SQL AWS Docker microservice",
        ["Core Java", "Spring", "SQL", "AWS", "Docker"],
    ),
    (
        "C3 — contact center high volume",
        "contact center call center agents high volume English language screening",
        ["Contact Center", "SVAR", "Customer Service", "Phone"],
    ),
    (
        "Vague query",
        "I need an assessment",
        [],  # nothing specific — just shouldn't crash
    ),
    (
        "Personality test only",
        "personality assessment for managers",
        ["OPQ32r"],
    ),
    (
        "Numerical reasoning",
        "numerical reasoning ability test",
        ["Numerical", "Verify"],
    ),
]


def main() -> int:
    r = get_retriever()
    print(f"Catalog size: {len(r.items)}\n")

    for label, query, must in PROBES:
        print(f"=== {label} ===")
        print(f"  query: {query!r}")
        results = r.search(query, top_k=10)
        names = [item["name"] for item, _ in results]
        for i, (item, score) in enumerate(results, 1):
            print(f"  {i:2d}. [{score:.3f}] {item['name']}  ({','.join(item['test_types'])})")

        if must:
            joined = " | ".join(names).lower()
            hits = [m for m in must if m.lower() in joined]
            misses = [m for m in must if m.lower() not in joined]
            mark = "OK" if not misses else "MISS"
            print(f"  -> {mark}: matched {hits}, missed {misses}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
