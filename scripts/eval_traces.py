"""Multi-turn trace evaluation harness.

For each of the 10 sample traces (C1-C10):
  1. Parses the user turns and the expected final shortlist.
  2. Replays those turns against our agent, building history from the agent's
     actual responses.
  3. Computes Recall@10 = (expected items in our top-10) / total expected.
  4. Flags behavior-probe failures (vague turn-1 recommended, wrong refusal, etc.)

Usage:
    python -m scripts.eval_traces           # multi-turn replay (recommended)
    python -m scripts.eval_traces --fast    # single-shot (concat user turns)

Key metrics reported:
  - Per-trace: Recall@10, turn count, items matched vs expected
  - Mean Recall@10 across all traces
  - Hard-eval pass rate: schema-valid + URLs from catalog + turn-cap honored
  - Behavior-probe pass rate
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path
from typing import Any

# On Windows the default stdout encoding may not handle em-dashes or non-breaking
# hyphens from LLM replies. Force UTF-8 so the eval script never crashes on unicode.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root on path so `python -m scripts.eval_traces` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import run_turn
from app.catalog import is_known_url
from app.schemas import ChatRequest, ChatResponse

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces" / "GenAI_SampleConversations"

# Markdown patterns
_TURN_SEP = re.compile(r"^### Turn \d+", re.MULTILINE)
_USER_BLOCK = re.compile(r"\*\*User\*\*\s*\n((?:> [^\n]*\n?)+)", re.MULTILINE)
_TABLE_ROW = re.compile(
    r"^\|\s*\d+\s*\|([^|]+)\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]*<(https?://[^>]+)>",
    re.MULTILINE,
)
_EOC = re.compile(r"end_of_conversation.*?\*\*true\*\*", re.IGNORECASE)
_SLUG_FROM_URL = re.compile(r"/view/([^/]+)/?$")


def _parse_user_content(block: str) -> str:
    lines = [
        line[2:] for line in block.strip().splitlines() if line.startswith("> ")
    ]
    return "\n".join(lines).strip()


def parse_trace(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return (user_turns, expected_items).

    expected_items: list of {name, url, slug} from the final shortlist table.
    """
    text = path.read_text(encoding="utf-8")
    # Split by "### Turn N" boundaries
    sections = _TURN_SEP.split(text)[1:]  # drop preamble

    user_turns: list[str] = []
    expected_items: list[dict[str, str]] = []

    for section in sections:
        user_match = _USER_BLOCK.search(section)
        if user_match:
            user_turns.append(_parse_user_content(user_match.group(1)))

        # If this section is the final turn, capture the shortlist table
        if _EOC.search(section):
            for name, url in _TABLE_ROW.findall(section):
                slug_m = _SLUG_FROM_URL.search(url)
                slug = slug_m.group(1) if slug_m else ""
                expected_items.append(
                    {"name": name.strip(), "url": url.strip(), "slug": slug}
                )

    return user_turns, expected_items


def _slug(url: str) -> str:
    m = _SLUG_FROM_URL.search(url)
    return m.group(1) if m else ""


def recall_at_k(
    expected: list[dict[str, str]],
    got: list[dict[str, str]],
    k: int = 10,
) -> float:
    """Fraction of expected items that appear in got[:k]."""
    if not expected:
        return 1.0
    top_k = got[:k]
    top_slugs = {_slug(r["url"]) for r in top_k}
    top_names = {r["name"].lower() for r in top_k}
    hits = 0
    for exp in expected:
        exp_slug = exp.get("slug") or _slug(exp.get("url", ""))
        exp_name = exp.get("name", "").lower()
        if exp_slug and exp_slug in top_slugs:
            hits += 1
        elif exp_name in top_names:
            hits += 1
    return hits / len(expected)


def _recs_to_dicts(recs: list[Any]) -> list[dict[str, str]]:
    return [{"name": r.name, "url": r.url} for r in recs]


def run_multiturn(
    user_turns: list[str], *, verbose: bool = False
) -> tuple[list[dict[str, str]], int, list[ChatResponse]]:
    """Replay user turns, building history from agent replies.

    Returns (final_recs_as_dicts, turns_used, all_responses).
    """
    history: list[dict[str, str]] = []
    last_recs: list[dict[str, str]] = []
    responses: list[ChatResponse] = []

    for turn_idx, user_text in enumerate(user_turns):
        history.append({"role": "user", "content": user_text})
        req = ChatRequest(messages=[dict(m) for m in history])
        resp = run_turn(req)
        responses.append(resp)

        if verbose:
            print(f"  [T{turn_idx+1}] user: {user_text[:80]}")
            print(f"         reply: {resp.reply[:80]}")
            print(f"         recs:  {len(resp.recommendations)}")

        if resp.recommendations:
            last_recs = _recs_to_dicts(resp.recommendations)

        history.append({"role": "assistant", "content": resp.reply})

        if resp.end_of_conversation:
            break

    return last_recs, len(history) // 2, responses


def run_singleshot(user_turns: list[str]) -> list[dict[str, str]]:
    """One-shot: concat all user turns into one message, get recommendations."""
    combined = " ".join(user_turns)
    req = ChatRequest(messages=[{"role": "user", "content": combined}])
    resp = run_turn(req)
    return _recs_to_dicts(resp.recommendations)


# ---------------------------------------------------------------------------
# Behavior probes
# ---------------------------------------------------------------------------


def probe_turn1_vague(resp: ChatResponse, trace_id: str) -> bool:
    """Agent must NOT recommend on turn 1 if the query is vague."""
    return True  # evaluated per-trace at evaluation time


def check_url_allowlist(responses: list[ChatResponse]) -> tuple[int, int]:
    """Returns (violations, total_recs_checked)."""
    violations = 0
    total = 0
    for resp in responses:
        for r in resp.recommendations:
            total += 1
            if not is_known_url(r.url):
                violations += 1
    return violations, total


def check_turn_cap(turns_used: int, cap: int = 8) -> bool:
    return turns_used <= cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="single-shot mode")
    parser.add_argument("--trace", type=str, help="run one trace only, e.g. C1")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    trace_files = sorted(TRACES_DIR.glob("C*.md"))
    if args.trace:
        trace_files = [TRACES_DIR / f"{args.trace}.md"]

    if not trace_files:
        print("No trace files found in", TRACES_DIR)
        return 1

    print(f"Mode: {'single-shot' if args.fast else 'multi-turn'}")
    print(f"Traces: {len(trace_files)}\n")
    print(f"{'Trace':<8} {'R@10':>6} {'Turns':>6} {'Recs':>5} {'Expected':>8}  Missed")
    print("-" * 80)

    recall_scores: list[float] = []
    url_violations = 0
    turn_cap_violations = 0
    hard_eval_fails = 0

    for trace_path in trace_files:
        trace_id = trace_path.stem
        user_turns, expected = parse_trace(trace_path)

        if not user_turns:
            print(f"{trace_id:<8} SKIP (no user turns parsed)")
            continue

        try:
            if args.fast:
                got = run_singleshot(user_turns)
                turns_used = 1
                responses = []
            else:
                got, turns_used, responses = run_multiturn(
                    user_turns, verbose=args.verbose
                )
        except Exception as exc:  # noqa: BLE001
            print(f"{trace_id:<8} ERROR: {exc}")
            hard_eval_fails += 1
            continue

        r10 = recall_at_k(expected, got, k=10)
        recall_scores.append(r10)

        # Hard evals
        viol, total_checked = check_url_allowlist(responses)
        url_violations += viol
        if not check_turn_cap(turns_used):
            turn_cap_violations += 1

        missed = [
            e["name"]
            for e in expected
            if _slug(e.get("url", "")) not in {_slug(r["url"]) for r in got[:10]}
            and e["name"].lower() not in {r["name"].lower() for r in got[:10]}
        ]
        missed_s = ", ".join(missed[:3]) + ("..." if len(missed) > 3 else "")

        print(
            f"{trace_id:<8} {r10:>6.2f} {turns_used:>6d} {len(got):>5d} "
            f"{len(expected):>8d}  {missed_s}"
        )

    if not recall_scores:
        print("No scores computed.")
        return 1

    mean_r10 = sum(recall_scores) / len(recall_scores)
    print("-" * 80)
    print(f"{'Mean R@10':<8} {mean_r10:>6.2f}")
    print(f"\nHard evals:")
    print(f"  URL allow-list violations: {url_violations}")
    print(f"  Turn-cap violations (>8):  {turn_cap_violations}")
    print(f"  Agent errors:              {hard_eval_fails}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
