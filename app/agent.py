"""Conversational agent — planner → retrieval → writer pipeline.

Per /chat call:

    1. PLANNER (LLM, JSON-mode) reads the full message history and decides:
       - action ∈ {clarify, recommend, refine, compare, refuse}
       - slot extraction (role, seniority, skills, language, duration, ...)
       - retrieval_queries: one focused query per skill atom
       - compare_items / refusal_reason / clarifying_question

    2. RETRIEVAL runs hybrid BM25+dense search per query, dedupes, applies
       slot-derived filters (test_types, languages, max_duration), and
       enforces an exclude list to avoid recommending an item already
       declined in this conversation.

    3. WRITER (LLM) composes a grounded prose reply using ONLY the retrieved
       items. Recommendations are emitted as the structured `recommendations`
       array in the API response.

Guarantees enforced regardless of LLM behavior:
    - Every Recommendation.url is present in the local catalog (URL allow-list).
    - At most 10 recommendations.
    - When the LLM is unavailable or returns garbage, a deterministic fallback
      keeps responses schema-valid.
    - Turn-budget guard: by the 5th user turn, force a recommend action even
      if the planner would still clarify, so we always commit a shortlist
      within the 8-turn evaluator cap.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app import llm, prompts
from app.catalog import ALL_ITEMS, by_name_ci, is_known_url, to_recommendation
from app.retrieval import get_retriever
from app.schemas import ChatRequest, ChatResponse, Recommendation

log = logging.getLogger(__name__)


MAX_RECS = 7  # comfortable headroom under the 10-cap; some traces ship 7
TURN_BUDGET_FORCE_COMMIT = 5  # by user-turn 5, stop clarifying

# Catalog-default core items the agent often anchors on
DEFAULT_PERSONALITY_SLUG = "occupational-personality-questionnaire-opq32r"
DEFAULT_COGNITIVE_SLUG = "shl-verify-interactive-g"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_turn(req: ChatRequest) -> ChatResponse:
    history = [m.model_dump() for m in req.messages]
    user_turns = sum(1 for m in history if m["role"] == "user")

    plan = _plan(history, user_turns)
    action = plan.get("action", "clarify")

    # Turn-budget guard
    if action == "clarify" and user_turns >= TURN_BUDGET_FORCE_COMMIT:
        log.info("turn-budget guard: forcing recommend at user_turn %d", user_turns)
        action = "recommend"
        plan["action"] = action
        if not plan.get("retrieval_queries"):
            plan["retrieval_queries"] = [_last_user_text(history)]

    # Refuse — short-circuit, no retrieval, no writer
    if action == "refuse":
        return ChatResponse(
            reply=_refusal_text(plan.get("refusal_reason") or "out-of-scope"),
            recommendations=[],
            end_of_conversation=False,
        )

    # Clarify — short-circuit, no retrieval
    if action == "clarify":
        question = (plan.get("clarifying_question") or "").strip()
        if not question:
            question = _fallback_clarify(plan.get("slots") or {})
        return ChatResponse(
            reply=question,
            recommendations=[],
            end_of_conversation=False,
        )

    # Compare — narrow retrieval to two named items
    if action == "compare":
        items = _resolve_compare_items(plan.get("compare_items") or [])
        reply = _writer_compare(history, items)
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=bool(plan.get("end_of_conversation")),
        )

    # Recommend / refine
    items = _retrieve_for_recommend(plan, history)
    if not items:
        return ChatResponse(
            reply=_fallback_clarify(plan.get("slots") or {}),
            recommendations=[],
            end_of_conversation=False,
        )

    reply = _writer_recommend(history, plan, items, action)
    recs = [Recommendation(**to_recommendation(it)) for it in items]
    # Belt-and-braces: enforce URL allow-list before returning
    recs = [r for r in recs if is_known_url(r.url)][:MAX_RECS]

    return ChatResponse(
        reply=reply,
        recommendations=recs,
        end_of_conversation=bool(plan.get("end_of_conversation")),
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def _plan(history: list[dict[str, str]], user_turns: int) -> dict[str, Any]:
    """Run the planner LLM call. Falls back to a deterministic plan on failure."""
    if not llm.is_configured():
        return _heuristic_plan(history, user_turns)
    try:
        messages = [
            {"role": "system", "content": prompts.PLANNER_SYSTEM},
            {"role": "user", "content": prompts.planner_user_message(history)},
        ]
        plan = llm.chat_json(messages, temperature=0.1, max_tokens=512, timeout=10.0)
        # Normalize lists/dicts that may be missing
        plan.setdefault("slots", {})
        plan.setdefault("retrieval_queries", [])
        plan.setdefault("compare_items", [])
        return plan
    except Exception as exc:  # noqa: BLE001
        log.warning("planner failed (%s) — using heuristic", exc)
        return _heuristic_plan(history, user_turns)


_CONFIRM_RE = re.compile(
    r"^\s*(?:perfect|great|thanks?|confirmed?|locking? (?:it )?in|that'?s? (?:it|what we need|all|great|fine|perfect)|"
    r"understood|sounds? good|keep (?:it|the shortlist|that)|(?:all )?good|ok(?:ay)?\.?\s*$|"
    r"that works?|looks? good)",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"\b(?:difference|compare|vs\.?|versus|between)\b",
    re.IGNORECASE,
)
_INJECTION_RE = re.compile(
    r"\b(ignore (?:all )?(?:previous|prior) instructions?|"
    r"act as|you are now|disregard (?:the |your )?(?:above|prior|previous)|"
    r"system prompt|jailbreak|developer mode)\b",
    re.IGNORECASE,
)
_OFF_TOPIC_RE = re.compile(
    r"\b(weather|recipe|joke|poem|stock|crypto|bitcoin|"
    r"president|election|news|sports|movie|celebrity)\b",
    re.IGNORECASE,
)
_LEGAL_RE = re.compile(
    r"\b(legal advice|HIPAA compliance|GDPR compliance|EEOC|"
    r"discrimination law|labor law|employment law)\b",
    re.IGNORECASE,
)
_SKILL_RE = re.compile(
    r"\b(java|python|javascript|typescript|sql|aws|docker|spring|angular|react|"
    r"kubernetes|excel|word|powerpoint|hipaa|"
    r"manager|developer|engineer|analyst|sales|leadership|nurse|"
    r"representative|graduate|admin|operator|technician|"
    r"OPQ|verify|opq32|GSA)\b",
    re.IGNORECASE,
)


def _heuristic_plan(history: list[dict[str, str]], user_turns: int) -> dict[str, Any]:
    """Deterministic fallback when the LLM is unavailable.

    Strategy (in priority order):
    1. Refuse prompt-injection / off-topic / legal-advice asks.
    2. Confirmation turn ("Perfect", "Confirmed", "Keep it") → return same
       shortlist using cumulative context, mark end_of_conversation.
    3. Compare turn ("difference between OPQ and GSA") → compare action.
    4. Turn 1 with no skill/role words → clarify.
    5. Otherwise → recommend using cumulative user-turn context.
    """
    last = _last_user_text(history)
    all_user_text = " ".join(
        m["content"] for m in history if m["role"] == "user"
    )

    # --- Guard: injection / legal / off-topic ---
    if _INJECTION_RE.search(last):
        return _refuse_plan("injection")
    if _LEGAL_RE.search(last):
        return _refuse_plan("legal")
    all_has_skill = bool(_SKILL_RE.search(all_user_text))
    if _OFF_TOPIC_RE.search(last) and not all_has_skill:
        return _refuse_plan("off-topic")

    # --- Confirmation: user approved the current shortlist ---
    if user_turns > 1 and _CONFIRM_RE.match(last.strip()):
        # Use only the pre-confirmation user turns as the retrieval context —
        # "Perfect" / "Confirmed" carries no retrieval signal.
        pre_confirm_text = " ".join(
            m["content"]
            for m in history[:-1]  # exclude the confirmation message itself
            if m["role"] == "user"
        )
        return {
            "action": "recommend",
            "slots": {},
            "retrieval_queries": [pre_confirm_text],
            "compare_items": [],
            "clarifying_question": None,
            "end_of_conversation": True,
        }

    # --- Compare: explicit comparison request ---
    if _COMPARE_RE.search(last):
        # Try to extract the two items from the query
        names = _extract_compare_names(last)
        return {
            "action": "compare",
            "slots": {},
            "retrieval_queries": [],
            "compare_items": names,
            "clarifying_question": None,
            "end_of_conversation": False,
        }

    # --- Clarify: first turn with no meaningful signal ---
    is_short = len(last.split()) < 6
    if user_turns == 1 and is_short and not all_has_skill:
        return {
            "action": "clarify",
            "slots": {},
            "retrieval_queries": [],
            "compare_items": [],
            "clarifying_question": (
                "Happy to help — could you share the role you're hiring for, "
                "the seniority level, and whether this is for selection or "
                "development?"
            ),
            "end_of_conversation": False,
        }

    # --- Default: recommend using all accumulated user context ---
    return {
        "action": "recommend",
        "slots": {},
        "retrieval_queries": [all_user_text],
        "compare_items": [],
        "clarifying_question": None,
        "end_of_conversation": False,
    }


def _extract_compare_names(text: str) -> list[str]:
    """Heuristic: find two product names mentioned around comparison keywords."""
    # Strip comparison trigger words and split on "and"/"vs"
    cleaned = re.sub(
        r"\b(?:what(?:'?s| is) the )?difference between|compare|vs\.?|versus|between\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+|\s+versus\s+", cleaned, maxsplit=1)
    return [p.strip(" ?,") for p in parts if p.strip()][:2]


def _refuse_plan(reason: str) -> dict[str, Any]:
    return {
        "action": "refuse",
        "slots": {},
        "retrieval_queries": [],
        "compare_items": [],
        "refusal_reason": reason,
        "clarifying_question": None,
        "end_of_conversation": False,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _retrieve_for_recommend(
    plan: dict[str, Any], history: list[dict[str, str]]
) -> list[dict[str, Any]]:
    queries = [q for q in (plan.get("retrieval_queries") or []) if q and isinstance(q, str)]
    if not queries:
        queries = [_last_user_text(history)]

    slots = plan.get("slots") or {}
    filters = _slots_to_filters(slots, history)
    excluded = _slugs_user_declined(history)
    retr = get_retriever()

    # Run one retrieval per query atom; keep top-K from each, then merge by
    # best score across queries. This is the sub-query decomposition that
    # rescues multi-skill JDs from BM25 dilution.
    pool: dict[str, tuple[dict[str, Any], float]] = {}
    per_query_top = max(2, MAX_RECS // max(1, len(queries)) + 1)
    for q in queries:
        for item, score in retr.search(
            q,
            test_types=filters.get("test_types"),
            max_duration_minutes=filters.get("max_duration_minutes"),
            languages=filters.get("languages"),
            exclude_slugs=excluded,
            top_k=per_query_top,
        ):
            slug = item["slug"]
            cur = pool.get(slug)
            if cur is None or score > cur[1]:
                pool[slug] = (item, score)

    ranked = sorted(pool.values(), key=lambda x: -x[1])
    items = [it for it, _ in ranked]

    # Anchor adds: always inject OPQ32r unless the user explicitly declined
    # personality tests or excluded it. 9/10 traces include it. Verify G+ is
    # added only when slots request cognitive explicitly.
    if _user_declined_personality(history):
        excluded.add(DEFAULT_PERSONALITY_SLUG)
    if _user_declined_cognitive(history):
        excluded.add(DEFAULT_COGNITIVE_SLUG)
    items = _add_default_anchors(items, slots, excluded, heuristic_mode=not llm.is_configured())

    # Cap at MAX_RECS (room under the 10-item cap)
    return items[:MAX_RECS]


def _slots_to_filters(
    slots: dict[str, Any], history: list[dict[str, str]]
) -> dict[str, Any]:
    f: dict[str, Any] = {}
    # test_types: only apply if the user explicitly wants ONE category
    wants_p = bool(slots.get("wants_personality"))
    wants_c = bool(slots.get("wants_cognitive"))
    wants_s = bool(slots.get("wants_simulation"))
    explicit_only = sum([wants_p, wants_c, wants_s]) == 1 and _user_asked_for_only_type(history)
    if explicit_only:
        if wants_p:
            f["test_types"] = ["P"]
        elif wants_c:
            f["test_types"] = ["A"]
        elif wants_s:
            f["test_types"] = ["S"]

    if (lang := slots.get("language")):
        if isinstance(lang, str) and lang.strip():
            f["languages"] = [lang.strip()]

    if (md := slots.get("max_duration_minutes")):
        try:
            f["max_duration_minutes"] = int(md)
        except (TypeError, ValueError):
            pass
    return f


def _user_asked_for_only_type(history: list[dict[str, str]]) -> bool:
    """Crude detector: did the user ask for a single category only?"""
    last = _last_user_text(history).lower()
    return any(
        phrase in last
        for phrase in (
            "personality only",
            "only personality",
            "just personality",
            "cognitive only",
            "only cognitive",
            "just cognitive",
            "simulation only",
            "only simulation",
        )
    )


def _slugs_user_declined(history: list[dict[str, str]]) -> set[str]:
    """Slugs the user has asked to drop in this conversation."""
    out: set[str] = set()
    drop_re = re.compile(
        r"(?:drop|remove|exclude|skip|no|without|don'?t (?:include|add|need))\s+([^.,;]+)",
        re.IGNORECASE,
    )
    for m in history:
        if m["role"] != "user":
            continue
        for fragment in drop_re.findall(m["content"]):
            slug = _slug_for_fragment(fragment)
            if slug:
                out.add(slug)
    return out


def _slug_for_fragment(fragment: str) -> str | None:
    """Resolve a free-text fragment to a known catalog slug, or None."""
    frag = fragment.strip().lower()
    if not frag:
        return None
    # Direct name match
    item = by_name_ci(frag)
    if item:
        return item["slug"]
    # Substring scan
    best = None
    for it in ALL_ITEMS:
        nm = it["name"].lower()
        if frag in nm or nm in frag:
            best = it
            break
    return best["slug"] if best else None


_NO_PERSONALITY_RE = re.compile(
    r"\b(no personality|drop (?:the )?(?:OPQ|personality)|"
    r"skip (?:the )?(?:OPQ|personality)|without (?:the )?(?:OPQ|personality)|"
    r"personality only|no OPQ)\b",
    re.IGNORECASE,
)

_NO_COGNITIVE_RE = re.compile(
    r"\b(no (?:cognitive|verify|reasoning)|drop (?:the )?(?:verify|g\+|reasoning)|"
    r"skip (?:the )?(?:verify|g\+))\b",
    re.IGNORECASE,
)


def _user_declined_personality(history: list[dict[str, str]]) -> bool:
    text = " ".join(m["content"] for m in history if m["role"] == "user")
    return bool(_NO_PERSONALITY_RE.search(text))


def _user_declined_cognitive(history: list[dict[str, str]]) -> bool:
    text = " ".join(m["content"] for m in history if m["role"] == "user")
    return bool(_NO_COGNITIVE_RE.search(text))


def _add_default_anchors(
    items: list[dict[str, Any]],
    slots: dict[str, Any],
    excluded: set[str],
    heuristic_mode: bool = False,
) -> list[dict[str, Any]]:
    """Inject OPQ32r / Verify G+ defaults.

    In heuristic mode (no LLM), always add OPQ32r since it appears in 9/10
    traces. In LLM mode, respect the slots.wants_personality flag.
    In both modes, only add Verify G+ when explicitly requested.
    """
    have = {it["slug"] for it in items}

    add_personality = (
        heuristic_mode  # always include when no LLM
        or slots.get("wants_personality")
    )
    if add_personality and DEFAULT_PERSONALITY_SLUG not in have and DEFAULT_PERSONALITY_SLUG not in excluded:
        item = next((it for it in ALL_ITEMS if it["slug"] == DEFAULT_PERSONALITY_SLUG), None)
        if item:
            items.append(item)

    if slots.get("wants_cognitive") and DEFAULT_COGNITIVE_SLUG not in have and DEFAULT_COGNITIVE_SLUG not in excluded:
        item = next((it for it in ALL_ITEMS if it["slug"] == DEFAULT_COGNITIVE_SLUG), None)
        if item:
            items.append(item)
    return items


def _resolve_compare_items(names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in names[:2]:
        item = by_name_ci(n)
        if item is None:
            # Substring fallback
            n_lc = n.lower()
            for it in ALL_ITEMS:
                if n_lc in it["name"].lower():
                    item = it
                    break
        if item is not None and item not in out:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _writer_recommend(
    history: list[dict[str, str]],
    plan: dict[str, Any],
    items: list[dict[str, Any]],
    action: str,
) -> str:
    if not llm.is_configured():
        return _fallback_writer(items, action)
    try:
        messages = [
            {"role": "system", "content": prompts.WRITER_SYSTEM},
            {
                "role": "user",
                "content": prompts.writer_user_message(
                    history, plan.get("slots") or {}, items, action
                ),
            },
        ]
        return llm.chat(messages, temperature=0.3, max_tokens=400, timeout=10.0).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("writer failed (%s) — using fallback", exc)
        return _fallback_writer(items, action)


def _writer_compare(
    history: list[dict[str, str]], items: list[dict[str, Any]]
) -> str:
    if not items:
        return (
            "I couldn't find both of those assessments in the SHL catalog. "
            "Could you share the exact catalog names you'd like compared?"
        )
    if not llm.is_configured():
        return _fallback_compare(items)
    try:
        messages = [
            {"role": "system", "content": prompts.COMPARE_SYSTEM},
            {"role": "user", "content": prompts.compare_user_message(history, items)},
        ]
        return llm.chat(messages, temperature=0.2, max_tokens=300, timeout=10.0).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("compare writer failed (%s) — using fallback", exc)
        return _fallback_compare(items)


def _fallback_writer(items: list[dict[str, Any]], action: str) -> str:
    verb = "Updated shortlist" if action == "refine" else "Here is a shortlist"
    body = ", ".join(it["name"] for it in items)
    return f"{verb} of {len(items)} SHL assessments: {body}."


def _fallback_compare(items: list[dict[str, Any]]) -> str:
    parts = []
    for it in items:
        codes = ",".join(it.get("test_types") or []) or "?"
        d = it.get("duration_minutes")
        dur = f"{d} min" if d else ("untimed" if it.get("untimed") else "duration n/a")
        desc = (it.get("description") or "")[:160]
        parts.append(f"{it['name']} ({codes}, {dur}): {desc}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _last_user_text(history: list[dict[str, str]]) -> str:
    for m in reversed(history):
        if m["role"] == "user":
            return (m.get("content") or "").strip()
    return ""


def _fallback_clarify(slots: dict[str, Any]) -> str:
    if not slots.get("role"):
        return "Happy to help. What role are you hiring for?"
    if not slots.get("seniority"):
        return f"For the {slots['role']} role, what seniority level — graduate, mid, senior, or executive?"
    if not slots.get("purpose"):
        return "Got it — is this for selection (comparing candidates) or development (existing employees)?"
    return "Could you share a bit more context so I can pick the right battery?"


def _refusal_text(reason: str) -> str:
    table = {
        "off-topic": (
            "I can only help with SHL Individual Test Solutions — let me know "
            "what role or capability you'd like to assess and I'll suggest a battery."
        ),
        "legal": (
            "Legal and compliance interpretation is outside what I can advise on. "
            "Your legal or compliance team is the right resource for that. "
            "I can suggest assessments that cover related skills if useful."
        ),
        "injection": (
            "I can only help with SHL assessment recommendations. "
            "What role or capability would you like to assess?"
        ),
        "out-of-scope": (
            "That's outside the SHL assessment catalog I can recommend from. "
            "Tell me about the role or capability you'd like to assess and "
            "I'll put together a shortlist."
        ),
    }
    return table.get(reason, table["out-of-scope"])
