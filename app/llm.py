"""OpenRouter chat-completion client.

Reads OPENROUTER_API_KEY / OPENROUTER_MODEL / OPENROUTER_BASE_URL from env
(or .env via python-dotenv if present). Exposes:

  - chat(messages, *, json_mode=False, temperature=..., timeout=...) -> str
  - chat_json(messages, ...) -> dict        # parses + repairs trailing prose
  - is_configured() -> bool                  # True iff API key present

Designed for graceful failure: if the API key is missing OR the request raises,
the caller can fall back to a deterministic non-LLM path. That guarantees the
/chat endpoint always returns a schema-valid response even in degraded modes.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

try:  # optional .env loading
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


def _model() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def _base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def is_configured() -> bool:
    key = _api_key()
    return bool(key) and not key.startswith("sk-or-...")


class LLMError(RuntimeError):
    """Raised when an LLM call cannot succeed; callers should fall back."""


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _post_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    json_mode: bool,
    temperature: float,
    timeout: float,
    max_tokens: int | None,
) -> str:
    api_key = _api_key()
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set")

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter uses these for analytics/attribution; harmless if blank.
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://shl-recommender.local"),
        "X-Title": "SHL Conversational Recommender",
    }

    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{_base_url()}/chat/completions", json=body, headers=headers)
        if r.status_code >= 400:
            raise LLMError(
                f"OpenRouter {r.status_code}: {r.text[:300]}"
            )
        data = r.json()

    try:
        content = data["choices"][0]["message"]["content"]
        if content is None:
            raise LLMError("model returned null content")
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Malformed OpenRouter response: {data}") from exc


def chat(
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    temperature: float = 0.2,
    timeout: float = 20.0,
    max_tokens: int | None = 1024,
    model: str | None = None,
) -> str:
    """Single chat completion. Raises LLMError on failure."""
    try:
        return _post_chat(
            messages,
            model=model or _model(),
            json_mode=json_mode,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMError(str(exc)) from exc


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of the model output.

    Tolerates:
    - Fenced code blocks (```json ... ```)
    - Leading/trailing prose
    - Models that escape double-quotes as \" instead of emitting raw "
    - Literal newlines inside string values (replaced with \\n)
    """
    if text is None:
        raise LLMError("model returned null content")
    s = text.strip()

    # Strip markdown fences
    fence = _FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()

    # Some models output escaped JSON (backslash-quoted keys/values)
    if s.startswith('{\\"') or '\\"' in s[:50]:
        s = s.replace('\\"', '"')

    # Find the first balanced { ... } object
    start = s.find("{")
    if start < 0:
        raise LLMError(f"no JSON object found in: {text[:200]}")
    depth = 0
    end = -1
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise LLMError(f"unbalanced JSON in: {text[:200]}")
    candidate = s[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Last resort: strip literal control characters that break json.loads
    cleaned = re.sub(r"[\x00-\x1f]", " ", candidate)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON: {exc}; payload={candidate[:300]}") from exc


def chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    timeout: float = 20.0,
    max_tokens: int | None = 1024,
    model: str | None = None,
) -> dict[str, Any]:
    """Chat completion that must return a JSON object."""
    raw = chat(
        messages,
        json_mode=True,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        model=model,
    )
    return _extract_json(raw)
