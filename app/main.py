"""FastAPI app exposing /health and /chat.

The /chat handler is a thin shell here; the real work is delegated to
app.agent.run_turn so the orchestration is testable without HTTP.

The endpoint is stateless: every request carries the full message history.
We never store anything per-conversation server-side.
"""
from __future__ import annotations

import logging
import os

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.catalog import ALL_ITEMS
from app.schemas import ChatRequest, ChatResponse, HealthResponse

STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger("shl-agent")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="SHL Conversational Recommender", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("startup")
def _startup() -> None:
    log.info("loaded catalog: %d items", len(ALL_ITEMS))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Lazy import keeps cold-start fast for /health probes.
    from app.agent import run_turn

    try:
        return run_turn(req)
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        log.exception("chat handler failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
