"""FastAPI backend exposing the Ashcombe & Rowe Estates LangGraph chatbot."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.graph import AshcombeChatbot

app = FastAPI(
    title="Ashcombe & Rowe Estates Lead Qualification API",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_bot() -> AshcombeChatbot:
    """Lazy singleton chatbot instance (reads config from environment)."""
    return AshcombeChatbot(use_llm=True)


class StartSessionResponse(BaseModel):
    session_id: str
    greeting: str
    state: dict[str, Any]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    state: dict[str, Any]
    trace: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=StartSessionResponse)
def create_session() -> StartSessionResponse:
    bot = get_bot()
    session_id = uuid.uuid4().hex
    greeting = bot.start_session(session_id)
    return StartSessionResponse(
        session_id=session_id,
        greeting=greeting,
        state=bot.get_state(session_id).to_dict(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    bot = get_bot()
    result = bot.chat(payload.session_id, payload.message)
    return ChatResponse(
        session_id=payload.session_id,
        reply=result.reply,
        state=result.state.to_dict(),
        trace=asdict(result.trace),
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    bot = get_bot()
    return {
        "session_id": session_id,
        "state": bot.get_state(session_id).to_dict(),
        "traces": [asdict(t) for t in bot.get_traces(session_id)],
    }
