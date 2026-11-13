"""FastAPI backend exposing the Ashcombe & Rowe Estates Conversation Engine."""

from __future__ import annotations

from dataclasses import asdict
import threading
from typing import Any, Callable
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.core.config import get_settings
from backend.core.engine import AshcombeChatbot


class StartSessionResponse(BaseModel):
    session_id: str
    greeting: str
    state: dict[str, Any]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

    @field_validator("session_id", "message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty or whitespace-only")
        return stripped


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    state: dict[str, Any]
    trace: dict[str, Any]


def create_app(bot_factory: Callable[[], AshcombeChatbot] | None = None) -> FastAPI:
    """Create the FastAPI backend app backed by an injected or lazily initialized engine."""
    factory = bot_factory or (lambda: AshcombeChatbot(use_llm=True))
    bot_lock = threading.Lock()
    cached_bot: AshcombeChatbot | None = None

    def resolve_bot() -> AshcombeChatbot:
        nonlocal cached_bot
        if cached_bot is None:
            with bot_lock:
                if cached_bot is None:
                    cached_bot = factory()
        return cached_bot

    cors_origins = [
        origin.strip() for origin in get_settings().cors_origins.split(",") if origin.strip()
    ] or ["http://localhost:8501", "http://127.0.0.1:8501"]

    application = FastAPI(
        title="Ashcombe & Rowe Estates Lead Qualification API",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/sessions", response_model=StartSessionResponse)
    def create_session() -> StartSessionResponse:
        bot = resolve_bot()
        session_id = uuid.uuid4().hex
        greeting = bot.start_session(session_id)
        return StartSessionResponse(
            session_id=session_id,
            greeting=greeting,
            state=bot.get_state(session_id).to_dict(),
        )

    @application.post("/chat", response_model=ChatResponse)
    def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        bot = resolve_bot()
        if not bot.has_session(payload.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        result = bot.chat(payload.session_id, payload.message)
        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            state=result.state.to_dict(),
            trace=asdict(result.trace),
        )

    @application.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        bot = resolve_bot()
        if not bot.has_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "state": bot.get_state(session_id).to_dict(),
            "traces": [asdict(t) for t in bot.get_traces(session_id)],
        }

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
