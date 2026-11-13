"""Tests for the FastAPI delivery adapter using factory injection (no monkeypatching)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.core.config import Settings
from backend.core.engine import AshcombeChatbot
from backend.core.knowledge_base import AgencyKnowledgeBase
from tests.test_validators_and_tools import DeterministicHashEmbeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_fastapi_session_and_chat() -> None:
    settings = Settings(data_dir=DATA_DIR, llm_api_key="", embedding_provider="local")
    kb = AgencyKnowledgeBase.from_data_dir(DATA_DIR, DeterministicHashEmbeddings())
    offline_bot = AshcombeChatbot(settings=settings, kb=kb, use_llm=False)

    client = TestClient(create_app(bot_factory=lambda: offline_bot))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    session_resp = client.post("/sessions")
    assert session_resp.status_code == 200
    sid = session_resp.json()["session_id"]

    r1 = client.post("/chat", json={"session_id": sid, "message": "Oliver"})
    assert r1.status_code == 200
    assert r1.json()["state"]["visitor_name"] == "Oliver"

    r2 = client.post("/chat", json={"session_id": sid, "message": "sell"})
    assert r2.status_code == 200
    assert r2.json()["state"]["current_step"] == "D1_ASK_POSTCODE"

    snap = client.get(f"/sessions/{sid}")
    assert snap.status_code == 200
    assert len(snap.json()["traces"]) == 2

    # Unknown session IDs return 404 on both GET and POST
    assert client.get("/sessions/nonexistent-session").status_code == 404
    assert client.post("/chat", json={"session_id": "nonexistent-session", "message": "Hi"}).status_code == 404

    # Whitespace-only payloads fail schema validation (422)
    assert client.post("/chat", json={"session_id": sid, "message": "   "}).status_code == 422
    assert client.post("/chat", json={"session_id": "   ", "message": "Hi"}).status_code == 422


def test_fastapi_bot_factory_runs_once_under_concurrent_requests() -> None:
    from concurrent.futures import ThreadPoolExecutor

    settings = Settings(data_dir=DATA_DIR, llm_api_key="", embedding_provider="local")
    kb = AgencyKnowledgeBase.from_data_dir(DATA_DIR, DeterministicHashEmbeddings())
    calls = 0

    def _factory() -> AshcombeChatbot:
        nonlocal calls
        calls += 1
        return AshcombeChatbot(settings=settings, kb=kb, use_llm=False)

    client = TestClient(create_app(bot_factory=_factory))
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: client.post("/sessions"), range(6)))

    assert all(r.status_code == 200 for r in responses)
    assert calls == 1
