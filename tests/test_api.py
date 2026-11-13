"""Smoke tests for the FastAPI endpoints using an offline AshcombeChatbot instance."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import src.api as api_module
from src.config import Settings
from src.graph import AshcombeChatbot
from src.tools import AgencyKnowledgeBase
from tests.test_validators_and_tools import DeterministicHashEmbeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_fastapi_session_and_chat(monkeypatch) -> None:
    settings = Settings(data_dir=DATA_DIR, llm_api_key="", embedding_provider="local")
    kb = AgencyKnowledgeBase.from_data_dir(DATA_DIR, DeterministicHashEmbeddings())
    offline_bot = AshcombeChatbot(settings=settings, kb=kb, use_llm=False)

    monkeypatch.setattr(api_module, "get_bot", lambda: offline_bot)
    client = TestClient(api_module.app)

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
