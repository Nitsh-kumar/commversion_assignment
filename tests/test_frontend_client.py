"""Tests enforcing strict frontend/backend separation and verifying the HTTP client."""

from __future__ import annotations

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.core.config import Settings
from backend.core.engine import AshcombeChatbot
from backend.core.knowledge_base import AgencyKnowledgeBase
from frontend.app import _mask_lead_state, resolve_backend_url
from frontend.client import ChatbotApiClient
from tests.test_validators_and_tools import DeterministicHashEmbeddings

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def test_frontend_has_zero_backend_imports() -> None:
    """Enforce that no module inside frontend/ imports anything from backend."""
    for py_file in (REPO_ROOT / "frontend").rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("backend"), (
                        f"{py_file} imports backend module '{alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("backend"), (
                    f"{py_file} imports from backend module '{node.module}'"
                )


def test_frontend_http_client_talks_to_fastapi_backend() -> None:
    settings = Settings(data_dir=DATA_DIR, llm_api_key="", embedding_provider="local")
    kb = AgencyKnowledgeBase.from_data_dir(DATA_DIR, DeterministicHashEmbeddings())
    offline_bot = AshcombeChatbot(settings=settings, kb=kb, use_llm=False)

    fastapi_app = create_app(bot_factory=lambda: offline_bot)
    test_http = TestClient(fastapi_app)
    client = ChatbotApiClient(base_url="http://testserver", http_client=test_http)

    assert client.health() == {"status": "ok"}

    session = client.start_session()
    sid = session["session_id"]
    assert "Ashcombe & Rowe" in session["greeting"]

    r1 = client.send_message(sid, "Alice")
    assert r1["state"]["visitor_name"] == "Alice"
    assert r1["trace"]["step_after"] == "BUY_OR_SELL"

    snap = client.get_session(sid)
    assert snap["session_id"] == sid
    assert len(snap["traces"]) == 1


def test_resolve_backend_url_env_precedence(monkeypatch) -> None:
    monkeypatch.delenv("BACKEND_API_URL", raising=False)
    monkeypatch.delenv("CHATBOT_API_URL", raising=False)
    assert resolve_backend_url() == "http://localhost:8000"

    monkeypatch.setenv("BACKEND_API_URL", "http://127.0.0.1:9000")
    assert resolve_backend_url() == "http://127.0.0.1:9000"


def test_mask_lead_state_masks_pii() -> None:
    masked = _mask_lead_state(
        {
            "visitor_name": "Alice",
            "phone": "07700 900123",
            "email": "alice@example.com",
            "postcode": "SW3 3RB",
        }
    )
    assert masked["visitor_name"] == "Alice"
    assert masked["postcode"] == "SW3 3RB"
    assert masked["phone"] == "********0123"
    assert masked["email"] == "a***@example.com"
