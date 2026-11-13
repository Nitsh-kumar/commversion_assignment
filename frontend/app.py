"""Standalone Streamlit frontend talking strictly to the FastAPI backend over HTTP."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from frontend.client import ChatbotApiClient

DEFAULT_BACKEND_URL = "http://localhost:8000"


def resolve_backend_url() -> str:
    """Resolve FastAPI backend URL from BACKEND_API_URL (or legacy CHATBOT_API_URL)."""
    return (
        os.getenv("BACKEND_API_URL", "").strip()
        or os.getenv("CHATBOT_API_URL", "").strip()
        or DEFAULT_BACKEND_URL
    )


def _mask_lead_state(state: dict[str, object]) -> dict[str, object]:
    """Mask phone and email fields before displaying lead state in the sidebar."""
    masked = dict(state)
    phone = masked.get("phone")
    if isinstance(phone, str) and len(phone) > 4:
        masked["phone"] = f"{'*' * (len(phone) - 4)}{phone[-4:]}"
    email = masked.get("email")
    if isinstance(email, str) and "@" in email:
        local, domain = email.split("@", 1)
        prefix = local[0] if local else ""
        masked["email"] = f"{prefix}***@{domain}"
    return masked


def render_app() -> None:
    """Render the Streamlit chat UI backed exclusively by FastAPI HTTP calls."""
    import streamlit as st

    load_dotenv(_REPO_ROOT / ".env")
    st.set_page_config(page_title="Ashcombe & Rowe Estates", page_icon="🏠", layout="wide")
    st.title("Ashcombe & Rowe Estates — Lead Qualification Assistant")

    backend_url = resolve_backend_url()
    client = ChatbotApiClient(base_url=backend_url)

    def _start_new_session() -> None:
        payload = client.start_session()
        st.session_state.session_id = payload["session_id"]
        st.session_state.messages = [{"role": "assistant", "content": payload["greeting"]}]
        st.session_state.lead_state = payload["state"]
        st.session_state.traces = []

    if "session_id" not in st.session_state:
        try:
            _start_new_session()
        except httpx.HTTPError as exc:
            st.error(
                f"Cannot reach FastAPI backend at `{backend_url}` ({exc}). "
                "Start the backend first with: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`."
            )
            st.stop()

    with st.sidebar:
        st.caption(f"Backend: `{backend_url}`")
        st.header("Lead State Inspector")
        if st.button("Reset Conversation", use_container_width=True):
            try:
                _start_new_session()
                st.rerun()
            except httpx.HTTPError as exc:
                st.error(f"Backend error while resetting session: {exc}")

        st.json(_mask_lead_state(st.session_state.get("lead_state", {})))

        traces = st.session_state.get("traces", [])
        if traces:
            st.subheader("Observability (Last Turn)")
            last = traces[-1]
            st.metric("Latency (ms)", last.get("latency_ms", 0))
            st.metric("Total Tokens", last.get("total_tokens", 0))
            st.caption(f"Tools: {', '.join(last.get('tools_used') or ['none'])}")

    if notice := st.session_state.pop("session_notice", None):
        st.warning(notice)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if raw_prompt := st.chat_input("Ask about buying, selling, a property, or our branches..."):
        prompt = raw_prompt.strip()
        if not prompt:
            st.rerun()

        try:
            data = client.send_message(st.session_state.session_id, prompt)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                _start_new_session()
                st.session_state["session_notice"] = (
                    "Previous session expired. Started a new session—please send your message again."
                )
                st.rerun()
            st.error(f"FastAPI backend request failed: {exc}")
            st.stop()
        except httpx.HTTPError as exc:
            st.error(f"FastAPI backend request failed: {exc}")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        reply = data["reply"]
        st.session_state.lead_state = data["state"]
        st.session_state.traces.append(data["trace"])

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":
    render_app()
