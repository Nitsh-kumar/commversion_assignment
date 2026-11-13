"""Streamlit chat UI for Ashcombe & Rowe Estates with live state and trace inspector."""

from __future__ import annotations

import os
import uuid
import httpx
import streamlit as st

from src.graph import AshcombeChatbot


st.set_page_config(page_title="Ashcombe & Rowe Estates", page_icon="🏠", layout="wide")
st.title("Ashcombe & Rowe Estates — Lead Qualification Assistant")

API_URL = os.getenv("CHATBOT_API_URL", "").strip()


@st.cache_resource
def _local_bot() -> AshcombeChatbot:
    return AshcombeChatbot(use_llm=True)


def _start_new_session() -> None:
    if API_URL:
        resp = httpx.post(f"{API_URL}/sessions", timeout=30.0)
        resp.raise_for_status()
        payload = resp.json()
        st.session_state.session_id = payload["session_id"]
        st.session_state.messages = [{"role": "assistant", "content": payload["greeting"]}]
        st.session_state.lead_state = payload["state"]
        st.session_state.traces = []
    else:
        bot = _local_bot()
        sid = uuid.uuid4().hex
        greeting = bot.start_session(sid)
        st.session_state.session_id = sid
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.session_state.lead_state = bot.get_state(sid).to_dict()
        st.session_state.traces = []


if "session_id" not in st.session_state:
    _start_new_session()

with st.sidebar:
    st.header("Lead State Inspector")
    if st.button("Reset Conversation", use_container_width=True):
        _start_new_session()
        st.rerun()

    st.json(st.session_state.get("lead_state", {}))

    traces = st.session_state.get("traces", [])
    if traces:
        st.subheader("Observability (Last Turn)")
        last = traces[-1]
        st.metric("Latency (ms)", last.get("latency_ms", 0))
        st.metric("Total Tokens", last.get("total_tokens", 0))
        st.caption(f"Tools: {', '.join(last.get('tools_used') or ['none'])}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about buying, selling, a property, or our branches..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if API_URL:
        resp = httpx.post(
            f"{API_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": prompt},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["reply"]
        st.session_state.lead_state = data["state"]
        st.session_state.traces.append(data["trace"])
    else:
        bot = _local_bot()
        result = bot.chat(st.session_state.session_id, prompt)
        reply = result.reply
        st.session_state.lead_state = result.state.to_dict()
        st.session_state.traces.append(
            {
                "latency_ms": result.trace.latency_ms,
                "total_tokens": result.trace.total_tokens,
                "tools_used": result.trace.tools_used,
            }
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.rerun()
