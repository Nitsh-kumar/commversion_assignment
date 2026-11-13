"""Comprehensive end-to-end tests covering LLM mock orchestration, retries, edge cases, CLI, and real FAISS RAG."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from src.config import Settings
from src.flow import Step
from src.graph import AshcombeChatbot
from src.llm_factory import call_with_retry, create_embeddings
from src.nlu import TurnAnalysisSchema
from src.tools import AgencyKnowledgeBase
from tests.test_validators_and_tools import DeterministicHashEmbeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ScriptedMockChatModel(BaseChatModel):
    """Fake BaseChatModel supporting structured output, token metadata, and simulated 429 errors."""

    _structured_responses: list[TurnAnalysisSchema | Exception] = PrivateAttr(default_factory=list)
    _text_responses: list[str | Exception] = PrivateAttr(default_factory=list)
    _s_idx: int = PrivateAttr(default=0)
    _t_idx: int = PrivateAttr(default=0)

    def __init__(
        self,
        structured_responses: list[TurnAnalysisSchema | Exception] | None = None,
        text_responses: list[str | Exception] | None = None,
    ) -> None:
        super().__init__()
        self._structured_responses = list(structured_responses or [])
        self._text_responses = list(text_responses or [])

    @property
    def _llm_type(self) -> str:
        return "scripted-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._t_idx < len(self._text_responses):
            item = self._text_responses[self._t_idx]
            self._t_idx += 1
            if isinstance(item, Exception):
                raise item
            text = item
        else:
            text = "Verified answer from agency records."
        msg = AIMessage(
            content=text,
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema: Any, include_raw: bool = False, **kwargs: Any) -> Any:
        def _invoke(messages: list[BaseMessage]) -> Any:
            if self._s_idx >= len(self._structured_responses):
                raise RuntimeError("No more scripted structured responses")
            item = self._structured_responses[self._s_idx]
            self._s_idx += 1
            if isinstance(item, Exception):
                raise item
            raw_msg = AIMessage(
                content=json.dumps(item.model_dump()),
                usage_metadata={"input_tokens": 80, "output_tokens": 25, "total_tokens": 105},
            )
            if include_raw:
                return {"raw": raw_msg, "parsed": item, "parsing_error": None}
            return item

        return RunnableLambda(_invoke)


def _make_offline_bot(use_real_embeddings: bool = False) -> AshcombeChatbot:
    settings = Settings(data_dir=DATA_DIR, llm_api_key="", embedding_provider="local")
    emb = create_embeddings(settings) if use_real_embeddings else DeterministicHashEmbeddings()
    kb = AgencyKnowledgeBase.from_data_dir(DATA_DIR, emb)
    return AshcombeChatbot(settings=settings, kb=kb, use_llm=False)


def test_first_message_names_property_jumps_to_lane_c() -> None:
    bot = _make_offline_bot()
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    res = bot.chat(sid, "Hi, my name is Alice and I want to ask about Rosemount House")
    assert res.state.visitor_name == "Alice"
    assert res.state.selected_property_id == "AR-0008"
    assert res.state.current_step == Step.E1_ASK_PHONE
    assert "3,345,000" in res.reply
    assert "phone number" in res.reply.lower()


def test_mid_flow_property_side_question_and_resume() -> None:
    bot = _make_offline_bot()
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    bot.chat(sid, "Noah")
    bot.chat(sid, "buy")
    bot.chat(sid, "no")
    bot.chat(sid, "new-build")
    assert bot.get_state(sid).current_step == Step.A1_ASK_BUDGET

    # Mid-flow specific property question must look up properties_for_buyers.csv and resume A1_ASK_BUDGET
    res = bot.chat(sid, "How many bathrooms does Rosemount House have?")
    assert "1 bathrooms" in res.reply or "1 bathroom" in res.reply.lower() or "detached house" in res.reply.lower()
    assert "property_lookup" in res.trace.tools_used
    assert "budget" in res.reply.lower()
    assert res.state.current_step == Step.A1_ASK_BUDGET


def test_lane_c_under_offer_resale_routes_to_b2_without_reasking_budget() -> None:
    bot = _make_offline_bot()
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    bot.chat(sid, "Zoe")
    bot.chat(sid, "buy")
    bot.chat(sid, "yes")
    r_prop = bot.chat(sid, "87 Church Row").reply
    assert "under offer" in r_prop.lower()
    assert bot.get_state(sid).current_step == Step.C5_ASK_BUDGET

    # Resale has no minimum: £600k should proceed directly to B2_ASK_POSTCODE
    r_budget = bot.chat(sid, "£600k").reply
    assert "postcode" in r_budget.lower()
    assert bot.get_state(sid).current_step == Step.B2_ASK_POSTCODE

    # NW3 2LD has 79 Church Row (AR-0045) available
    r_pc = bot.chat(sid, "nw3 2ld").reply
    assert "79 Church Row" in r_pc
    assert bot.get_state(sid).current_step == Step.E1_ASK_PHONE


def test_reassist_keeps_contact_details_across_second_journey() -> None:
    bot = _make_offline_bot()
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    # First journey: Seller
    for t in ["Clara", "sell", "GU5 7AQ", "07700 900333", "clara@example.com"]:
        bot.chat(sid, t)
    assert bot.get_state(sid).current_step == Step.E3_REASSIST

    # Reassist -> new-build buyer
    r_reassist = bot.chat(sid, "Yes, I also want to buy a new-build").reply
    assert "budget" in r_reassist.lower()
    bot.chat(sid, "£2m")
    r_match = bot.chat(sid, "SW3 3RB").reply
    # Contact details are already kept from first journey, so E3 is entered directly
    assert "already have your contact details" in r_match.lower()
    assert "07700 900333" in r_match
    assert bot.get_state(sid).current_step == Step.E3_REASSIST

    r_bye = bot.chat(sid, "No thanks").reply
    assert "Thank you for chatting with us. Goodbye." in r_bye
    assert bot.get_state(sid).current_step == Step.ENDED


def test_real_embeddings_out_of_data_faq_returns_office_number() -> None:
    bot = _make_offline_bot(use_real_embeddings=True)
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)
    bot.chat(sid, "Mia")

    res = bot.chat(sid, "Do you offer helicopter tours for viewings?")
    assert "020 7946 0100" in res.reply
    assert "buy or sell" in res.reply.lower()
    assert res.state.current_step == Step.BUY_OR_SELL


def test_llm_structured_output_rate_limit_retry_and_fallback() -> None:
    mock_llm = ScriptedMockChatModel(
        structured_responses=[
            RuntimeError("429 rate limit exceeded"),
            TurnAnalysisSchema(message_kind="flow_answer", visitor_name="Owen"),
            TurnAnalysisSchema(message_kind="faq_question", side_question_text="What are your selling fees?"),
            RuntimeError("Permanent upstream error"),  # triggers deterministic fallback!
        ],
        text_responses=[
            "Our standard sole-agency selling fee is 1.5% plus VAT.",
        ],
    )
    settings = Settings(data_dir=DATA_DIR, llm_api_key="dummy", llm_model="llama-3.3-70b-versatile")
    kb = AgencyKnowledgeBase.from_data_dir(DATA_DIR, DeterministicHashEmbeddings())
    bot = AshcombeChatbot(settings=settings, llm=mock_llm, kb=kb, use_llm=True)

    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    # Turn 1: hits 429 once -> call_with_retry recovers and consumes response #2 ("Owen")
    r1 = bot.chat(sid, "Hi I'm Owen")
    assert r1.state.visitor_name == "Owen"
    assert r1.state.current_step == Step.BUY_OR_SELL
    assert r1.trace.total_tokens == 105

    # Turn 2: FAQ question -> LLM synthesizes grounded answer + resumes BUY_OR_SELL
    r2 = bot.chat(sid, "What are your selling fees?")
    assert "1.5%" in r2.reply
    assert r2.state.current_step == Step.BUY_OR_SELL
    assert r2.trace.total_tokens == 175  # 105 NLU + 70 synthesis

    # Turn 3: permanent LLM error -> graceful deterministic heuristic fallback still works!
    r3 = bot.chat(sid, "I want to sell")
    assert r3.state.intent == "sell"
    assert r3.state.current_step == Step.D1_ASK_POSTCODE


def test_call_with_retry_non_transient_raises_immediately() -> None:
    calls = 0

    def _fail() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("Invalid request schema")

    try:
        call_with_retry(_fail, max_attempts=3, base_delay=0.01)
    except ValueError:
        pass
    assert calls == 1


def test_cli_subprocess_e2e() -> None:
    scripted_stdin = "\n".join(
        [
            "Toby",
            "sell",
            "GU5 7AQ",
            "07700 900555",
            "toby@example.com",
            "no",
            "quit",
        ]
    ) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "--offline", "--trace"],
        input=scripted_stdin,
        text=True,
        capture_output=True,
        cwd=str(DATA_DIR.parent),
        timeout=90,
        check=True,
    )
    assert "Ashcombe & Rowe Estates" in proc.stdout
    assert "GU5 7AQ" in proc.stdout
    assert "Thank you for chatting with us. Goodbye." in proc.stdout
    assert "[trace]" in proc.stdout
