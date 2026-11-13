"""Comprehensive end-to-end tests covering LLM mock orchestration, retries, edge cases, CLI, and real FAISS RAG."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import TypeVar
import uuid

from langchain_core.messages import AIMessage
from pydantic import BaseModel
import pytest

from backend.cli import run_cli
from backend.core.config import Settings, get_settings
from backend.core.engine import AshcombeChatbot
from backend.core.flow import Step, TurnAnalysisSchema
from backend.core.knowledge_base import AgencyKnowledgeBase, FaqRagIndex
from backend.core.llm import (
    LangChainLLMAdapter,
    TokenUsage,
    call_with_retry,
    create_embeddings,
)
from tests.test_validators_and_tools import DeterministicHashEmbeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeLLMAdapter:
    """Lightweight test adapter at the LLMAdapter seam (no LangChain subclassing)."""

    def __init__(
        self,
        structured_responses: list[TurnAnalysisSchema | Exception] | None = None,
        text_responses: list[str | Exception] | None = None,
    ) -> None:
        self._structured_responses = list(structured_responses or [])
        self._text_responses = list(text_responses or [])
        self._s_idx = 0
        self._t_idx = 0

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, TokenUsage]:
        if self._s_idx >= len(self._structured_responses):
            raise RuntimeError("No more scripted structured responses")
        item = self._structured_responses[self._s_idx]
        self._s_idx += 1
        if isinstance(item, Exception):
            raise item
        return item, TokenUsage(prompt_tokens=80, completion_tokens=25, total_tokens=105)  # type: ignore[return-value]

    def generate_text(self, system_prompt: str, user_prompt: str) -> tuple[str, TokenUsage]:
        if self._t_idx < len(self._text_responses):
            item = self._text_responses[self._t_idx]
            self._t_idx += 1
            if isinstance(item, Exception):
                raise item
            text = item
        else:
            text = "Verified answer from agency records."
        return text, TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70)


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
    assert "property_lookup" in res.trace.tools_used


def test_mixed_turn_answers_step_and_looks_up_named_property() -> None:
    bot = _make_offline_bot()
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    bot.chat(sid, "Noah")
    bot.chat(sid, "buy")
    bot.chat(sid, "no")
    bot.chat(sid, "new-build")
    assert bot.get_state(sid).current_step == Step.A1_ASK_BUDGET

    # Mixed turn: supplies £2m budget AND asks about Rosemount House without an AR-xxxx ID
    res = bot.chat(sid, "My budget is £2m, and how many bathrooms does Rosemount House have?")
    assert "property_lookup" in res.trace.tools_used
    assert "rosemount house" in res.reply.lower()
    assert res.state.budget_gbp == 2_000_000
    assert res.state.current_step == Step.A3_ASK_POSTCODE
    assert "postcode" in res.reply.lower()


def test_multi_slot_seller_turn_records_seller_coverage_tool() -> None:
    bot = _make_offline_bot()
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    bot.chat(sid, "Clara")
    res = bot.chat(sid, "I want to sell my property in GU5 7AQ")
    assert res.state.postcode == "GU5 7AQ"
    assert res.state.current_step == Step.E1_ASK_PHONE
    assert "seller_coverage_check" in res.trace.tools_used


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


def test_llm_adapter_structured_output_fallback_and_retry() -> None:
    adapter = FakeLLMAdapter(
        structured_responses=[
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
    bot = AshcombeChatbot(settings=settings, llm_adapter=adapter, kb=kb, use_llm=True)

    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)

    # Turn 1: structured NLU extracts visitor name ("Owen")
    r1 = bot.chat(sid, "Hi I'm Owen")
    assert r1.state.visitor_name == "Owen"
    assert r1.state.current_step == Step.BUY_OR_SELL
    assert r1.trace.total_tokens == 105

    # Turn 2: FAQ question -> adapter synthesizes grounded answer + resumes BUY_OR_SELL
    r2 = bot.chat(sid, "What are your selling fees?")
    assert "1.5%" in r2.reply
    assert r2.state.current_step == Step.BUY_OR_SELL
    assert r2.trace.total_tokens == 175  # 105 NLU + 70 synthesis

    # Turn 3: permanent adapter error -> graceful deterministic heuristic fallback still works!
    r3 = bot.chat(sid, "I want to sell")
    assert r3.state.intent == "sell"
    assert r3.state.current_step == Step.D1_ASK_POSTCODE


def test_langchain_adapter_retries_transient_error_and_extracts_tokens() -> None:
    class _StubRunnable:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages: list[object]) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 rate limit exceeded")
            return {
                "raw": AIMessage(
                    content="{}",
                    usage_metadata={"input_tokens": 40, "output_tokens": 12, "total_tokens": 52},
                ),
                "parsed": TurnAnalysisSchema(message_kind="flow_answer", visitor_name="Owen"),
                "parsing_error": None,
            }

    class _StubChatModel:
        def with_structured_output(self, _schema: object, include_raw: bool = False) -> _StubRunnable:
            assert include_raw is True
            return _StubRunnable()

        def invoke(self, _messages: list[object]) -> AIMessage:
            return AIMessage(
                content="Ground truth answer.",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    adapter = LangChainLLMAdapter(_StubChatModel())  # type: ignore[arg-type]
    parsed, usage = adapter.complete_structured("sys", "usr", TurnAnalysisSchema)
    assert parsed.visitor_name == "Owen"
    assert usage == TokenUsage(prompt_tokens=40, completion_tokens=12, total_tokens=52)

    text, text_usage = adapter.generate_text("sys", "usr")
    assert text == "Ground truth answer."
    assert text_usage == TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


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


def test_cli_in_process() -> None:
    bot = _make_offline_bot()
    inputs = iter(["Toby", "sell", "GU5 7AQ", "07700 900555", "toby@example.com", "no", "quit"])
    outputs: list[str] = []
    run_cli(bot, show_trace=True, input_fn=lambda _prompt: next(inputs), output_fn=outputs.append)
    joined = "\n".join(outputs)
    assert "Ashcombe & Rowe Estates" in joined
    assert "GU5 7AQ" in joined
    assert "Thank you for chatting with us. Goodbye." in joined
    assert "[trace]" in joined


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
        [sys.executable, "-m", "backend.cli", "--offline", "--trace"],
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


def test_create_embeddings_gemini_defaults_and_threshold() -> None:
    s1 = Settings(data_dir=DATA_DIR, embedding_provider="gemini", embedding_api_key="dummy-key")
    emb1 = create_embeddings(s1)
    assert emb1.model == "gemini-embedding-001"  # type: ignore[attr-defined]

    # Legacy/local default leaked into embedding_model should still resolve to Gemini default
    s2 = Settings(
        data_dir=DATA_DIR,
        embedding_provider="gemini",
        embedding_model="all-MiniLM-L6-v2",
        embedding_api_key="dummy-key",
    )
    emb2 = create_embeddings(s2)
    assert emb2.model == "gemini-embedding-001"  # type: ignore[attr-defined]

    # Calibrated L2 thresholds per embedding type (unit check without live API calls)
    local_rag = FaqRagIndex.from_csv(DATA_DIR, DeterministicHashEmbeddings())
    assert local_rag.default_max_l2_distance == 1.05


def test_faq_rag_index_caches_embeddings_locally(tmp_path: Path) -> None:
    import shutil

    temp_data = tmp_path / "data"
    temp_data.mkdir()
    shutil.copy2(DATA_DIR / "faq.csv", temp_data / "faq.csv")

    class _SpyEmbeddings(DeterministicHashEmbeddings):
        model = "spy-embedding-model"

        def __init__(self) -> None:
            self.doc_calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.doc_calls += 1
            return super().embed_documents(texts)

    spy1 = _SpyEmbeddings()
    FaqRagIndex.from_csv(temp_data, spy1)
    assert spy1.doc_calls == 1

    # Second load with same CSV + model must hit local JSON cache (0 embed_documents calls)
    spy2 = _SpyEmbeddings()
    rag2 = FaqRagIndex.from_csv(temp_data, spy2)
    assert spy2.doc_calls == 0
    hits = rag2.retrieve("Chelsea branch opening hours", k=3, max_l2_distance=2.0)
    assert any(e.faq_id == "FAQ-023" for e, _ in hits)


def _has_gemini_embedding_config() -> bool:
    get_settings.cache_clear()
    cfg = get_settings()
    return cfg.embedding_provider.strip().lower() in {"gemini", "google"} and bool(
        cfg.embedding_api_key.strip() or cfg.llm_api_key.strip()
    )


@pytest.mark.skipif(
    not _has_gemini_embedding_config(),
    reason="Requires EMBEDDING_PROVIDER=gemini and EMBEDDING_API_KEY in .env",
)
def test_gemini_embeddings_e2e_faq_and_out_of_domain() -> None:
    get_settings.cache_clear()
    cfg = get_settings()
    bot = AshcombeChatbot(settings=cfg, use_llm=False)
    assert bot.kb.faq_rag.default_max_l2_distance == 0.65

    sid = f"gemini-e2e-{uuid.uuid4().hex[:8]}"
    bot.start_session(sid)
    bot.chat(sid, "Elena")

    # In-domain FAQ matches should retrieve accurate answers and preserve flow step
    r_hours = bot.chat(sid, "What are your Chelsea branch opening hours?")
    assert "11:00 am" in r_hours.reply or "9:00 am" in r_hours.reply
    assert "faq_rag" in r_hours.trace.tools_used
    assert r_hours.state.current_step == Step.BUY_OR_SELL

    r_fees = bot.chat(sid, "How much commission do you charge sellers?")
    assert "1.5%" in r_fees.reply
    assert r_fees.state.current_step == Step.BUY_OR_SELL

    r_mayfair = bot.chat(sid, "Where is your Mayfair office located?")
    assert "22 Halden Mews" in r_mayfair.reply
    assert r_mayfair.state.current_step == Step.BUY_OR_SELL

    # Out-of-domain queries must be rejected by the calibrated Gemini L2 threshold
    for ood_query in (
        "Do you offer helicopter tours for viewings?",
        "Do you handle commercial properties or offices?",
    ):
        r_ood = bot.chat(sid, ood_query)
        assert "020 7946 0100" in r_ood.reply
        assert "faq_rag" in r_ood.trace.tools_used
        assert r_ood.state.current_step == Step.BUY_OR_SELL

    assert bot.kb.faq_rag.retrieve("Can I buy a house using Bitcoin?", k=3) == []
