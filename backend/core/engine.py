"""Single-turn LangGraph Conversation Engine with integrated telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import threading
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from backend.core.config import Settings, get_settings
from backend.core.flow import (
    ConversationData,
    Step,
    TurnAnalysisSchema,
    advance_conversation,
    prompt_for_step,
)
from backend.core.llm import (
    LLMAdapter,
    LangChainLLMAdapter,
    TokenUsage,
    create_chat_model,
    create_embeddings,
)
from backend.core.nlu import analyze_turn_with_llm, answer_side_question
from backend.core.knowledge_base import AgencyKnowledgeBase

# Approximate blended USD cost per 1M tokens for common dev/eval models (informational only)
_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}

_INTERRUPT_KINDS = {
    "faq_question",
    "property_question",
    "mixed_answer_and_question",
    "off_topic",
    "rule_override",
}
_FLOW_ADVANCING_KINDS = {"flow_answer", "mind_change", "mixed_answer_and_question"}


@dataclass
class TurnTrace:
    """Telemetry captured for a single conversational turn."""

    turn_index: int
    step_before: str
    step_after: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tools_used: list[str] = field(default_factory=list)


def estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single call given model name and token counts."""
    lowered = model_name.lower()
    for key, (in_rate, out_rate) in _MODEL_PRICING_PER_1M.items():
        if key in lowered:
            return round((prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000, 6)
    return 0.0


class GraphState(TypedDict, total=False):
    """LangGraph session state persisted per thread via MemorySaver."""

    messages: Annotated[list[AnyMessage], add_messages]
    flow_data: dict[str, Any]
    traces: list[dict[str, Any]]


@dataclass
class ChatTurnResult:
    """Structured result returned to CLI, API, tests, and eval harness."""

    reply: str
    state: ConversationData
    trace: TurnTrace


class AshcombeChatbot:
    """Deep Conversation Engine backed by single-turn LangGraph checkpointing."""

    def __init__(
        self,
        settings: Settings | None = None,
        kb: AgencyKnowledgeBase | None = None,
        use_llm: bool = True,
        llm_adapter: LLMAdapter | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if kb is not None:
            self.kb = kb
        else:
            embeddings = create_embeddings(self.settings)
            self.kb = AgencyKnowledgeBase.from_data_dir(
                self.settings.data_dir,
                embeddings,
                faq_max_l2_distance=self.settings.faq_max_l2_distance,
            )

        if llm_adapter is not None:
            self.llm_adapter: LLMAdapter | None = llm_adapter
        elif use_llm:
            self.llm_adapter = LangChainLLMAdapter(create_chat_model(self.settings))
        else:
            self.llm_adapter = None

        self.checkpointer = checkpointer or MemorySaver()
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self.graph = self._build_graph()

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("process_turn", self._node_process_turn)
        builder.add_edge(START, "process_turn")
        builder.add_edge("process_turn", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _resolve_interrupt(
        self,
        analysis: TurnAnalysisSchema,
        user_msg: str,
    ) -> tuple[str | None, list[str], TokenUsage]:
        """Handle rule overrides, off-topic messages, and FAQ/property side questions."""
        if analysis.message_kind not in _INTERRUPT_KINDS:
            return None, [], TokenUsage()

        if analysis.message_kind == "rule_override":
            return (
                "I'm sorry, I can't override Ashcombe & Rowe Estates' rules—including our £1 million "
                "minimum for new-build properties, postcode coverage limits, and listing availability. "
                f"If you'd like to discuss exceptions with our team, please call our head office on {self.settings.office_phone}."
            ), [], TokenUsage()

        if analysis.message_kind == "off_topic":
            return (
                "I'm here to help with buying or selling properties and answering questions about "
                "Ashcombe & Rowe Estates, so I can't help with unrelated requests."
            ), [], TokenUsage()

        question = analysis.side_question_text or user_msg
        return answer_side_question(
            question=question,
            analysis=analysis,
            kb=self.kb,
            settings=self.settings,
            llm=self.llm_adapter,
        )

    def _execute_turn(
        self,
        data: ConversationData,
        user_msg: str,
    ) -> tuple[str, list[str], TokenUsage]:
        """Run NLU, interrupt handling, flow advancement, and reply composition for one turn."""
        if data.current_step == Step.ENDED:
            return (
                "This conversation has ended. Please start a new chat if you need further assistance.",
                [],
                TokenUsage(),
            )

        analysis, nlu_usage = analyze_turn_with_llm(
            self.llm_adapter,
            user_msg,
            data,
            catalog=self.kb.catalog,
        )

        side_answer, interrupt_tools, side_usage = self._resolve_interrupt(analysis, user_msg)
        total_usage = nlu_usage + side_usage
        tools_used = list(interrupt_tools)

        flow_reply: str | None = None
        if analysis.message_kind in _FLOW_ADVANCING_KINDS:
            flow_result = advance_conversation(
                data=data,
                extracted=analysis,
                raw_user_message=user_msg,
                catalog=self.kb.catalog,
                seller_coverage=self.kb.seller_coverage,
                settings=self.settings,
            )
            flow_reply = flow_result.reply
            tools_used.extend(flow_result.tools_used)

        if side_answer and flow_reply:
            if data.current_step == Step.ENDED:
                full_reply = f"{side_answer}\n\n{flow_reply}"
            else:
                full_reply = f"{side_answer}\n\nTo continue where we left off — {flow_reply}"
        elif side_answer:
            resume_prompt = prompt_for_step(data.current_step, data)
            if data.current_step == Step.ENDED:
                full_reply = f"{side_answer}\n\n{resume_prompt}"
            else:
                full_reply = f"{side_answer}\n\nTo continue where we left off — {resume_prompt}"
        else:
            full_reply = flow_reply if flow_reply is not None else prompt_for_step(data.current_step, data)

        return full_reply, list(dict.fromkeys(tools_used)), total_usage

    def _node_process_turn(self, state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        data = ConversationData.from_dict(state.get("flow_data"))
        step_before = data.current_step.value

        user_msg = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_msg = str(msg.content)
                break

        full_reply, tools_used, usage = self._execute_turn(data, user_msg)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        traces = list(state.get("traces", []))
        trace = TurnTrace(
            turn_index=len(traces) + 1,
            step_before=step_before,
            step_after=data.current_step.value,
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=estimate_cost_usd(
                self.settings.llm_model,
                usage.prompt_tokens,
                usage.completion_tokens,
            ),
            tools_used=tools_used,
        )
        traces.append(asdict(trace))

        return {
            "messages": [AIMessage(content=full_reply)],
            "flow_data": data.to_dict(),
            "traces": traces,
        }

    def start_session(self, session_id: str) -> str:
        """Initialize a new conversation thread and return the opening greeting."""
        with self._get_session_lock(session_id):
            return self._init_session_unlocked(session_id)

    def _init_session_unlocked(self, session_id: str) -> str:
        initial = ConversationData(current_step=Step.ASK_NAME)
        greeting = f"Hello! Welcome to Ashcombe & Rowe Estates. {prompt_for_step(Step.ASK_NAME, initial)}"
        config = {"configurable": {"thread_id": session_id}}
        self.graph.update_state(
            config,
            {
                "messages": [AIMessage(content=greeting)],
                "flow_data": initial.to_dict(),
                "traces": [],
            },
        )
        return greeting

    def has_session(self, session_id: str) -> bool:
        """Return True if `session_id` has persisted conversation state."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(config)
        return bool(snapshot and snapshot.values and "flow_data" in snapshot.values)

    def chat(self, session_id: str, user_message: str) -> ChatTurnResult:
        """Process one user message for `session_id` and return reply, state, and trace."""
        config = {"configurable": {"thread_id": session_id}}
        with self._get_session_lock(session_id):
            if not self.has_session(session_id):
                self._init_session_unlocked(session_id)

            out = self.graph.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
            )
        reply = str(out["messages"][-1].content)
        data = ConversationData.from_dict(out.get("flow_data"))
        trace = TurnTrace(**out["traces"][-1])
        return ChatTurnResult(reply=reply, state=data, trace=trace)

    def get_state(self, session_id: str) -> ConversationData:
        """Return current ConversationData for a session."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot or not snapshot.values:
            return ConversationData()
        return ConversationData.from_dict(snapshot.values.get("flow_data"))

    def get_traces(self, session_id: str) -> list[TurnTrace]:
        """Return all per-turn telemetry records for a session."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot or not snapshot.values:
            return []
        return [TurnTrace(**item) for item in snapshot.values.get("traces", [])]
