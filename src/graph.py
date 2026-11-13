"""LangGraph orchestration for Ashcombe & Rowe Estates lead qualification chatbot."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.config import Settings, get_settings
from src.flow import ConversationData, Step, advance_conversation, prompt_for_step
from src.llm_factory import create_chat_model, create_embeddings
from src.nlu import (
    TurnAnalysisSchema,
    analyze_turn_with_llm,
    answer_side_question,
    to_extracted_turn,
)
from src.tools import AgencyKnowledgeBase
from src.tracing import TurnTrace, estimate_cost_usd


class GraphState(TypedDict, total=False):
    """LangGraph conversation state persisted per session via MemorySaver."""

    messages: Annotated[list[AnyMessage], add_messages]
    flow_data: dict[str, Any]
    last_user_message: str
    last_analysis: dict[str, Any]
    side_answer: str | None
    flow_reply: str | None
    tools_used: list[str]
    step_before: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    turn_started_at: float
    traces: list[dict[str, Any]]


@dataclass
class ChatTurnResult:
    """Structured result returned to CLI, API, tests, and eval harness."""

    reply: str
    state: ConversationData
    trace: TurnTrace


_QUESTION_WORDS = ("?", "what", "how", "when", "where", "who", "which", "does", "do ", "is ", "are ", "can ")


class AshcombeChatbot:
    """Stateful LangGraph-based lead qualification chatbot."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        kb: AgencyKnowledgeBase | None = None,
        use_llm: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        if kb is not None:
            self.kb = kb
        else:
            embeddings = create_embeddings(self.settings)
            self.kb = AgencyKnowledgeBase.from_data_dir(self.settings.data_dir, embeddings)

        if llm is not None:
            self.llm: BaseChatModel | None = llm
        elif use_llm:
            self.llm = create_chat_model(self.settings)
        else:
            self.llm = None

        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("analyze_turn", self._node_analyze_turn)
        builder.add_node("handle_interrupt", self._node_handle_interrupt)
        builder.add_node("advance_flow", self._node_advance_flow)
        builder.add_node("compose_response", self._node_compose_response)

        builder.add_edge(START, "analyze_turn")
        builder.add_conditional_edges(
            "analyze_turn",
            self._route_after_analysis,
            {
                "interrupt": "handle_interrupt",
                "flow": "advance_flow",
                "compose": "compose_response",
            },
        )
        builder.add_conditional_edges(
            "handle_interrupt",
            self._route_after_interrupt,
            {
                "flow": "advance_flow",
                "compose": "compose_response",
            },
        )
        builder.add_edge("advance_flow", "compose_response")
        builder.add_edge("compose_response", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _node_analyze_turn(self, state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        data = ConversationData.from_dict(state.get("flow_data"))
        user_msg = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_msg = str(msg.content)
                break

        if data.current_step == Step.ENDED:
            return {
                "last_user_message": user_msg,
                "last_analysis": TurnAnalysisSchema(message_kind="flow_answer").model_dump(),
                "side_answer": None,
                "flow_reply": None,
                "tools_used": [],
                "step_before": data.current_step.value,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "turn_started_at": started,
            }

        analysis, (p_tok, c_tok, t_tok) = analyze_turn_with_llm(
            self.llm, user_msg, data, catalog=self.kb.catalog
        )

        # Safeguard: if user is answering C1 with a plain property name, route to flow rather than side-Q&A
        if (
            data.current_step in {Step.C1_ASK_PROPERTY, Step.C1_RETRY_PROPERTY, Step.C1_DISAMBIGUATE}
            and analysis.message_kind == "property_question"
            and not any(q in user_msg.lower() for q in _QUESTION_WORDS)
        ):
            analysis.message_kind = "flow_answer"
            if not analysis.specific_property_query:
                analysis.specific_property_query = user_msg.strip()

        return {
            "last_user_message": user_msg,
            "last_analysis": analysis.model_dump(),
            "side_answer": None,
            "flow_reply": None,
            "tools_used": [],
            "step_before": data.current_step.value,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "turn_started_at": started,
        }

    @staticmethod
    def _route_after_analysis(state: GraphState) -> str:
        data = ConversationData.from_dict(state.get("flow_data"))
        if data.current_step == Step.ENDED:
            return "flow"
        kind = state.get("last_analysis", {}).get("message_kind", "flow_answer")
        if kind in {
            "faq_question",
            "property_question",
            "mixed_answer_and_question",
            "off_topic",
            "rule_override",
        }:
            return "interrupt"
        return "flow"

    def _node_handle_interrupt(self, state: GraphState) -> dict[str, Any]:
        analysis = TurnAnalysisSchema(**state["last_analysis"])
        user_msg = state.get("last_user_message", "")
        p_tok = int(state.get("prompt_tokens", 0))
        c_tok = int(state.get("completion_tokens", 0))
        t_tok = int(state.get("total_tokens", 0))
        tools_used = list(state.get("tools_used", []))

        if analysis.message_kind == "rule_override":
            side_answer = (
                "I'm sorry, I can't override Ashcombe & Rowe Estates' rules—including our £1 million "
                "minimum for new-build properties, postcode coverage limits, and listing availability. "
                f"If you'd like to discuss exceptions with our team, please call our head office on {self.settings.office_phone}."
            )
            return {"side_answer": side_answer, "tools_used": tools_used}

        if analysis.message_kind == "off_topic":
            side_answer = (
                "I'm here to help with buying or selling properties and answering questions about "
                "Ashcombe & Rowe Estates, so I can't help with unrelated requests."
            )
            return {"side_answer": side_answer, "tools_used": tools_used}

        question = analysis.side_question_text or user_msg
        side_answer, added_tools, (sp, sc, st) = answer_side_question(
            question=question,
            analysis=analysis,
            kb=self.kb,
            settings=self.settings,
            llm=self.llm,
        )
        tools_used.extend(added_tools)
        return {
            "side_answer": side_answer,
            "tools_used": tools_used,
            "prompt_tokens": p_tok + sp,
            "completion_tokens": c_tok + sc,
            "total_tokens": t_tok + st,
        }

    @staticmethod
    def _route_after_interrupt(state: GraphState) -> str:
        kind = state.get("last_analysis", {}).get("message_kind", "flow_answer")
        if kind == "mixed_answer_and_question":
            return "flow"
        return "compose"

    def _node_advance_flow(self, state: GraphState) -> dict[str, Any]:
        data = ConversationData.from_dict(state.get("flow_data"))
        analysis = TurnAnalysisSchema(**state["last_analysis"])
        extracted = to_extracted_turn(analysis)
        user_msg = state.get("last_user_message", "")
        tools_used = list(state.get("tools_used", []))

        # Track deterministic data tools invoked by this step
        if data.current_step in {Step.A3_ASK_POSTCODE, Step.B2_ASK_POSTCODE}:
            tools_used.append("properties_search")
        elif data.current_step in {Step.C1_ASK_PROPERTY, Step.C1_RETRY_PROPERTY, Step.C1_DISAMBIGUATE}:
            tools_used.append("property_lookup")
        elif data.current_step == Step.D1_ASK_POSTCODE:
            tools_used.append("seller_coverage_check")

        reply = advance_conversation(
            data=data,
            extracted=extracted,
            raw_user_message=user_msg,
            catalog=self.kb.catalog,
            seller_coverage=self.kb.seller_coverage,
            settings=self.settings,
        )
        return {
            "flow_data": data.to_dict(),
            "flow_reply": reply,
            "tools_used": tools_used,
        }

    def _node_compose_response(self, state: GraphState) -> dict[str, Any]:
        data = ConversationData.from_dict(state.get("flow_data"))
        side_answer = state.get("side_answer")
        flow_reply = state.get("flow_reply")

        if side_answer and flow_reply:
            full_reply = f"{side_answer}\n\nTo continue where we left off — {flow_reply}"
        elif side_answer:
            resume_prompt = prompt_for_step(data.current_step, data)
            full_reply = f"{side_answer}\n\nTo continue where we left off — {resume_prompt}"
        else:
            full_reply = flow_reply or prompt_for_step(data.current_step, data)

        elapsed_ms = round((time.perf_counter() - state.get("turn_started_at", time.perf_counter())) * 1000, 2)
        p_tok = int(state.get("prompt_tokens", 0))
        c_tok = int(state.get("completion_tokens", 0))
        t_tok = int(state.get("total_tokens", 0))
        traces = list(state.get("traces", []))
        trace_dict = {
            "turn_index": len(traces) + 1,
            "step_before": state.get("step_before", data.current_step.value),
            "step_after": data.current_step.value,
            "latency_ms": elapsed_ms,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "estimated_cost_usd": estimate_cost_usd(self.settings.llm_model, p_tok, c_tok),
            "tools_used": list(dict.fromkeys(state.get("tools_used", []))),
        }
        traces.append(trace_dict)

        return {
            "messages": [AIMessage(content=full_reply)],
            "flow_data": data.to_dict(),
            "traces": traces,
        }

    def start_session(self, session_id: str) -> str:
        """Initialize a new conversation thread and return the opening greeting."""
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

    def chat(self, session_id: str, user_message: str) -> ChatTurnResult:
        """Process one user message for `session_id` and return reply, state, and trace."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot or not snapshot.values or "flow_data" not in snapshot.values:
            self.start_session(session_id)

        out = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        reply = str(out["messages"][-1].content)
        data = ConversationData.from_dict(out.get("flow_data"))
        last_trace_dict = out["traces"][-1]
        trace = TurnTrace(**last_trace_dict)
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
