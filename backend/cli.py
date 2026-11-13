"""Interactive CLI delivery adapter for the Ashcombe & Rowe Estates Conversation Engine."""

from __future__ import annotations

import argparse
from typing import Callable
import uuid

from backend.core.engine import AshcombeChatbot
from backend.core.flow import Step


def run_cli(
    bot: AshcombeChatbot,
    show_trace: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive CLI loop against an injected AshcombeChatbot."""
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    output_fn("=" * 64)
    output_fn("Ashcombe & Rowe Estates — Lead Qualification Assistant")
    output_fn("Type 'quit' to exit or 'reset' to start a new conversation.")
    output_fn("=" * 64)

    greeting = bot.start_session(session_id)
    output_fn(f"\nAssistant: {greeting}\n")

    while True:
        try:
            user_input = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            output_fn("Goodbye!")
            break
        if user_input.lower() == "reset":
            session_id = f"cli-{uuid.uuid4().hex[:8]}"
            greeting = bot.start_session(session_id)
            output_fn(f"\n[Session reset]\nAssistant: {greeting}\n")
            continue

        result = bot.chat(session_id, user_input)
        output_fn(f"\nAssistant: {result.reply}\n")
        if show_trace:
            t = result.trace
            output_fn(
                f"  [trace] {t.step_before} -> {t.step_after} | "
                f"{t.latency_ms}ms | tokens={t.total_tokens} | tools={t.tools_used}"
            )
        if result.state.current_step == Step.ENDED:
            output_fn("[Conversation completed. Type 'reset' for a new visitor or 'quit' to exit.]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ashcombe & Rowe Estates CLI Chatbot")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run with deterministic heuristic NLU instead of calling an external LLM.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print per-turn step transitions, latency, and token usage.",
    )
    args = parser.parse_args()

    bot = AshcombeChatbot(use_llm=not args.offline)
    run_cli(bot, show_trace=args.trace)


if __name__ == "__main__":
    main()
