"""Interactive CLI runner for the Ashcombe & Rowe Estates lead qualification chatbot."""

from __future__ import annotations

import argparse
import uuid

from src.flow import Step
from src.graph import AshcombeChatbot


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
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    print("=" * 64)
    print("Ashcombe & Rowe Estates — Lead Qualification Assistant")
    print("Type 'quit' to exit or 'reset' to start a new conversation.")
    print("=" * 64)

    greeting = bot.start_session(session_id)
    print(f"\nAssistant: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            session_id = f"cli-{uuid.uuid4().hex[:8]}"
            greeting = bot.start_session(session_id)
            print(f"\n[Session reset]\nAssistant: {greeting}\n")
            continue

        result = bot.chat(session_id, user_input)
        print(f"\nAssistant: {result.reply}\n")
        if args.trace:
            t = result.trace
            print(
                f"  [trace] {t.step_before} -> {t.step_after} | "
                f"{t.latency_ms}ms | tokens={t.total_tokens} | tools={t.tools_used}"
            )
        if result.state.current_step == Step.ENDED:
            print("[Conversation completed. Type 'reset' for a new visitor or 'quit' to exit.]\n")


if __name__ == "__main__":
    main()
