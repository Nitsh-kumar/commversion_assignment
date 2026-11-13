"""Generate verified sample transcripts covering Lanes A, B, C, D and mid-flow FAQ."""

from __future__ import annotations

import argparse
from pathlib import Path
import uuid

from src.config import get_settings
from src.graph import AshcombeChatbot
from src.llm_factory import create_embeddings
from src.tools import AgencyKnowledgeBase

TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "transcripts"


def _record_session(bot: AshcombeChatbot, title: str, filename: str, turns: list[str]) -> None:
    sid = f"transcript-{uuid.uuid4().hex[:8]}"
    greeting = bot.start_session(sid)
    lines = [
        f"# {title}",
        "",
        f"**Assistant:** {greeting}",
        "",
    ]
    for user_msg in turns:
        res = bot.chat(sid, user_msg)
        lines.append(f"**Visitor:** {user_msg}")
        lines.append("")
        lines.append(f"**Assistant:** {res.reply}")
        lines.append("")
        lines.append(
            f"> _Step: `{res.trace.step_before}` -> `{res.trace.step_after}` | "
            f"Tools: `{res.trace.tools_used}` | Latency: `{res.trace.latency_ms} ms`_"
        )
        lines.append("")

    final_state = bot.get_state(sid)
    lines.extend(
        [
            "---",
            "## Final Qualification State",
            f"- `qualified`: `{final_state.qualified}`",
            f"- `closing_code`: `{final_state.closing_code}`",
            f"- `visitor_name`: `{final_state.visitor_name}`",
            f"- `intent`: `{final_state.intent}`",
            f"- `listing_category`: `{final_state.listing_category}`",
            f"- `budget_gbp`: `{final_state.budget_gbp}`",
            f"- `postcode`: `{final_state.postcode}`",
            f"- `phone`: `{final_state.phone}`",
            f"- `email`: `{final_state.email}`",
        ]
    )
    (TRANSCRIPT_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample conversation transcripts")
    parser.add_argument("--llm", action="store_true", help="Use live configured LLM instead of offline mode")
    args = parser.parse_args()

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    if args.llm:
        bot = AshcombeChatbot(settings=settings, use_llm=True)
    else:
        embeddings = create_embeddings(settings)
        kb = AgencyKnowledgeBase.from_data_dir(settings.data_dir, embeddings)
        bot = AshcombeChatbot(settings=settings, kb=kb, use_llm=False)

    _record_session(
        bot,
        "Lane A — New-Build Buyer (Qualified) + Mid-Flow FAQ",
        "lane_a_new_build_with_faq.md",
        [
            "Alice",
            "I am looking to buy a home",
            "No",
            "New-build",
            "What are your Chelsea branch opening hours?",
            "£1.5m",
            "sw3 3rb",
            "07700 900123",
            "alice@example.com",
            "No, thank you",
        ],
    )
    _record_session(
        bot,
        "Lane B — Resale Buyer (Qualified under £1M)",
        "lane_b_resale_qualified.md",
        [
            "Elena",
            "I want to buy",
            "No",
            "Resale",
            "£600k",
            "SW10 9RR",
            "07700 900456",
            "elena@example.com",
            "No",
        ],
    )
    _record_session(
        bot,
        "Lane C — Specific Property (Sold) -> Re-qualified via Lane A",
        "lane_c_specific_property_sold.md",
        [
            "Henry",
            "I want to buy",
            "Yes",
            "Townhouse 2, Riverside Quarter",
            "£2m",
            "SW3 3RB",
            "07700 900111",
            "henry@example.com",
            "No",
        ],
    )
    _record_session(
        bot,
        "Lane D — Seller (Covered vs Reassist) & Uncovered (M2)",
        "lane_d_seller.md",
        [
            "Ivy",
            "I want to sell my house",
            "gu5 7aq",
            "07700 900222",
            "ivy@example.com",
            "No",
        ],
    )
    print(f"Generated 4 sample transcripts in {TRANSCRIPT_DIR}")


if __name__ == "__main__":
    main()
