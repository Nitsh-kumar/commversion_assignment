"""Lightweight per-turn observability: latency, token usage, and estimated cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Approximate blended USD cost per 1M tokens for common dev/eval models (informational only)
_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


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


def extract_token_usage(response: Any) -> tuple[int, int, int]:
    """Extract (prompt_tokens, completion_tokens, total_tokens) from a LangChain AIMessage."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        p = int(usage.get("input_tokens", 0) or 0)
        c = int(usage.get("output_tokens", 0) or 0)
        t = int(usage.get("total_tokens", p + c) or 0)
        return p, c, t

    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(token_usage, dict):
        p = int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0)
        c = int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0)
        t = int(token_usage.get("total_tokens", p + c) or 0)
        return p, c, t

    return 0, 0, 0


def estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single call given model name and token counts."""
    lowered = model_name.lower()
    for key, (in_rate, out_rate) in _MODEL_PRICING_PER_1M.items():
        if key in lowered:
            return round((prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000, 6)
    return 0.0
