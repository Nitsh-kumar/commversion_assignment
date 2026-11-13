# 0001: Single-Turn LangGraph Conversation Engine with Integrated Telemetry

Status: accepted

## Context

Turn execution was previously split across four LangGraph `StateGraph` nodes (`analyze_turn`, `handle_interrupt`, `advance_flow`, `compose_response`) in `src/graph.py`, communicating through a `GraphState` dictionary that serialized `ConversationData` and `TurnAnalysisSchema` on every hop. Meanwhile, `src/tracing.py` forced `(prompt_tokens, completion_tokens, total_tokens)` tuple plumbing through `src/nlu.py` and forced `src/graph.py` to guess `tools_used` from `step_before` before `advance_conversation` ran.

## Decision

Collapse turn analysis, side-Q&A handling, deterministic flow transitions, response composition, and turn telemetry into a single Conversation Engine turn implementation, keeping LangGraph (`StateGraph` + `MemorySaver`) hidden inside the engine solely for session checkpointing.

## Considered Options

- **Keep the 4-node `StateGraph`**: Rejected because `GraphState` dict serialization across nodes leaked `Step` checks into `graph.py` and caused missed `tools_used` telemetry on multi-slot turns.
- **Remove LangGraph completely**: Rejected to preserve LangGraph `StateGraph` + `MemorySaver` session checkpointing compatibility documented in `README.md`.

## Consequences

- `src/tracing.py` is absorbed directly into the engine and LLM adapter; `tools_used` is recorded where knowledge-base lookups actually occur.
- Internal turn stages can change without modifying `GraphState` schemas or breaking `AshcombeChatbot` callers.
