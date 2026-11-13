# 0002: Narrow LLM Adapter Seam at the Conversation Engine Edge

Status: accepted

## Context

`src/nlu.py` previously depended directly on LangChain's `BaseChatModel`, calling `llm.with_structured_output(TurnAnalysisSchema, include_raw=True)` and `invoke(...)` inline. Because of that coupling, offline LLM-path tests in `tests/test_e2e_comprehensive.py` had to build a 59-line `ScriptedMockChatModel(BaseChatModel)` subclass reproducing LangChain internals (`_generate`, `ChatResult`, `RunnableLambda`).

## Decision

Place a narrow **LLM adapter** seam at the Conversation Engine's edge. The production LangChain adapter owns `BaseChatModel` invocation, structured output parsing, exponential-backoff retries, and token-usage extraction, while tests supply a lightweight fake adapter (alongside the existing in-process deterministic fallback).

## Considered Options

- **Keep passing `BaseChatModel` directly into `nlu.py`**: Rejected because every test of LLM-driven turn routing must mock LangChain's `Runnable` and `with_structured_output` contract.

## Consequences

- `src/nlu.py` and the Conversation Engine no longer import or depend on LangChain `BaseChatModel` internals or token-metadata extraction quirks.
- Production and test adapters satisfy the same seam, making both halves of "one adapter = hypothetical seam, two = real" true.
