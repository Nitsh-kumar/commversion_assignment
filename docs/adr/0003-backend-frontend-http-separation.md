# 0003: Strict Backend (FastAPI) and Frontend (HTTP-Only Streamlit) Separation

Status: accepted

## Context

Previously, the FastAPI endpoints, Streamlit UI, CLI runner, and core domain modules all lived under a single `src/` tree, and the Streamlit UI imported `AshcombeChatbot` directly to run in-process when `CHATBOT_API_URL` was unset. That blurred the boundary between frontend presentation and backend FastAPI orchestration.

## Decision

Split the codebase into top-level `backend/` (FastAPI server, Conversation Engine, Knowledge Base, CLI, and scripts) and `frontend/` (Streamlit UI + `ChatbotApiClient` over HTTP). Remove the frontend's in-process engine fallback so `frontend/` has zero imports from `backend/` and always communicates with FastAPI over HTTP (`BACKEND_API_URL`, defaulting to `http://localhost:8000`).

## Considered Options

- **Keep an in-process fallback inside Streamlit**: Rejected because it couples the frontend runtime to LangGraph, FAISS, and SentenceTransformers and hides whether FastAPI is actually working.

## Consequences

- Running the UI requires the FastAPI backend (`uvicorn backend.main:app --host 0.0.0.0 --port 8000`) to be running first, with CORS enabled on FastAPI.
- An AST import check in `tests/test_frontend_client.py` enforces zero `backend.*` imports inside `frontend/`.
