"""Provider-agnostic LLM and embedding factory with retry/rate-limit resilience."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

_DEFAULT_BASE_URLS = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Lightweight LangChain-compatible wrapper around sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Create a LangChain chat model configured from environment variables."""
    cfg = settings or get_settings()
    provider = cfg.llm_provider.strip().lower()
    api_key = cfg.llm_api_key.strip()
    model = cfg.llm_model.strip()

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=groq.")
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=cfg.llm_temperature,
            timeout=cfg.llm_timeout_seconds,
            max_retries=2,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai.")
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": cfg.llm_temperature,
            "timeout": cfg.llm_timeout_seconds,
            "max_retries": 2,
        }
        if cfg.llm_base_url.strip():
            kwargs["base_url"] = cfg.llm_base_url.strip()
        return ChatOpenAI(**kwargs)

    if provider in {"gemini", "google"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=gemini.")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=cfg.llm_temperature,
            max_retries=2,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": model, "temperature": cfg.llm_temperature}
        if cfg.llm_base_url.strip():
            kwargs["base_url"] = cfg.llm_base_url.strip()
        return ChatOllama(**kwargs)

    if provider in {"nvidia", "openrouter", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        base_url = cfg.llm_base_url.strip() or _DEFAULT_BASE_URLS.get(provider, "")
        if not base_url:
            raise ValueError(f"LLM_BASE_URL is required when LLM_PROVIDER={provider}.")
        if not api_key:
            raise ValueError(f"LLM_API_KEY is required when LLM_PROVIDER={provider}.")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=cfg.llm_temperature,
            timeout=cfg.llm_timeout_seconds,
            max_retries=2,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{cfg.llm_provider}'. "
        "Supported: groq, openai, gemini, nvidia, openrouter, ollama, openai_compatible."
    )


def create_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create an embeddings instance configured from environment variables."""
    cfg = settings or get_settings()
    provider = cfg.embedding_provider.strip().lower()
    model = cfg.embedding_model.strip()
    api_key = cfg.embedding_api_key.strip() or cfg.llm_api_key.strip()

    if provider in {"local", "huggingface", "sentence-transformers"}:
        return LocalSentenceTransformerEmbeddings(model_name=model or "all-MiniLM-L6-v2")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not api_key:
            raise ValueError("EMBEDDING_API_KEY (or LLM_API_KEY) is required for OpenAI embeddings.")
        return OpenAIEmbeddings(model=model or "text-embedding-3-small", api_key=api_key)

    if provider in {"gemini", "google"}:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not api_key:
            raise ValueError("EMBEDDING_API_KEY (or LLM_API_KEY) is required for Gemini embeddings.")
        return GoogleGenerativeAIEmbeddings(model=model or "models/text-embedding-004", google_api_key=api_key)

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{cfg.embedding_provider}'. "
        "Supported: local, openai, gemini."
    )


def call_with_retry(fn: Callable[[], T], max_attempts: int = 3, base_delay: float = 0.8) -> T:
    """Execute an LLM call with exponential backoff on rate limits or transient network errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_transient = any(
                token in msg
                for token in ("rate limit", "429", "timeout", "temporarily unavailable", "connection", "503")
            )
            if not is_transient or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Transient LLM error (attempt %d/%d): %s; retrying in %.1fs", attempt, max_attempts, exc, delay)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
