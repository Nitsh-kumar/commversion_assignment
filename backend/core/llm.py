"""Provider-agnostic LLM and embedding adapters with retry/rate-limit resilience."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Protocol, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from backend.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token counts returned across LLM adapter implementations."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


def extract_token_usage(response: Any) -> TokenUsage:
    """Extract normalized token counts from a LangChain AIMessage or metadata payload."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        p = int(usage.get("input_tokens", 0) or 0)
        c = int(usage.get("output_tokens", 0) or 0)
        t = int(usage.get("total_tokens", p + c) or 0)
        return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=t)

    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(token_usage, dict):
        p = int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0)
        c = int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0)
        t = int(token_usage.get("total_tokens", p + c) or 0)
        return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=t)

    return TokenUsage()


class LLMAdapter(Protocol):
    """Narrow LLM seam consumed by the Conversation Engine."""

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, TokenUsage]:
        """Return a validated structured schema instance and token usage."""
        ...

    def generate_text(self, system_prompt: str, user_prompt: str) -> tuple[str, TokenUsage]:
        """Return a synthesized text response and token usage."""
        ...


class LangChainLLMAdapter:
    """Production LLM adapter wrapping LangChain BaseChatModel with retry resilience."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, TokenUsage]:
        structured_llm = self._llm.with_structured_output(schema, include_raw=True)
        raw_out = call_with_retry(
            lambda: structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        )
        parsed = raw_out.get("parsed")
        if raw_out.get("parsing_error") is not None or parsed is None:
            raise ValueError(f"Structured output parsing failed: {raw_out.get('parsing_error')}")
        usage = extract_token_usage(raw_out.get("raw"))
        return parsed, usage

    def generate_text(self, system_prompt: str, user_prompt: str) -> tuple[str, TokenUsage]:
        msg = call_with_retry(
            lambda: self._llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        )
        return str(msg.content).strip(), extract_token_usage(msg)


_DEFAULT_BASE_URLS = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Lightweight LangChain-compatible wrapper around sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
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
        resolved_model = model if model and model != "all-MiniLM-L6-v2" else "text-embedding-3-small"
        return OpenAIEmbeddings(model=resolved_model, api_key=api_key)

    if provider in {"gemini", "google"}:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not api_key:
            raise ValueError("EMBEDDING_API_KEY (or LLM_API_KEY) is required for Gemini embeddings.")
        resolved_model = model if model and model != "all-MiniLM-L6-v2" else "gemini-embedding-001"
        return GoogleGenerativeAIEmbeddings(model=resolved_model, google_api_key=api_key)

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
