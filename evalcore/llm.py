"""Gemini model factories, rate limiting and a persistent response cache.

Three things make an LLM call safe to put inside an evaluation loop:

1. **Determinism.** Temperature is pinned to the configured value (0.0 by
   default) so a rerun of the same suite differs only because of provider-side
   nondeterminism, not because of our sampling.
2. **Rate limiting.** The Gemini free tier is quota limited; a naive
   ``asyncio.gather`` over 500 test cases will trip 429s and silently poison a
   score with error rows. :class:`RateLimiter` enforces a token bucket.
3. **Caching.** Judge calls are pure functions of (model, prompt, params). The
   SQLite cache turns a second run of the same suite into a near-free operation,
   which is what makes per-commit continuous evaluation affordable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from evalcore.config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    """Raised when a live model call is attempted without a Gemini API key."""


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def build_chat_model(
    *,
    role: str = "generation",
    model: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Create a Gemini chat model for a given evaluation role.

    Args:
        role: ``"generation"`` for the system under test, ``"judge"`` for the
            grader. The role selects which configured model and temperature
            apply, which is how the lab keeps generator and judge distinct.
        model: Explicit model override (used by the model-comparison labs).
        temperature: Explicit temperature override.
        max_output_tokens: Cap on response length; judges are capped tightly
            because a rubric verdict should never be an essay.
        settings: Injected settings, mainly for tests.

    Raises:
        MissingAPIKeyError: if no Gemini key is configured.
    """
    settings = settings or get_settings()
    if not settings.has_api_key:
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add a free "
            "key from https://aistudio.google.com/app/apikey"
        )

    if role == "judge":
        resolved_model = model or settings.judge_model
        resolved_temp = settings.judge_temperature if temperature is None else temperature
    else:
        resolved_model = model or settings.generation_model
        resolved_temp = settings.temperature if temperature is None else temperature

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=resolved_temp,
        max_output_tokens=max_output_tokens,
        timeout=settings.request_timeout_s,
        max_retries=0,  # retries are owned by evalcore.runner, which records them
        google_api_key=settings.google_api_key.get_secret_value(),
    )


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create the Gemini embedding model used by the vector store and drift labs."""
    settings = settings or get_settings()
    if not settings.has_api_key:
        raise MissingAPIKeyError("GOOGLE_API_KEY is not set; embeddings require a live key.")
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key.get_secret_value(),
    )


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """An async token bucket that keeps a suite inside the provider's quota.

    A bucket rather than a fixed sleep: bursts up to ``capacity`` are allowed
    (which is what makes short suites fast), and the long-run average is pinned
    to ``rate_per_minute`` (which is what keeps long suites out of 429s).
    """

    def __init__(self, rate_per_minute: int, capacity: int | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._rate_per_second = rate_per_minute / 60.0
        self._capacity = float(capacity if capacity is not None else max(1, rate_per_minute // 2))
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate_per_second
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_s = deficit / self._rate_per_second
            await asyncio.sleep(min(wait_s, 5.0))


# ---------------------------------------------------------------------------
# Persistent response cache
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheKey:
    """The complete set of inputs that determine an LLM response."""

    model: str
    temperature: float
    prompt: str
    extra: str = ""

    def digest(self) -> str:
        blob = json.dumps(
            {
                "model": self.model,
                "temperature": round(self.temperature, 6),
                "prompt": self.prompt,
                "extra": self.extra,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class ResponseCache:
    """Thread-safe SQLite cache for LLM responses.

    SQLite rather than an in-memory dict because the value of the cache is
    realised *across* processes: the Streamlit UI, the pytest suite and the CI
    job all hit the same file, so a judge verdict is paid for once.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS llm_cache (
        key         TEXT PRIMARY KEY,
        model       TEXT NOT NULL,
        temperature REAL NOT NULL,
        response    TEXT NOT NULL,
        created_at  REAL NOT NULL
    );
    """

    def __init__(self, path, enabled: bool = True) -> None:
        self._path = str(path)
        self._enabled = enabled
        self._lock = threading.Lock()
        if self._enabled:
            with self._connect() as conn:
                conn.execute(self._SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, key: CacheKey) -> str | None:
        if not self._enabled:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key.digest(),)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: CacheKey, response: str) -> None:
        if not self._enabled:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?)",
                (key.digest(), key.model, key.temperature, response, time.time()),
            )

    def stats(self) -> dict[str, Any]:
        if not self._enabled:
            return {"enabled": False, "rows": 0}
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        return {"enabled": True, "rows": rows, "path": self._path}

    def clear(self) -> int:
        if not self._enabled:
            return 0
        with self._lock, self._connect() as conn:
            deleted = conn.execute("DELETE FROM llm_cache").rowcount
        return deleted


def default_cache(settings: Settings | None = None) -> ResponseCache:
    settings = settings or get_settings()
    return ResponseCache(settings.cache_path)


def render_messages(messages: Sequence[Any]) -> str:
    """Flatten a LangChain message list into a stable cache-key string."""
    parts = []
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", str(message))
        parts.append(f"{role}:{content}")
    return "\n---\n".join(parts)
