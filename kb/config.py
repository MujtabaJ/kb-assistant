"""Configuration read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # "offline" runs with no API keys (deterministic embeddings + extractive answers).
    # "openai" or "anthropic" use real models.
    llm_provider: str = _env("KB_LLM_PROVIDER", "offline")
    embedding_provider: str = _env("KB_EMBEDDING_PROVIDER", "offline")
    openai_chat_model: str = _env("KB_OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = _env("KB_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    anthropic_model: str = _env("KB_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    # "memory" keeps vectors in RAM (demo/tests); "pgvector" uses PostgreSQL.
    store: str = _env("KB_STORE", "memory")
    database_url: str = _env("DATABASE_URL", "postgresql://kb:kb@localhost:5432/kb")

    chunk_size: int = int(_env("KB_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(_env("KB_CHUNK_OVERLAP", "150"))
    top_k: int = int(_env("KB_TOP_K", "4"))
    # Below this retrieval score the assistant refuses instead of guessing.
    min_score: float = float(_env("KB_MIN_SCORE", "0.22"))

    api_key: str = _env("KB_API_KEY", "change-me")


def get_settings() -> Settings:
    return Settings()
