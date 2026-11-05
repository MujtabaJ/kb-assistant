"""Build an Assistant from Settings."""

from __future__ import annotations

from .config import Settings
from .embeddings import make_embedder
from .llm import make_llm
from .rag import Assistant
from .store import VectorStore, make_store

PG_VECTOR_DIM = 1536  # must match vector(N) in db/schema.sql


def build_assistant(settings: Settings, store: VectorStore | None = None, company: str = "the company") -> Assistant:
    embed_model = settings.openai_embedding_model if settings.embedding_provider == "openai" else None
    embedder = make_embedder(settings.embedding_provider, embed_model)
    if settings.store == "pgvector" and embedder.dim != PG_VECTOR_DIM:
        raise ValueError(
            f"db/schema.sql stores vector({PG_VECTOR_DIM}) but the '{settings.embedding_provider}' embedder "
            f"produces {embedder.dim} dims. Use KB_EMBEDDING_PROVIDER=openai or change the schema."
        )
    return Assistant(
        store=store or make_store(settings.store, settings.database_url),
        embedder=embedder,
        llm=make_llm(settings.llm_provider, settings.openai_chat_model, settings.anthropic_model),
        company=company,
        top_k=settings.top_k,
        min_score=settings.min_score,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
