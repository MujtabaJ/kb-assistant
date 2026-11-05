"""Vector stores with hybrid (vector + keyword) search.

MemoryStore: numpy, for demos and tests (can be saved to and loaded from disk).
PgVectorStore: PostgreSQL + pgvector for production (see db/schema.sql).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunking import Chunk
from .embeddings import tokenize

VECTOR_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35


class VectorStore(Protocol):
    def add(self, org_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, org_id: str, query_vec: np.ndarray, query_text: str, k: int) -> list[tuple[Chunk, float]]: ...
    def delete_source(self, org_id: str, source: str) -> int: ...
    def count(self, org_id: str) -> int: ...


class MemoryStore:
    def __init__(self) -> None:
        self._chunks: dict[str, list[Chunk]] = {}
        self._vecs: dict[str, np.ndarray] = {}

    # --- write -----------------------------------------------------------
    def add(self, org_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        # re-ingesting a source replaces it (idempotent ingestion)
        for src in {c.source for c in chunks}:
            self.delete_source(org_id, src)
        existing = self._vecs.get(org_id)
        self._chunks.setdefault(org_id, []).extend(chunks)
        self._vecs[org_id] = vectors if existing is None or len(existing) == 0 else np.vstack([existing, vectors])

    def delete_source(self, org_id: str, source: str) -> int:
        chunks = self._chunks.get(org_id, [])
        keep = [i for i, c in enumerate(chunks) if c.source != source]
        removed = len(chunks) - len(keep)
        if removed:
            self._chunks[org_id] = [chunks[i] for i in keep]
            self._vecs[org_id] = self._vecs[org_id][keep]
        return removed

    def count(self, org_id: str) -> int:
        return len(self._chunks.get(org_id, []))

    # --- read ------------------------------------------------------------
    def _keyword_scores(self, chunks: list[Chunk], query_text: str) -> np.ndarray:
        q = set(tokenize(query_text))
        if not q or not chunks:
            return np.zeros(len(chunks), dtype=np.float32)
        docs = [Counter(tokenize(f"{c.section} {c.text}")) for c in chunks]
        n = len(docs)
        idf = {t: math.log(1 + n / (1 + sum(1 for d in docs if t in d))) for t in q}
        total = sum(idf.values()) or 1.0
        return np.array([sum(idf[t] for t in q if t in d) / total for d in docs], dtype=np.float32)

    def search(self, org_id: str, query_vec: np.ndarray, query_text: str, k: int) -> list[tuple[Chunk, float]]:
        chunks = self._chunks.get(org_id, [])
        if not chunks:
            return []
        vec_scores = self._vecs[org_id] @ query_vec.reshape(-1)
        vec_scores = np.clip(vec_scores, 0, 1)
        scores = VECTOR_WEIGHT * vec_scores + KEYWORD_WEIGHT * self._keyword_scores(chunks, query_text)
        order = np.argsort(-scores)[:k]
        return [(chunks[i], float(scores[i])) for i in order]

    # --- persistence for the CLI demo -----------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {org: [asdict(c) for c in cs] for org, cs in self._chunks.items()}
        (path / "chunks.json").write_text(json.dumps(meta))
        np.savez(path / "vectors.npz", **self._vecs)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        path = Path(path)
        store = cls()
        meta = json.loads((path / "chunks.json").read_text())
        vecs = np.load(path / "vectors.npz")
        for org, cs in meta.items():
            store._chunks[org] = [Chunk(**c) for c in cs]
            store._vecs[org] = vecs[org]
        return store


class PgVectorStore:
    """PostgreSQL + pgvector. Every query is scoped by org_id (tenant isolation)."""

    def __init__(self, dsn: str):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self.conn)

    def add(self, org_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.conn.transaction():
            for src in {c.source for c in chunks}:
                self.delete_source(org_id, src)
            with self.conn.cursor() as cur:
                for c, v in zip(chunks, vectors):
                    cur.execute(
                        """INSERT INTO chunks (org_id, chunk_key, source, title, section, position, content, metadata, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (org_id, c.id, c.source, c.title, c.section, c.position, c.text, json.dumps(c.metadata), v),
                    )

    def delete_source(self, org_id: str, source: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE org_id=%s AND source=%s", (org_id, source))
            return cur.rowcount

    def count(self, org_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE org_id=%s", (org_id,))
            return cur.fetchone()[0]

    def search(self, org_id: str, query_vec: np.ndarray, query_text: str, k: int) -> list[tuple[Chunk, float]]:
        # Over-fetch by vector similarity, then blend with full-text rank.
        sql = """
        WITH cand AS (
          SELECT chunk_key, source, title, section, position, content, metadata,
                 GREATEST(0, 1 - (embedding <=> %(v)s)) AS vscore,
                 ts_rank_cd(tsv, plainto_tsquery('english', %(q)s)) AS kscore
          FROM chunks WHERE org_id = %(org)s
          ORDER BY embedding <=> %(v)s
          LIMIT %(n)s)
        SELECT *, (%(vw)s * vscore + %(kw)s * LEAST(kscore, 1)) AS score
        FROM cand ORDER BY score DESC LIMIT %(k)s"""
        params = {"v": query_vec, "q": query_text, "org": org_id, "n": k * 5, "k": k,
                  "vw": VECTOR_WEIGHT, "kw": KEYWORD_WEIGHT}
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            (Chunk(id=r[0], source=r[1], title=r[2], section=r[3], position=r[4], text=r[5], metadata=r[6] or {}), float(r[-1]))
            for r in rows
        ]


def make_store(kind: str, dsn: str | None = None) -> VectorStore:
    if kind == "memory":
        return MemoryStore()
    if kind == "pgvector":
        return PgVectorStore(dsn or "")
    raise ValueError(f"Unknown store: {kind}")
