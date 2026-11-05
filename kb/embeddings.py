"""Embedding providers.

`HashingEmbedder` needs no API key: it hashes word unigrams and bigrams into a
fixed-size vector. It's good enough for demos and tests. Use OpenAI embeddings
for real semantic search.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or of to in on for with is are was were be been it this that these those "
    "i you we they he she my your our their do does did can could should would will how what "
    "when where which who why if at by from as about into than then there here have has had".split()
)


def tokenize(text: str) -> list[str]:
    toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
    return [_stem(t) for t in toks]


def _stem(t: str) -> str:
    """Tiny suffix stripper: refunds->refund, replies->reply, processed->process."""
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("sses"):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    for suf in ("ing", "ed"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        n = int.from_bytes(h, "little")
        return n % self.dim, 1.0 if (n >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for g in grams:
                j, sign = self._bucket(g)
                mat[i, j] += sign
            norm = np.linalg.norm(mat[i])
            if norm:
                mat[i] /= norm
        return mat


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for start in range(0, len(texts), 96):  # batch to stay under request limits
            resp = self.client.embeddings.create(model=self.model, input=texts[start : start + 96])
            vecs.extend(d.embedding for d in resp.data)
        mat = np.asarray(vecs, dtype=np.float32)
        return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def make_embedder(provider: str, model: str | None = None) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    if provider == "offline":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider}")
