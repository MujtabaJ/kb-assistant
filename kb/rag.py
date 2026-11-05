"""The Assistant: ingest documents, retrieve, answer with validated citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .chunking import Chunk, Document, chunk_document
from .embeddings import Embedder
from .llm import LLM, REFUSAL, SYSTEM_PROMPT
from .store import VectorStore

_CITE = re.compile(r"\[(\d+)\]")

# Prompt-injection defence, layer 1: strip instruction-like text from retrieved
# sources before any model sees it. Layer 2 is the system prompt, which tells
# the model to treat sources as data. Both are logged and covered by the eval set.
_INJECTION = re.compile(
    r"[^.!?\n]*("
    r"ignore (all |any |your |the )?(previous|prior|above|earlier) (instructions|prompts|rules)"
    r"|disregard (the |your |all )?(system|previous|prior)"
    r"|\b(note|message|instruction)s? (to|for) (ai|llm|chatbot|assistant|language model)s?\b"
    r"|you are now\b|reveal (the |your )?system prompt"
    r")[^.!?\n]*[.!?]?",
    re.I,
)


def sanitize(text: str) -> tuple[str, int]:
    """Remove sentences that look like instructions aimed at an AI. Returns (clean_text, removed_count)."""
    cleaned, n = _INJECTION.subn("", text)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), n
# Rough public list prices in USD per 1M tokens, for cost estimates only. Update as needed.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "claude-3-5-haiku-latest": (0.80, 4.00)}


@dataclass
class Citation:
    number: int
    source: str
    section: str
    snippet: str
    score: float


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    grounded: bool = True  # False if the model answered without valid citations
    top_score: float = 0.0
    injection_flags: int = 0  # suspicious instructions removed from sources
    latency_ms: int = 0
    cost_usd: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "refused": self.refused,
            "grounded": self.grounded,
            "top_score": round(self.top_score, 3),
            "injection_flags": self.injection_flags,
            "latency_ms": self.latency_ms,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "citations": [c.__dict__ for c in self.citations],
        }


class Assistant:
    def __init__(self, store: VectorStore, embedder: Embedder, llm: LLM, *, company: str = "the company",
                 top_k: int = 4, min_score: float = 0.22, chunk_size: int = 900, chunk_overlap: int = 150):
        self.store, self.embedder, self.llm = store, embedder, llm
        self.company, self.top_k, self.min_score = company, top_k, min_score
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    # ------------------------------------------------------------------ ingest
    def ingest(self, org_id: str, docs: list[Document]) -> int:
        chunks: list[Chunk] = []
        for d in docs:
            chunks.extend(chunk_document(d, self.chunk_size, self.chunk_overlap))
        if not chunks:
            return 0
        vectors = self.embedder.embed([f"{c.title} - {c.section}\n{c.text}" for c in chunks])
        self.store.add(org_id, chunks, vectors)
        return len(chunks)

    # --------------------------------------------------------------- retrieval
    def retrieve(self, org_id: str, question: str) -> list[tuple[Chunk, float]]:
        qv = self.embedder.embed([question])[0]
        return self.store.search(org_id, qv, question, self.top_k)

    @staticmethod
    def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> tuple[str, int]:
        blocks, flags = [], 0
        for i, (c, _) in enumerate(hits, 1):
            clean, n = sanitize(c.text)
            flags += n
            blocks.append(f"[{i}] {c.title} > {c.section} ({c.source})\n{clean}")
        return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}", flags

    # ------------------------------------------------------------------- answer
    def ask(self, org_id: str, question: str) -> Answer:
        t0 = time.perf_counter()
        question = question.strip()[:2000]  # bound input size
        hits = self.retrieve(org_id, question)
        top = hits[0][1] if hits else 0.0

        if not hits or top < self.min_score:
            return Answer(question, REFUSAL, refused=True, top_score=top,
                          latency_ms=int((time.perf_counter() - t0) * 1000))

        prompt, flags = self.build_prompt(question, hits)
        result = self.llm.generate(SYSTEM_PROMPT.format(company=self.company), prompt)
        text = result.text.strip()
        refused = REFUSAL.lower().rstrip(".") in text.lower()

        cited = sorted({int(n) for n in _CITE.findall(text)})
        valid = [n for n in cited if 1 <= n <= len(hits)]
        citations = [
            Citation(n, hits[n - 1][0].source, hits[n - 1][0].section, hits[n - 1][0].text[:220], round(hits[n - 1][1], 3))
            for n in valid
        ]
        grounded = refused or (bool(valid) and len(valid) == len(cited))

        pin, pout = PRICES.get(result.model, (0.0, 0.0))
        cost = (result.input_tokens * pin + result.output_tokens * pout) / 1_000_000
        return Answer(question, text, citations, refused=refused, grounded=grounded, top_score=top,
                      injection_flags=flags, latency_ms=int((time.perf_counter() - t0) * 1000), cost_usd=cost, model=result.model)
