"""Split documents into overlapping, heading-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Document:
    source: str  # file name or URL
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str
    position: int
    metadata: dict = field(default_factory=dict)


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (section heading, body) pairs for markdown; plain text is one section."""
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _split(body: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into windows of ~size chars, overlapping by ~overlap chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    for p in paras:  # hard-split very long paragraphs on sentence boundaries
        if len(p) <= size:
            pieces.append(p)
            continue
        sent, cur = [], ""
        for s in re.split(r"(?<=[.!?])\s+", p):
            # a "sentence" longer than the window (no punctuation) is split on words
            while len(s) > size:
                cut = s.rfind(" ", 0, size)
                cut = cut if cut > 0 else size
                sent.append(s[:cut])
                s = s[cut:].strip()
            sent.append(s)
        for s in sent:
            if cur and len(cur) + len(s) + 1 > size:
                pieces.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            pieces.append(cur)

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            cur = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            cur = f"{cur}\n\n{piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc: Document, size: int = 900, overlap: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    for heading, body in _sections(doc.text):
        for piece in _split(body, size, overlap):
            pos = len(out)
            out.append(
                Chunk(
                    id=f"{doc.source}#{pos}",
                    source=doc.source,
                    title=doc.title,
                    section=heading or doc.title,
                    text=piece,
                    position=pos,
                    metadata=dict(doc.metadata),
                )
            )
    return out
