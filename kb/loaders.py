"""Load .md, .txt and .pdf files from a folder into Documents."""

from __future__ import annotations

from pathlib import Path

from .chunking import Document

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader  # optional dependency, imported lazily

        reader = PdfReader(str(path))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
        text = "\n\n".join(f"## Page {i + 1}\n\n{t}" for i, t in enumerate(pages) if t)
        return Document(source=path.name, title=path.stem, text=text, metadata={"pages": len(pages)})
    text = path.read_text(encoding="utf-8")
    return Document(source=path.name, title=_title_from_markdown(text, path.stem), text=text)


def load_folder(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    return [load_file(p) for p in files]
