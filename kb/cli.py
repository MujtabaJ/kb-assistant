"""Command-line demo. Works offline, with no API keys and no database.

  python -m kb.cli ingest sample_data/northwind
  python -m kb.cli ask "How long do I have to request a refund?"
  python -m kb.cli eval evals/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .evals import run_eval
from .factory import build_assistant
from .loaders import load_folder
from .store import MemoryStore

INDEX_DIR = Path(".kb_index")
ORG = "demo"
COMPANY = "Northwind Desk"


def _assistant(load: bool = True):
    settings = get_settings()
    store = None
    if settings.store == "memory":
        store = MemoryStore.load(INDEX_DIR) if load and INDEX_DIR.exists() else MemoryStore()
    return build_assistant(settings, store=store, company=COMPANY), store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kb")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("folder")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    sub.add_parser("eval").add_argument("file")
    args = p.parse_args(argv)

    if args.cmd == "ingest":
        assistant, store = _assistant(load=False)
        docs = load_folder(args.folder)
        n = assistant.ingest(ORG, docs)
        if isinstance(store, MemoryStore):
            store.save(INDEX_DIR)
        print(f"Indexed {len(docs)} documents into {n} chunks.")
        return 0

    assistant, _ = _assistant()
    if args.cmd == "ask":
        ans = assistant.ask(ORG, args.question)
        if args.json:
            print(json.dumps(ans.to_dict(), indent=2))
        else:
            print(ans.text)
            for c in ans.citations:
                print(f"  [{c.number}] {c.source} > {c.section} (score {c.score})")
        return 0

    if args.cmd == "eval":
        report = run_eval(assistant, ORG, args.file)
        print(report.summary())
        return 0 if report.metrics["pass_rate"] >= 0.8 else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
