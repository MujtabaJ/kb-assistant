"""FastAPI service around the kb core.

Run:  uvicorn app.api:app --reload
Docs: http://localhost:8000/docs   Demo chat: http://localhost:8000/
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from kb.config import get_settings
from kb.factory import build_assistant
from kb.loaders import SUPPORTED, load_file, load_folder

settings = get_settings()
assistant = build_assistant(settings, company=os.environ.get("KB_COMPANY", "Northwind Desk"))
log = logging.getLogger("kb")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Knowledge Base Assistant", version="0.1.0")

# Demo convenience: preload a folder into the "default" tenant on startup.
_preload = os.environ.get("KB_PRELOAD_DIR", "sample_data/northwind")
if _preload and Path(_preload).is_dir():
    assistant.ingest("default", load_folder(_preload))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
RATE_LIMIT_PER_MIN = int(os.environ.get("KB_RATE_LIMIT_PER_MIN", "60"))
_calls: dict[str, deque] = defaultdict(deque)


# ---------------------------------------------------------------- auth & limits
def _key_map() -> dict[str, str]:
    """KB_API_KEYS="key1:org1,key2:org2" maps API keys to tenants. Falls back to KB_API_KEY -> 'default'."""
    raw = os.environ.get("KB_API_KEYS", "")
    pairs = [p.split(":", 1) for p in raw.split(",") if ":" in p]
    return {k.strip(): o.strip() for k, o in pairs} or {settings.api_key: "default"}


def tenant(x_api_key: str = Header(default="")) -> str:
    for key, org in _key_map().items():
        if x_api_key and hmac.compare_digest(x_api_key, key):
            window = _calls[org]
            now = time.time()
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= RATE_LIMIT_PER_MIN:
                raise HTTPException(429, "Rate limit exceeded")
            window.append(now)
            return org
    raise HTTPException(401, "Invalid or missing X-API-Key")


# ------------------------------------------------------------------- endpoints
class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "store": settings.store, "llm": settings.llm_provider}


@app.post("/v1/ask")
def ask(body: AskIn, org: str = Depends(tenant)) -> dict:
    ans = assistant.ask(org, body.question)
    # Structured log line per request: feeds dashboards and cost tracking.
    log.info(json.dumps({"event": "ask", "org": org, "refused": ans.refused, "grounded": ans.grounded,
                         "top_score": round(ans.top_score, 3), "injection_flags": ans.injection_flags,
                         "latency_ms": ans.latency_ms, "cost_usd": ans.cost_usd, "model": ans.model}))
    return ans.to_dict()


@app.post("/v1/documents")
async def upload(files: list[UploadFile] = File(...), org: str = Depends(tenant)) -> dict:
    docs = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = Path(f.filename or "upload").name  # strip any path components
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(415, f"Unsupported file type: {name}")
            data = await f.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File too large: {name}")
            path = Path(tmp) / name
            path.write_bytes(data)
            docs.append(load_file(path))
        chunks = assistant.ingest(org, docs)
    return {"documents": len(docs), "chunks": chunks}


@app.delete("/v1/documents/{source}")
def delete(source: str, org: str = Depends(tenant)) -> dict:
    return {"deleted_chunks": assistant.store.delete_source(org, source)}


# --------------------------------------------------------------- demo chat page
@app.get("/", response_class=HTMLResponse)
def chat_page() -> str:
    return (Path(__file__).parent / "chat.html").read_text(encoding="utf-8")
