"""kb: a small, production-minded RAG (retrieval-augmented generation) core.

The core has no web-framework dependency so it can be tested anywhere.
`app/api.py` wraps it in a FastAPI service.
"""

from .rag import Answer, Assistant, Citation

__all__ = ["Answer", "Assistant", "Citation"]
