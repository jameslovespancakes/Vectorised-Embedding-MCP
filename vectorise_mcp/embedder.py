"""Embedding model wrapper.

Default: BAAI/bge-small-en-v1.5 — 384-dim, ~130MB, MTEB-strong.
Lazy-loaded; one global instance per process. Override via VECTORISE_MCP_EMBED_MODEL env var.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("VECTORISE_MCP_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_BATCH_SIZE = int(os.environ.get("VECTORISE_MCP_EMBED_BATCH", "32"))
EMBEDDING_DIM = 384  # bge-small-en-v1.5; must match store.EMBEDDING_DIM

# BGE retrieval models expect a query-side prompt for asymmetric search.
# Passages stay raw; queries get prefixed.
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

_model = None
_lock = threading.Lock()


def _is_bge(name: str) -> bool:
    return "bge-" in name.lower()


def get_model(name: str = DEFAULT_MODEL):
    """Return loaded SentenceTransformer. Loads once, cached for process lifetime."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s (one-time download if absent)", name)
            _model = SentenceTransformer(name)
    return _model


def get_tokenizer():
    return get_model().tokenizer


def warm_up() -> None:
    """Force model load. Use during `serve` boot or `setup` to avoid first-call latency."""
    get_model()


def embed_passages(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """Embed document chunks. Returns float32 (N, EMBEDDING_DIM), L2-normalized."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = get_model()
    out = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return out.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a query. Adds BGE query prompt for asymmetric retrieval."""
    name = DEFAULT_MODEL
    payload = (BGE_QUERY_PROMPT + text) if _is_bge(name) else text
    return embed_passages([payload])[0]
