"""Cross-encoder reranker. Scores (query, passage) pairs jointly — much higher precision
than bi-encoder retrieval alone. Used to re-order the top-N from hybrid search.

Default: BAAI/bge-reranker-base — ~110MB, balanced speed/quality.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("VECTORISE_MCP_RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_BATCH_SIZE = int(os.environ.get("VECTORISE_MCP_RERANKER_BATCH", "16"))

_model = None
_lock = threading.Lock()


def get_model(name: str = DEFAULT_MODEL):
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker %s (one-time download if absent)", name)
            _model = CrossEncoder(name)
    return _model


def warm_up() -> None:
    """Force model load. Used by setup + serve boot."""
    get_model()


def rerank(query: str, passages: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """Score each passage's relevance to query. Returns float array of length len(passages).

    Higher score = more relevant. Caller sorts by score descending.
    """
    if not passages:
        return np.zeros((0,), dtype=np.float32)
    model = get_model()
    pairs = [(query, p) for p in passages]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)
