"""Sentence-aware fixed-size chunking sized in tokens of the embedding model's tokenizer.

Greedy sentence packing up to chunk_tokens; sliding overlap of overlap_tokens between
adjacent chunks. Sentences longer than chunk_tokens are hard-split on token boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_TOKENS = 384  # smaller chunks → finer retrieval; fits BGE 512 context comfortably
DEFAULT_OVERLAP_TOKENS = 96

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class TextChunk:
    text: str
    page: int | None
    chunk_index: int


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]


def chunk_text(
    text: str,
    page: int | None,
    start_index: int,
    tokenizer,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    sent_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in sentences]

    chunks: list[TextChunk] = []
    cur_indices: list[int] = []  # indices into `sentences` currently buffered
    cur_count = 0
    out_idx = start_index

    def emit() -> None:
        nonlocal cur_indices, cur_count, out_idx
        if not cur_indices:
            return
        chunk_text_str = " ".join(sentences[j] for j in cur_indices)
        chunks.append(TextChunk(chunk_text_str, page, out_idx))
        out_idx += 1
        # Build overlap from tail.
        tail: list[int] = []
        tail_count = 0
        for j in reversed(cur_indices):
            n = len(sent_tokens[j])
            if tail_count + n > overlap_tokens:
                break
            tail.insert(0, j)
            tail_count += n
        cur_indices = tail
        cur_count = tail_count

    i = 0
    while i < len(sentences):
        n = len(sent_tokens[i])
        if n > chunk_tokens:
            emit()
            ids = sent_tokens[i]
            stride = max(1, chunk_tokens - overlap_tokens)
            for start in range(0, len(ids), stride):
                window = ids[start:start + chunk_tokens]
                chunks.append(TextChunk(tokenizer.decode(window), page, out_idx))
                out_idx += 1
            i += 1
            continue

        if cur_count + n <= chunk_tokens:
            cur_indices.append(i)
            cur_count += n
            i += 1
        else:
            # Sentence i won't fit in the current chunk. Flush, then retry under
            # the new chunk (which now contains the overlap-tail of the old one).
            emit()
            # If the overlap-tail PLUS sentence i still doesn't fit, the overlap
            # itself is too big for this sentence. Drop the overlap to prevent an
            # infinite loop where emit() keeps regenerating the same tail and
            # sentence i never makes progress.
            if cur_count + n > chunk_tokens:
                cur_indices = []
                cur_count = 0
            # Retry sentence i.

    if cur_indices:
        chunk_text_str = " ".join(sentences[j] for j in cur_indices)
        chunks.append(TextChunk(chunk_text_str, page, out_idx))

    return chunks
