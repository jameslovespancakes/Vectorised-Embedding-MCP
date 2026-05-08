"""Regression test for chunking infinite-loop bug.

Bug: when overlap-tail tokens + next sentence > chunk_tokens, emit() rebuilt
the tail from itself and the loop never advanced. Repro: any text with
sentences whose token count is between (chunk_tokens - overlap_tokens) and
chunk_tokens, packed densely enough that overlap doesn't free room.
"""

from __future__ import annotations

import sys
import time

from vectorise_mcp import embedder
from vectorise_mcp.chunking import chunk_text


REPRO_TEXT = (
    # 6 sentences, each ~290 tokens. Overlap of 96 with chunk_tokens=384 means
    # tail (one sentence) PLUS next sentence = 580 > 384, so overlap can't
    # absorb. Without the fix, this loops forever.
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    * 35  # one big sentence ~280 tokens
    + ". "
    + "kilo lima mike november oscar papa quebec romeo sierra tango "
    * 35
    + ". "
    + "uniform victor whiskey xray yankee zulu apple banana cherry date "
    * 35
    + ". "
    + "elder fig grape honeydew iris jasmine kiwi lemon mango nutmeg "
    * 35
    + ". "
    + "olive papaya quince raspberry strawberry tangerine ugli vanilla "
    * 35
    + ". "
    + "watermelon xigua yuzu zucchini avocado blueberry cucumber durian "
    * 35
    + "."
)


def main() -> int:
    tok = embedder.get_tokenizer()
    print(f"text length: {len(REPRO_TEXT)} chars", flush=True)
    sent_count = REPRO_TEXT.count(".")
    print(f"sentence count: {sent_count}", flush=True)

    t0 = time.time()
    DEADLINE = 30.0  # if this hangs, fail in 30s instead of forever

    import threading
    result = {"chunks": None, "err": None}
    def worker():
        try:
            result["chunks"] = chunk_text(REPRO_TEXT, page=None, start_index=0, tokenizer=tok)
        except Exception as e:
            result["err"] = e
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=DEADLINE)

    if t.is_alive():
        print(f"FAIL: chunk_text hung > {DEADLINE}s — infinite loop bug not fixed")
        return 1
    if result["err"]:
        print(f"FAIL: {result['err']!r}")
        return 1

    chunks = result["chunks"]
    elapsed = time.time() - t0
    print(f"chunked in {elapsed:.2f}s, {len(chunks)} chunks")
    for c in chunks:
        # ~ token count check
        n_tokens = len(tok.encode(c.text, add_special_tokens=False))
        print(f"  chunk {c.chunk_index}: {len(c.text)} chars, ~{n_tokens} tokens")
    print("\nCHUNKING OK — no infinite loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
