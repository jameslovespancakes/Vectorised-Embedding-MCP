"""End-to-end smoke test against the project API: index, search, cleanup."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE_DIR = HERE / "sample_docs"
PROJECT = "smoke_test_sample_docs"

from vectorise_mcp import embedder, indexer, reranker, store


async def fake_progress(progress: int, total: int, message: str) -> None:
    print(f"  [progress {progress}/{total}] {message}")


async def main() -> int:
    db = store.project_path(PROJECT)
    if db.exists():
        db.unlink()
        print(f"removed prior db: {db}")

    print("\n=== INDEX ===")
    t0 = time.time()
    result = await indexer.index_folder_into(
        folder_path=str(SAMPLE_DIR),
        project_name=PROJECT,
        progress=fake_progress,
    )
    print(f"indexed: {result}")
    print(f"wall time: {time.time() - t0:.2f}s")

    print("\n=== STATS ===")
    s = store.ProjectStore(PROJECT)
    print(s.stats())
    print("source_paths:", s.get_source_paths())
    s.close()

    queries = [
        ("How big was Q1 revenue?", "finance.md"),
        ("Tell me about cookies", "cooking.txt"),
        ("Where is Sagittarius A*?", "space.md"),
        ("buyback authorization size", "finance.md"),
    ]

    for q, expected_file in queries:
        print(f"\n=== SEARCH: {q!r} ===")
        s = store.ProjectStore(PROJECT)
        try:
            qe = embedder.embed_query(q)
            cands = s.hybrid_search(qe, q, n_per_side=10)
        finally:
            s.close()

        if not cands:
            print("  NO CANDIDATES")
            continue

        scores = reranker.rerank(q, [c.text for c in cands])
        ranked = sorted(zip(cands, scores), key=lambda p: float(p[1]), reverse=True)[:3]
        for i, (hit, sc) in enumerate(ranked, 1):
            marker = "*" if hit.display_name == expected_file else " "
            preview = hit.text[:120].replace("\n", " ")
            print(f"  {marker} #{i} score={float(sc):.3f} sim={hit.score:.3f} "
                  f"display={hit.display_name} :: {preview}...")

    print("\n=== CLEANUP ===")
    print(store.delete_project(PROJECT))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
