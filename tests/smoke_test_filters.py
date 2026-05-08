"""Smoke test for metadata filtering. Indexes sample_docs/ into a project, runs filtered searches."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE_DIR = HERE / "sample_docs"
PROJECT = "smoke_test_filters"

from vectorise_mcp import embedder, indexer, reranker, store


async def main() -> int:
    db = store.project_path(PROJECT)
    if db.exists():
        db.unlink()

    print("=== INDEX ===")
    await indexer.index_folder_into(str(SAMPLE_DIR), PROJECT, progress=None)

    s = store.ProjectStore(PROJECT)
    print("indexed paths:", s.list_indexed_files())
    s.close()

    print("\n=== TEST: file_glob='*.md' ===")
    await _run_search("buyback", file_glob="*.md")

    print("\n=== TEST: file_glob='cooking*' ===")
    await _run_search("buyback", file_glob="cooking*")

    print("\n=== TEST: subdirectory='sample_docs' ===")
    await _run_search("space", subdirectory="sample_docs")

    print("\n=== TEST: min_similarity=0.55 ===")
    await _run_search("Q1 revenue", min_similarity=0.55)

    print("\n=== TEST: page_min=2 (excludes non-PDFs) ===")
    await _run_search("anything", page_min=2)

    print("\n=== CLEANUP ===")
    print(store.delete_project(PROJECT))
    return 0


async def _run_search(query: str, **filter_kwargs) -> None:
    flt = store.SearchFilter(**filter_kwargs)
    s = store.ProjectStore(PROJECT)
    try:
        qe = embedder.embed_query(query)
        cands = s.hybrid_search(qe, query, n_per_side=20, filter=flt)
    finally:
        s.close()

    print(f"  query={query!r} filter={filter_kwargs} -> {len(cands)} candidates")
    if not cands:
        return
    scores = reranker.rerank(query, [c.text for c in cands])
    ranked = sorted(zip(cands, scores), key=lambda p: float(p[1]), reverse=True)[:3]
    for hit, sc in ranked:
        print(f"    score={float(sc):.3f} sim={hit.score:.3f} display={hit.display_name} page={hit.page}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
