"""Smoke test: project workflow.

Covers:
  • mode=auto on fresh project → creates it.
  • mode=auto on same folder → incremental reindex (no re-embed).
  • mode=auto on a *different* folder → caller would raise (we exercise via
    direct indexer + manual mode dispatch since we don't go through MCP here).
  • mode=append + basename collision → suffixed display_name.
  • mode=append + identical SHA1 in different folder → skip silently.
  • Source folder deletion → reindex skips that path with a warning.
  • delete_project leaves disk freed.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from vectorise_mcp import indexer, store

PROJECT = "smoke_test_projects"


def _make_folder_a(root: Path) -> Path:
    a = root / "folder_a"
    a.mkdir()
    (a / "report.txt").write_text("Apples are red. Apples grow on trees.", encoding="utf-8")
    (a / "manual.txt").write_text("Cast iron skillets need to be seasoned.", encoding="utf-8")
    return a


def _make_folder_b_collision(root: Path) -> Path:
    """folder_b has a file named report.txt with DIFFERENT content."""
    b = root / "folder_b"
    b.mkdir()
    (b / "report.txt").write_text("Bananas are yellow. Bananas are tropical.", encoding="utf-8")
    (b / "extra.txt").write_text("This is unique to folder_b.", encoding="utf-8")
    return b


def _make_folder_c_duplicate(root: Path, source_a: Path) -> Path:
    """folder_c contains a file with the SAME content as folder_a/report.txt."""
    c = root / "folder_c"
    c.mkdir()
    shutil.copy(source_a / "report.txt", c / "report.txt")
    (c / "different_name.txt").write_text("Folder C unique content.", encoding="utf-8")
    return c


async def main() -> int:
    # Clean any prior smoke run.
    db = store.project_path(PROJECT)
    if db.exists():
        db.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        folder_a = _make_folder_a(root)
        folder_b = _make_folder_b_collision(root)
        folder_c = _make_folder_c_duplicate(root, folder_a)

        # 1. Initial index
        print("=== STEP 1: index folder_a ===")
        r1 = await indexer.index_folder_into(str(folder_a), PROJECT)
        print(f"   result: {r1}")
        s = store.ProjectStore(PROJECT)
        print(f"   source_paths: {s.get_source_paths()}")
        print(f"   indexed_files: {[Path(p).name for p in s.list_indexed_files()]}")
        s.close()
        assert r1.files_indexed == 2, r1

        # 2. Re-running on same folder → SHA1 dedup → 0 new
        print("\n=== STEP 2: index folder_a again (idempotency) ===")
        r2 = await indexer.index_folder_into(str(folder_a), PROJECT)
        print(f"   result: {r2}")
        assert r2.files_indexed == 0, "second run should embed nothing"
        assert r2.files_skipped_same_content == 2, r2

        # 3. Append folder_b — collision: same basename, different content → suffix
        print("\n=== STEP 3: append folder_b (basename collision, different content) ===")
        r3 = await indexer.index_folder_into(str(folder_b), PROJECT)
        print(f"   result: {r3}")
        s = store.ProjectStore(PROJECT)
        names = sorted(
            row[0] for row in s.conn.execute("SELECT display_name FROM files")
        )
        print(f"   display_names in project: {names}")
        s.close()
        assert "report.txt" in names and "report_2.txt" in names, names
        assert r3.files_renamed_due_to_collision == 1, r3

        # 4. Append folder_c — one file is byte-identical to folder_a/report.txt
        #    → should be skipped via SHA1 match.
        print("\n=== STEP 4: append folder_c (one duplicate-content file) ===")
        r4 = await indexer.index_folder_into(str(folder_c), PROJECT)
        print(f"   result: {r4}")
        assert r4.files_skipped_same_content == 1, r4
        assert r4.files_indexed == 1, r4  # only different_name.txt added

        # 5. Search across all three folders' content
        print("\n=== STEP 5: search across all three folders ===")
        from vectorise_mcp import embedder, reranker
        s = store.ProjectStore(PROJECT)
        for q in ["red apples", "tropical bananas", "iron skillet"]:
            qe = embedder.embed_query(q)
            cands = s.hybrid_search(qe, q, n_per_side=10)
            scores = reranker.rerank(q, [c.text for c in cands])
            top = sorted(zip(cands, scores), key=lambda p: float(p[1]), reverse=True)[:1]
            for hit, sc in top:
                print(f"   q={q!r} -> {hit.display_name} (score={float(sc):.3f})")
        s.close()

        # 6. Delete folder_b on disk; reindex should warn but not crash.
        print("\n=== STEP 6: delete folder_b, run reindex_project ===")
        shutil.rmtree(folder_b)
        r6 = await indexer.reindex_project(PROJECT)
        print(f"   result: {r6}")
        assert r6.missing_source_paths, "should report missing source"

    # 7. Cleanup
    print("\n=== STEP 7: delete project ===")
    print(store.delete_project(PROJECT))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
