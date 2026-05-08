"""Verify .doc / .ppt files are detected, reported, and skipped without error."""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from vectorise_mcp import indexer, store


async def fake_progress(progress, total, message):
    print(f"  [progress] {message}")


async def main() -> int:
    project = "smoke_legacy_skip"
    db = store.project_path(project)
    if db.exists():
        db.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Real supported file:
        (root / "good.txt").write_text("Some real content. Indexable.", encoding="utf-8")
        # Fake legacy files (just empty bytes — we never parse them):
        (root / "old_report.doc").write_bytes(b"\xD0\xCF\x11\xE0fake")
        (root / "presentation.ppt").write_bytes(b"\xD0\xCF\x11\xE0fake")

        result = await indexer.index_folder_into(
            folder_path=str(root),
            project_name=project,
            progress=fake_progress,
        )

    print(f"\nresult: {result}")
    print(f"legacy skipped count: {len(result.legacy_files_skipped)}")
    for p in result.legacy_files_skipped:
        print(f"  - {Path(p).name}")

    assert result.files_indexed == 1, f"expected 1 supported file indexed, got {result.files_indexed}"
    assert len(result.legacy_files_skipped) == 2, f"expected 2 legacy detected, got {len(result.legacy_files_skipped)}"

    print("\nCLEANUP")
    print(store.delete_project(project))
    print("\nLEGACY SKIP OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
