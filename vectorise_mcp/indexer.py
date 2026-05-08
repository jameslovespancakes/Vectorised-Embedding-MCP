"""Project indexer. Walks files, parses, chunks, embeds, stores. Streams progress with ETA.

Handles same-name file collisions across folders: identical SHA1 → skip silently
(it's the same file). Different SHA1 → store with a suffixed display_name
(`report.pdf` → `report_2.pdf`) so search results stay disambiguated.

`reindex_project` walks every source path the project knows about, skipping any
path that no longer exists on disk (with a warning).
"""

from __future__ import annotations

import collections
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from vectorise_mcp import embedder, parsers
from vectorise_mcp.chunking import TextChunk, chunk_text
from vectorise_mcp.store import Chunk, ProjectStore, now_iso

logger = logging.getLogger(__name__)

AVG_BYTES_PER_CHUNK = 3072

ProgressCb = Callable[[int, int, str], Awaitable[None]]


@dataclass
class IndexResult:
    files_indexed: int = 0
    files_skipped_same_content: int = 0     # same SHA1 already in project
    files_renamed_due_to_collision: int = 0  # same basename, different SHA1
    files_failed: int = 0
    chunks_created: int = 0
    duration_sec: float = 0.0
    legacy_files_skipped: list[str] = None  # .doc / .ppt — unsupported

    def __post_init__(self):
        if self.legacy_files_skipped is None:
            self.legacy_files_skipped = []


@dataclass
class ReindexResult:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    missing_source_paths: list[str] = None
    duration_sec: float = 0.0

    def __post_init__(self):
        if self.missing_source_paths is None:
            self.missing_source_paths = []


def _sha1_of_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _list_supported_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and parsers.is_supported(p))


def _list_legacy_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and parsers.is_unsupported_legacy(p))


def _estimate_chunks(files: list[Path]) -> int:
    total_bytes = sum(f.stat().st_size for f in files)
    return max(1, total_bytes // AVG_BYTES_PER_CHUNK)


def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _build_chunks_for_file(path: Path, tokenizer) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_idx = 0
    for text_block, page in parsers.parse(path):
        block_chunks = chunk_text(text_block, page, chunk_idx, tokenizer)
        chunks.extend(block_chunks)
        chunk_idx += len(block_chunks)
    return chunks


def _classify_file(store: ProjectStore, path: Path, sha1: str) -> tuple[str, str]:
    """Decide what to do with a candidate file.

    Returns (action, display_name) where action is one of:
      - "skip_same_content": this file's SHA1 is already in the project (under
        any path) → no work to do.
      - "update": this exact path is already indexed but the content has changed
        → re-embed.
      - "new": new file. display_name is the de-collided basename (suffixed if
        another file with the same basename and different content exists).
      - "new_renamed": same as "new" but the basename collided and was suffixed.
    """
    existing_at_path = store.get_file_record(str(path))
    if existing_at_path:
        existing_sha, _mtime, _chunks, existing_display = existing_at_path
        if existing_sha == sha1:
            return "skip_same_content", existing_display
        return "update", existing_display

    same_content_path = store.find_file_by_hash(sha1)
    if same_content_path:
        return "skip_same_content", Path(same_content_path).name

    basename = path.name
    if not store.display_name_in_use(basename):
        return "new", basename
    suffixed = store.resolve_display_name(str(path), sha1)
    return "new_renamed", suffixed


async def _embed_file(
    store: ProjectStore,
    file: Path,
    file_hash: str,
    display_name: str,
    tokenizer,
) -> int:
    chunks = _build_chunks_for_file(file, tokenizer)
    if not chunks:
        return 0
    # Yield GIL to keep main asyncio loop (MCP server) responsive while indexer
    # worker thread does heavy CPU work. time.sleep(0) is the canonical way to
    # release GIL momentarily without sleeping in wallclock terms.
    import time
    time.sleep(0)
    embeddings = embedder.embed_passages([c.text for c in chunks])
    time.sleep(0)
    store.insert_chunks(
        file_path=str(file),
        file_hash=file_hash,
        display_name=display_name,
        chunks=[Chunk(text=c.text, page=c.page, chunk_index=c.chunk_index) for c in chunks],
        embeddings=embeddings,
    )
    store.upsert_file_record(
        path=str(file),
        sha1=file_hash,
        mtime=file.stat().st_mtime,
        chunk_count=len(chunks),
        display_name=display_name,
    )
    return len(chunks)


async def index_folder_into(
    folder_path: str,
    project_name: str,
    progress: ProgressCb | None = None,
) -> IndexResult:
    """Index a folder into a project. Caller is responsible for creating/clearing
    the project first if needed (see server.index_project for the mode dispatch).
    Adds folder to the project's source_paths list.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if progress:
        await progress(0, 1, f"Scanning '{folder}' for supported files...")
    files = _list_supported_files(folder)
    legacy = _list_legacy_files(folder)
    store = ProjectStore(project_name)
    store.add_source_path(str(folder))

    result = IndexResult()
    result.legacy_files_skipped = [str(p) for p in legacy]

    if legacy and progress:
        sample = ", ".join(p.name for p in legacy[:5])
        more = f" (and {len(legacy)-5} more)" if len(legacy) > 5 else ""
        await progress(
            0, 1,
            f"Skipping {len(legacy)} legacy file(s) (.doc/.ppt not supported): "
            f"{sample}{more}. Save them as .docx/.pptx to include."
        )

    if not files:
        if progress:
            msg = f"No supported files found under '{folder}'."
            if legacy:
                msg += f" Found {len(legacy)} legacy .doc/.ppt files (skipped)."
            await progress(0, 1, msg)
        store.set_meta("last_indexed_at", now_iso())
        store.commit()
        store.close()
        return result

    total_estimate = _estimate_chunks(files)
    if progress:
        await progress(
            0, total_estimate,
            f"Found {len(files)} files (~{total_estimate} chunks estimated). "
            f"Loading embedding model — first file slower if model not cached..."
        )

    tokenizer = embedder.get_tokenizer()
    if progress:
        await progress(
            0, total_estimate,
            f"Embedding model ready. Beginning indexing of {len(files)} files."
        )
    rate_window: collections.deque[float] = collections.deque(maxlen=10)
    t0 = time.time()
    done = 0

    try:
        for f in files:
            f_t0 = time.time()
            try:
                file_hash = _sha1_of_file(f)
                action, display_name = _classify_file(store, f, file_hash)

                if action == "skip_same_content":
                    result.files_skipped_same_content += 1
                    continue

                if action == "update":
                    store.delete_chunks_for_file(str(f))

                if action == "new_renamed":
                    result.files_renamed_due_to_collision += 1
                    if progress:
                        await progress(
                            done, total_estimate,
                            f"Renamed {f.name} → {display_name} (basename collision, different content)."
                        )

                n_chunks = await _embed_file(store, f, file_hash, display_name, tokenizer)
                if n_chunks == 0:
                    continue
                store.commit()

                result.files_indexed += 1
                result.chunks_created += n_chunks
                done += n_chunks

                rate_window.append(n_chunks / max(time.time() - f_t0, 1e-3))
                smoothed = sum(rate_window) / len(rate_window)
                eta = max(0, total_estimate - done) / max(smoothed, 1.0)
                if progress:
                    await progress(
                        done, total_estimate,
                        f"{display_name} done ({done}/{total_estimate} chunks). "
                        f"~{format_eta(eta)} remaining."
                    )

            except Exception as e:
                logger.exception("failed to index %s", f)
                result.files_failed += 1
                if progress:
                    await progress(done, total_estimate, f"Skipped {f.name}: {e}")

        store.set_meta("last_indexed_at", now_iso())
        store.commit()
    finally:
        store.close()

    result.duration_sec = round(time.time() - t0, 2)
    return result


async def reindex_project(
    project_name: str,
    progress: ProgressCb | None = None,
) -> ReindexResult:
    """Re-walk every source folder this project knows about. SHA1-incremental
    over the union: skip unchanged, re-embed changed, drop chunks for files no
    longer present in any source folder. Source folders that have been deleted
    from disk are skipped with a warning, not treated as "all files removed".
    """
    store = ProjectStore(project_name)
    try:
        sources = store.get_source_paths()
        if not sources:
            raise ValueError(
                f"Project '{project_name}' has no source paths recorded. "
                "Was it created by index_project?"
            )

        if progress:
            await progress(0, 1, f"Project '{project_name}' has {len(sources)} source folder(s). "
                                  f"Verifying which still exist on disk...")

        live_folders: list[Path] = []
        result = ReindexResult()
        for sp in sources:
            p = Path(sp)
            if p.exists() and p.is_dir():
                live_folders.append(p)
            else:
                result.missing_source_paths.append(sp)

        if not live_folders:
            store.close()
            raise FileNotFoundError(
                f"None of the source folders for project '{project_name}' exist on disk anymore: "
                f"{sources}. The existing index data is preserved; only re-walking is impossible."
            )

        if progress:
            await progress(0, 1, f"Walking {len(live_folders)} live folder(s) to find current files...")

        # Build the union of currently-present files across all live folders.
        current_files: list[Path] = []
        seen_paths: set[str] = set()
        for folder in live_folders:
            for f in _list_supported_files(folder):
                key = str(f)
                if key not in seen_paths:
                    seen_paths.add(key)
                    current_files.append(f)

        if progress:
            await progress(0, 1, f"Found {len(current_files)} files. Hashing for change detection...")

        indexed_paths = set(store.list_indexed_files())
        t0 = time.time()

        # Drop files that used to be tracked but are no longer present in any
        # *live* source folder. (Files under a missing source path are left alone.)
        live_prefixes = [str(p) for p in live_folders]

        def _under_live_source(path: str) -> bool:
            return any(path.startswith(prefix) for prefix in live_prefixes)

        for dp in indexed_paths - seen_paths:
            if _under_live_source(dp):
                store.delete_chunks_for_file(dp)
                result.deleted += 1
        if result.deleted:
            store.commit()

        to_process: list[tuple[Path, str, str, bool]] = []  # (path, sha1, display_name, is_update)
        for f in current_files:
            sha = _sha1_of_file(f)
            existing = store.get_file_record(str(f))
            if existing is None:
                action, display_name = _classify_file(store, f, sha)
                if action == "skip_same_content":
                    # Same SHA1 already in project (under a different path) — treat as unchanged.
                    result.unchanged += 1
                    continue
                to_process.append((f, sha, display_name, False))
            elif existing[0] != sha:
                to_process.append((f, sha, existing[3], True))
            else:
                result.unchanged += 1

        if not to_process:
            store.set_meta("last_indexed_at", now_iso())
            store.commit()
            result.duration_sec = round(time.time() - t0, 2)
            return result

        total_estimate = _estimate_chunks([p for p, *_ in to_process])
        if progress:
            new_n = sum(1 for x in to_process if not x[3])
            chg_n = sum(1 for x in to_process if x[3])
            warn = ""
            if result.missing_source_paths:
                warn = f" Warning: {len(result.missing_source_paths)} source folder(s) missing on disk; skipped."
            await progress(
                0, total_estimate,
                f"Reindex project '{project_name}': {new_n} new, {chg_n} changed, "
                f"{result.unchanged} unchanged, {result.deleted} deleted." + warn
            )

        tokenizer = embedder.get_tokenizer()
        rate_window: collections.deque[float] = collections.deque(maxlen=10)
        done = 0
        for f, sha, display_name, is_update in to_process:
            f_t0 = time.time()
            try:
                if is_update:
                    store.delete_chunks_for_file(str(f))
                n_chunks = await _embed_file(store, f, sha, display_name, tokenizer)
                if n_chunks == 0:
                    continue
                store.commit()
                done += n_chunks
                if is_update:
                    result.updated += 1
                else:
                    result.added += 1

                rate_window.append(n_chunks / max(time.time() - f_t0, 1e-3))
                smoothed = sum(rate_window) / len(rate_window)
                eta = max(0, total_estimate - done) / max(smoothed, 1.0)
                if progress:
                    await progress(
                        done, total_estimate,
                        f"{display_name} done ({done}/{total_estimate} chunks). "
                        f"~{format_eta(eta)} remaining."
                    )
            except Exception as e:
                logger.exception("reindex failed for %s", f)
                if progress:
                    await progress(done, total_estimate, f"Skipped {f.name}: {e}")

        store.set_meta("last_indexed_at", now_iso())
        store.commit()
        result.duration_sec = round(time.time() - t0, 2)
        return result
    finally:
        store.close()
