"""Per-project store: SQLite + sqlite-vec (semantic) + FTS5 (keyword) + RRF (hybrid).

One .db file per project under ~/.vectorise-mcp/. Embeddings are L2-normalized,
so sqlite-vec L2 distance preserves cosine ranking; we convert to cosine similarity
in [0, 1] for the score. Metadata filtering applied post-retrieval over a larger
candidate pool — caller should raise n_per_side when filters are restrictive.

A "project" is a named, persistent index. It tracks one or more source folders
(`source_paths`), the file SHA1s indexed, and chunk text + embeddings + FTS rows.
Projects survive across runs. Source folders may be deleted from disk after
indexing without breaking search (only affects future `reindex_project` runs).
"""

from __future__ import annotations

import fnmatch
import json
import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sqlite_vec

EMBEDDING_DIM = 384  # bge-small-en-v1.5

STORAGE_DIR = Path.home() / ".vectorise-mcp"

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_\-]")
_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_project_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name.strip()).strip("_")
    return cleaned or "project"


def project_path(name: str) -> Path:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR / f"{sanitize_project_name(name)}.db"


def project_exists(name: str) -> bool:
    return project_path(name).exists()


def _serialize_vector(vec) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _fts_query_from(query: str) -> str:
    """Sanitize free-form text into an FTS5 MATCH expression. OR all word tokens."""
    tokens = _FTS_TOKEN_RE.findall(query)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


@dataclass
class Chunk:
    text: str
    page: int | None
    chunk_index: int
    display_name: str | None = None  # auto-resolved on insert if None


@dataclass
class SearchFilter:
    """Optional metadata filter applied post-retrieval to hybrid search results.

    All fields ANDed. None means no constraint on that field. Restrictive filters
    benefit from raising n_per_side so enough survivors remain for reranking.
    """
    file_glob: str | None = None
    subdirectory: str | None = None
    page_min: int | None = None
    page_max: int | None = None
    min_similarity: float = 0.0

    def is_active(self) -> bool:
        return any([
            self.file_glob,
            self.subdirectory,
            self.page_min is not None,
            self.page_max is not None,
            self.min_similarity > 0.0,
        ])

    def matches(self, hit: "SearchHit") -> bool:
        if self.file_glob:
            if not fnmatch.fnmatch(hit.display_name, self.file_glob):
                if not fnmatch.fnmatch(Path(hit.source_file).name, self.file_glob):
                    return False
        if self.subdirectory and self.subdirectory not in hit.source_file:
            return False
        if self.page_min is not None and (hit.page is None or hit.page < self.page_min):
            return False
        if self.page_max is not None and (hit.page is None or hit.page > self.page_max):
            return False
        if self.min_similarity > 0.0 and hit.score < self.min_similarity:
            return False
        return True


@dataclass
class SearchHit:
    chunk_id: int
    text: str
    source_file: str    # absolute path on disk (may not exist anymore)
    display_name: str   # human-friendly name; suffixed if collision happened
    page: int | None
    chunk_index: int
    distance: float

    @property
    def score(self) -> float:
        cos_sim = 1.0 - (self.distance * self.distance) / 2.0
        return max(0.0, min(1.0, cos_sim))


class ProjectStore:
    """SQLite + sqlite-vec + FTS5 wrapper for one project."""

    def __init__(self, name: str):
        self.name = sanitize_project_name(name)
        self.path = project_path(self.name)
        self.conn = sqlite3.connect(self.path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()
        self._migrate_legacy()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(f"""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_display_name ON chunks(display_name);
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                sha1 TEXT NOT NULL,
                mtime REAL NOT NULL,
                chunk_count INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_files_display_name ON files(display_name);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding FLOAT[{EMBEDDING_DIM}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );
        """)
        self.conn.commit()

    def _migrate_legacy(self) -> None:
        """Backfill display_name on tables that pre-date the column."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(chunks)")
        chunk_cols = {row[1] for row in cur.fetchall()}
        if "display_name" not in chunk_cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN display_name TEXT")
            cur.execute(
                "UPDATE chunks SET display_name = "
                "substr(file_path, max(instr(file_path, '/'), instr(file_path, '\\\\')) + 1)"
            )
        cur.execute("PRAGMA table_info(files)")
        file_cols = {row[1] for row in cur.fetchall()}
        if "display_name" not in file_cols:
            cur.execute("ALTER TABLE files ADD COLUMN display_name TEXT")
            cur.execute(
                "UPDATE files SET display_name = "
                "substr(path, max(instr(path, '/'), instr(path, '\\\\')) + 1)"
            )
        self.conn.commit()

    # ---- meta ----

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # ---- source paths (multi-folder support) ----

    def get_source_paths(self) -> list[str]:
        raw = self.get_meta("source_paths")
        if raw:
            try:
                return list(json.loads(raw))
            except json.JSONDecodeError:
                pass
        legacy = self.get_meta("source_path")
        if legacy:
            return [legacy]
        return []

    def add_source_path(self, path: str) -> None:
        paths = self.get_source_paths()
        if path not in paths:
            paths.append(path)
        self.set_meta("source_paths", json.dumps(paths))
        self.set_meta("source_path", paths[0])  # legacy mirror, first path

    def has_source_path(self, path: str) -> bool:
        return path in self.get_source_paths()

    # ---- file tracking ----

    def get_file_record(self, path: str) -> tuple[str, float, int, str] | None:
        return self.conn.execute(
            "SELECT sha1, mtime, chunk_count, display_name FROM files WHERE path = ?",
            (path,),
        ).fetchone()

    def find_file_by_hash(self, sha1: str) -> str | None:
        row = self.conn.execute(
            "SELECT path FROM files WHERE sha1 = ? LIMIT 1", (sha1,)
        ).fetchone()
        return row[0] if row else None

    def display_name_in_use(self, display_name: str, exclude_path: str | None = None) -> bool:
        if exclude_path is None:
            row = self.conn.execute(
                "SELECT 1 FROM files WHERE display_name = ? LIMIT 1", (display_name,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT 1 FROM files WHERE display_name = ? AND path != ? LIMIT 1",
                (display_name, exclude_path),
            ).fetchone()
        return row is not None

    def resolve_display_name(self, file_path: str, sha1: str) -> str:
        """Return a non-colliding display_name for a file about to be indexed.

        Same basename + same SHA1 elsewhere → caller should skip (not our job here).
        Same basename + different SHA1 → suffix `_2`, `_3`, ... until unique.
        """
        basename = Path(file_path).name
        if not self.display_name_in_use(basename, exclude_path=file_path):
            return basename
        stem = Path(basename).stem
        suffix = Path(basename).suffix
        n = 2
        while True:
            candidate = f"{stem}_{n}{suffix}"
            if not self.display_name_in_use(candidate, exclude_path=file_path):
                return candidate
            n += 1

    def upsert_file_record(
        self,
        path: str,
        sha1: str,
        mtime: float,
        chunk_count: int,
        display_name: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO files(path, sha1, mtime, chunk_count, display_name, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "sha1 = excluded.sha1, mtime = excluded.mtime, "
            "chunk_count = excluded.chunk_count, display_name = excluded.display_name, "
            "indexed_at = excluded.indexed_at",
            (path, sha1, mtime, chunk_count, display_name, now_iso()),
        )

    def list_indexed_files(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT path FROM files")]

    # ---- chunks ----

    def delete_chunks_for_file(self, file_path: str) -> int:
        cur = self.conn.cursor()
        ids = [r[0] for r in cur.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (file_path,)
        )]
        if not ids:
            cur.execute("DELETE FROM files WHERE path = ?", (file_path,))
            return 0
        placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})", ids)
        cur.execute(f"DELETE FROM fts_chunks WHERE rowid IN ({placeholders})", ids)
        cur.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        cur.execute("DELETE FROM files WHERE path = ?", (file_path,))
        return len(ids)

    def insert_chunks(
        self,
        file_path: str,
        file_hash: str,
        display_name: str,
        chunks: list[Chunk],
        embeddings: Iterable,
    ) -> list[int]:
        cur = self.conn.cursor()
        ts = now_iso()
        ids: list[int] = []
        for ch, emb in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks(file_path, file_hash, display_name, page, chunk_index, "
                "text, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (file_path, file_hash, display_name, ch.page, ch.chunk_index, ch.text, ts),
            )
            chunk_id = cur.lastrowid
            ids.append(chunk_id)
            cur.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                (chunk_id, _serialize_vector(emb)),
            )
            cur.execute(
                "INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)",
                (chunk_id, ch.text),
            )
        return ids

    # ---- retrieval ----

    def vector_search(self, query_embedding, k: int) -> list[SearchHit]:
        rows = self.conn.execute(
            """
            SELECT v.rowid, c.text, c.file_path, c.display_name, c.page, c.chunk_index, v.distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_vector(query_embedding), k),
        ).fetchall()
        return [
            SearchHit(chunk_id=r[0], text=r[1], source_file=r[2], display_name=r[3],
                      page=r[4], chunk_index=r[5], distance=r[6])
            for r in rows
        ]

    def keyword_search(self, query: str, k: int) -> list[SearchHit]:
        match_expr = _fts_query_from(query)
        try:
            rows = self.conn.execute(
                """
                SELECT c.id, c.text, c.file_path, c.display_name, c.page, c.chunk_index
                FROM fts_chunks f
                JOIN chunks c ON c.id = f.rowid
                WHERE fts_chunks MATCH ?
                ORDER BY bm25(fts_chunks)
                LIMIT ?
                """,
                (match_expr, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            SearchHit(chunk_id=r[0], text=r[1], source_file=r[2], display_name=r[3],
                      page=r[4], chunk_index=r[5], distance=0.0)
            for r in rows
        ]

    def hybrid_search(
        self,
        query_embedding,
        query_text: str,
        n_per_side: int = 75,
        filter: SearchFilter | None = None,
    ) -> list[SearchHit]:
        vec_hits = self.vector_search(query_embedding, k=n_per_side)
        kw_hits = self.keyword_search(query_text, k=n_per_side)
        merged = _rrf_merge(vec_hits, kw_hits)
        if filter and filter.is_active():
            merged = [h for h in merged if filter.matches(h)]
        return merged

    # ---- stats ----

    def stats(self) -> dict:
        n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        size_mb = self.path.stat().st_size / (1024 * 1024) if self.path.exists() else 0
        return {
            "doc_count": n_files,
            "chunk_count": n_chunks,
            "size_mb": round(size_mb, 2),
        }

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def _rrf_merge(
    vec_hits: list[SearchHit],
    kw_hits: list[SearchHit],
    k_constant: int = 60,
) -> list[SearchHit]:
    scores: dict[int, float] = {}
    by_id: dict[int, SearchHit] = {}
    for rank, hit in enumerate(vec_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k_constant + rank)
        by_id[hit.chunk_id] = hit
    for rank, hit in enumerate(kw_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k_constant + rank)
        by_id.setdefault(hit.chunk_id, hit)
    return sorted(by_id.values(), key=lambda h: scores[h.chunk_id], reverse=True)


def list_projects() -> list[dict]:
    if not STORAGE_DIR.exists():
        return []
    out = []
    for db_file in sorted(STORAGE_DIR.glob("*.db")):
        name = db_file.stem
        try:
            store = ProjectStore(name)
            sources = store.get_source_paths()
            entry = {
                "name": name,
                "source_paths": sources,
                "source_path": sources[0] if sources else "",
                "indexed_at": store.get_meta("last_indexed_at") or "",
                **store.stats(),
            }
            store.close()
            out.append(entry)
        except Exception as e:
            out.append({"name": name, "error": str(e)})
    return out


def delete_project(name: str) -> dict:
    path = project_path(name)
    if not path.exists():
        return {"deleted": False, "name": name, "reason": "project not found"}
    size_mb = path.stat().st_size / (1024 * 1024)
    path.unlink()
    return {"deleted": True, "name": name, "freed_mb": round(size_mb, 2)}
