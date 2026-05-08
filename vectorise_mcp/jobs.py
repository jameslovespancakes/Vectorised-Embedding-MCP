"""In-process background job registry for long-running indexing operations.

MCP request timeouts (Claude Desktop's client-side cap) cancel any tool call that
runs too long, even when progress notifications are streaming. Solution: tools
return a `job_id` immediately and hand the actual work to a DAEMON THREAD with
its own asyncio loop. Callers poll `index_status` (instant) or call
`await_index` (blocks up to a small timeout) to drive the job to completion
across many short tool calls.

THREAD, not asyncio task: sentence-transformers.encode and OCR engines are
synchronous CPU-bound Python calls. Running them on the MCP server's asyncio
loop blocks all other tool calls (including the response to the launching
index_project call itself). Putting them in a separate thread keeps the main
asyncio loop free to serve MCP requests.

Jobs live in-process for the lifetime of the MCP server. Restart wipes them.
That's fine — the indexing result (chunks + embeddings) is durably committed to
the project's SQLite db incrementally as files complete, so a restart only loses
the *job summary*, not the indexed data.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: str                # "index" | "reindex"
    project: str             # which project this job targets
    status: str              # "pending" | "running" | "completed" | "failed" | "cancelled"
    progress: int = 0
    total: int = 1
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    # Internal:
    _thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    _done_event: threading.Event | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        # Manually enumerate to skip non-serializable internals (_thread, _done_event).
        return {
            "id": self.id,
            "kind": self.kind,
            "project": self.project,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
MAX_RETAINED_JOBS = 100


def create_job(kind: str, project: str) -> Job:
    """Create a new job in 'pending' state. Caller fills in the worker Thread."""
    job = Job(
        id=str(uuid.uuid4()),
        kind=kind,
        project=project,
        status="pending",
        started_at=_now_iso(),
        updated_at=_now_iso(),
    )
    job._done_event = threading.Event()
    with _jobs_lock:
        _jobs[job.id] = job
        _evict_if_overflow()
    return job


def _evict_if_overflow() -> None:
    """Keep at most MAX_RETAINED_JOBS jobs. Drop oldest completed/failed first."""
    if len(_jobs) <= MAX_RETAINED_JOBS:
        return
    finished = [j for j in _jobs.values() if j.status in ("completed", "failed", "cancelled")]
    finished.sort(key=lambda j: j.completed_at or j.started_at)
    for j in finished:
        if len(_jobs) <= MAX_RETAINED_JOBS:
            break
        _jobs.pop(j.id, None)


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def list_jobs(active_only: bool = False) -> list[Job]:
    jobs = list(_jobs.values())
    if active_only:
        jobs = [j for j in jobs if j.status in ("pending", "running")]
    jobs.sort(key=lambda j: j.started_at, reverse=True)
    return jobs


def make_progress_callback(job: Job) -> Callable[[int, int, str], Awaitable[None]]:
    """Build a progress callback that updates the job's state.

    Indexer expects an async callback (legacy from when progress was streamed
    over MCP). We satisfy the contract with an async wrapper, but the body is
    pure dict updates — atomic under the GIL — so it's safe to call from the
    worker thread's loop. The MAIN asyncio loop (MCP server) reads job state
    via simple attribute access, also GIL-safe.
    """
    async def _cb(progress: int, total: int, message: str) -> None:
        job.progress = progress
        job.total = total
        job.message = message
        job.updated_at = _now_iso()
    return _cb


def mark_running(job: Job) -> None:
    job.status = "running"
    job.updated_at = _now_iso()


def mark_completed(job: Job, result: dict[str, Any]) -> None:
    job.status = "completed"
    job.result = result
    job.completed_at = _now_iso()
    job.updated_at = job.completed_at
    if job._done_event is not None:
        job._done_event.set()
    _maybe_notify_completion(job, result)


def mark_failed(job: Job, error: str) -> None:
    job.status = "failed"
    job.error = error
    job.completed_at = _now_iso()
    job.updated_at = job.completed_at
    if job._done_event is not None:
        job._done_event.set()
    _maybe_notify_failure(job, error)


def _maybe_notify_completion(job: Job, result: dict[str, Any]) -> None:
    try:
        from vectorise_mcp import notifier
        files = result.get("files_indexed") or result.get("added") or 0
        chunks = result.get("chunks_created") or 0
        body_parts = [f"Project '{job.project}' {job.kind} complete."]
        if files or chunks:
            extras = []
            if files:
                extras.append(f"{files} file(s)")
            if chunks:
                extras.append(f"{chunks} new chunk(s)")
            body_parts.append(f"({', '.join(extras)})")
        body_parts.append("Return to Claude to query.")
        notifier.notify("vectorise-mcp: indexing done", " ".join(body_parts))
    except Exception:
        logger.debug("completion notification failed", exc_info=True)


def _maybe_notify_failure(job: Job, error: str) -> None:
    try:
        from vectorise_mcp import notifier
        notifier.notify(
            "vectorise-mcp: indexing failed",
            f"Project '{job.project}' job failed: {error[:160]}",
        )
    except Exception:
        logger.debug("failure notification failed", exc_info=True)


async def wait_for_job(job: Job, timeout: float) -> bool:
    """Wait up to `timeout` seconds for job to complete. Returns True if it
    finished within the window, False on timeout. Safe to call repeatedly.

    Uses asyncio.to_thread to wait on a threading.Event without blocking the
    main asyncio loop.
    """
    if job._done_event is None:
        return job.status in ("completed", "failed", "cancelled")
    return await asyncio.to_thread(job._done_event.wait, timeout)
