"""FastMCP stdio server. Project-vectorisation tools for Claude Desktop.

LONG OPS USE BACKGROUND JOBS.

Indexing 100M+ token corpora takes minutes-to-hours. MCP request timeouts
(Claude Desktop's hard cap) cancel any tool call that runs that long, even with
streaming progress notifications. So `index_project` and `reindex_project`
return a `job_id` IMMEDIATELY and hand the actual work to an asyncio background
task. Claude polls `index_status(job_id)` or calls `await_index(job_id)` (which
blocks up to ~25 sec, then returns) until the job completes.

Search pipeline (per call, fast — no job needed):
  1. Vector top-N (BGE) + BM25 top-N (FTS5).
  2. Reciprocal Rank Fusion → up to 2N candidates.
  3. Optional metadata filter (file/path/page/similarity).
  4. Cross-encoder rerank → top-K.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from vectorise_mcp import embedder, indexer, jobs, ocr, reranker, store

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL = 75
VALID_MODES = ("auto", "replace", "append", "fail")
AWAIT_DEFAULT_TIMEOUT_SEC = 25.0   # safely under most MCP request timeouts

mcp = FastMCP(
    name="vectorise-mcp",
    instructions=(
        "Local hybrid (BGE + BM25 FTS5) document index with cross-encoder reranking. "
        "Documents live in named PROJECTS that persist across restarts. "
        "\n\nWORKFLOW — FIRE-AND-FORGET. Do NOT poll. Do NOT loop.\n\n"
        "(1) Before any indexing, call vectorise_list_projects and ASK the user whether to "
        "create a new project, add to an existing one, or delete some old ones.\n\n"
        "(2) Call index_project / reindex_project — returns a job_id IMMEDIATELY. "
        "**MANDATORY**: your VERY NEXT response to the user MUST quote the job_id "
        "verbatim. Use the `user_message` field of the tool result as the literal "
        "text to show the user (or paraphrase but keep the job_id visible). "
        "Failing to surface the job_id is a bug. Then STOP. Hand control back.\n\n"
        "(3) Search works on partial index. The user can ask questions about the "
        "documents at ANY point — even mid-indexing. Each file commits as it "
        "finishes, so search returns whatever's been embedded so far.\n\n"
        "(4) WHEN THE USER ASKS ANYTHING about the indexed material — even casually "
        "— silently call index_status(job_id) FIRST if a job is still active for "
        "that project, then call search. Mention indexing status briefly if it "
        "changed since last check, then deliver the search result.\n\n"
        "(5) Only call await_index if the user explicitly says 'wait until done'."
    ),
)


def _make_progress_cb(ctx: Context):
    async def _cb(progress: int, total: int, message: str) -> None:
        try:
            await ctx.report_progress(progress=progress, total=total, message=message)
            await ctx.info(message)
        except Exception:
            logger.debug("progress notification failed", exc_info=True)
    return _cb


async def _announce_start(ctx: Context, message: str) -> None:
    try:
        await ctx.report_progress(progress=0, total=1, message=message)
        await ctx.info(message)
    except Exception:
        logger.debug("startup announcement failed", exc_info=True)


def _spawn_indexer_thread(job: jobs.Job, coro_factory) -> threading.Thread:
    """Run an async indexer coroutine in a daemon thread with its own event loop.

    `coro_factory` is a zero-arg callable that returns a fresh coroutine when
    called inside the worker thread (each thread needs its own loop, so we
    can't pass a coroutine created on the main loop).
    """
    def _runner_thread():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            jobs.mark_running(job)
            try:
                result = loop.run_until_complete(coro_factory())
                from dataclasses import asdict
                res_dict = asdict(result)
                try:
                    s = store.ProjectStore(job.project)
                    res_dict.update({"source_paths": s.get_source_paths(), **s.stats()})
                    s.close()
                except Exception:
                    logger.debug("stats fetch after job failed", exc_info=True)
                jobs.mark_completed(job, res_dict)
            except Exception as e:
                logger.exception("indexer job %s failed", job.id)
                jobs.mark_failed(job, str(e))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_runner_thread, daemon=True, name=f"vec-job-{job.id[:8]}")
    t.start()
    job._thread = t
    return t


# ------------------------------------------------------------------ projects --

@mcp.tool(name="vectorise_list_projects")
async def list_projects() -> list[dict]:
    """List every persistent project on disk.

    REQUIRED: call this BEFORE any index_project call so the user can decide
    whether to add to an existing project or create a new one.

    Returns one dict per project with: name, source_paths (every folder ever
    indexed into it), doc_count, chunk_count, size_mb, indexed_at.
    """
    return store.list_projects()


@mcp.tool(name="vectorise_delete_project")
async def delete_project(project: str) -> dict:
    """Delete a project permanently. Removes all chunks, embeddings, and metadata.
    Returns the disk space freed. Source folders on disk are NOT touched.

    Destructive — only call after explicit user confirmation.
    """
    return store.delete_project(project)


# ------------------------------------------------------------------- indexing --

@mcp.tool(name="vectorise_index_project")
async def index_project(
    folder_path: str,
    ctx: Context,
    project: str | None = None,
    mode: str = "auto",
) -> dict:
    """Start indexing a folder into a named project. RETURNS IMMEDIATELY with a
    job_id — does NOT wait for indexing to finish.

    **CRITICAL RESPONSE RULE**: After this returns, your VERY NEXT message to the
    user MUST visibly contain the `job_id` returned by this tool. The tool's
    return dict includes a `user_message` field — show that field's content to
    the user, or include the job_id verbatim in your own wording. Failing to
    surface the job_id is a bug; the user has no other way to track the job.

    Then STOP. DO NOT loop on await_index. DO NOT poll status repeatedly. Hand
    control back to the user. Indexing runs in the background regardless.

    The user can search the project at any point — even while indexing is still
    running. Files commit to the DB as they finish, so a search returns whatever
    has been embedded so far. You don't need to wait for "done" to start being
    useful.

    When the user later asks "is it done?" or "check progress", call
    `index_status(job_id)` for an instant snapshot and report it. Only call
    `await_index(job_id)` if the user explicitly says to wait until completion.

    REQUIRED PRE-WORK (do not skip):
      1. Call `vectorise_list_projects` first.
      2. Show user existing projects (if any).
      3. Ask user: add to existing, create new, or delete some old projects.
      4. Only after confirmation, call this tool with the chosen `project` name
         and an explicit `mode`.

    Modes:
      • "auto" (default): new project → create. Same folder already in project →
        incremental reindex. Different folder, same project → ERROR forcing user
        to pick replace/append/different-name.
      • "replace": delete existing project entirely, re-index from scratch.
        Destructive — confirm with user first.
      • "append": keep existing data; embed new folder's files alongside.
        Identical SHA1 → skipped silently. Same basename + different content →
        auto-renamed (`name_2.ext`, `name_3.ext`, ...).
      • "fail": raise if project exists at all.

    Supports .pdf, .docx, .txt, .md, plus images + scanned PDFs (with [ocr] extra).
    Returns: {job_id, status, project, message}. Use job_id with await_index.
    """
    await _announce_start(
        ctx,
        f"Validating index_project request for '{folder_path}' (mode={mode})..."
    )

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}.")

    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    proj = project or store.sanitize_project_name(folder.name)

    # Apply mode dispatch synchronously (these checks are cheap and we want errors
    # surfaced before we hand off to a background job).
    operation_kind = "index"
    if store.project_exists(proj):
        existing = store.ProjectStore(proj)
        sources = existing.get_source_paths()
        existing.close()
        already_indexed_here = str(folder) in sources

        if mode == "fail":
            raise ValueError(
                f"Project '{proj}' already exists with sources {sources}. "
                f"mode='fail' refuses to touch it."
            )
        if mode == "replace":
            await ctx.info(f"mode=replace: deleting existing project '{proj}' before re-creating.")
            store.delete_project(proj)
        elif mode == "auto":
            if already_indexed_here:
                operation_kind = "reindex"
                await ctx.info(
                    f"Project '{proj}' already covers '{folder}'. "
                    f"Will run incremental reindex of all source paths."
                )
            else:
                raise ValueError(
                    f"Project '{proj}' already exists with source folders {sources}. "
                    f"You're trying to index a different folder ('{folder}'). "
                    f"Ask the user, then re-call with mode='replace' (wipe + start over), "
                    f"mode='append' (add this folder alongside, auto-renaming basename "
                    f"collisions), or pick a different `project` name."
                )
        elif mode == "append":
            await ctx.info(f"mode=append: '{folder}' will be added to existing project '{proj}'.")

    # Spawn the background job in a DAEMON THREAD (NOT asyncio.create_task) so
    # synchronous CPU-bound calls inside the indexer (sentence-transformers
    # encode, OCR, pypdf parsing) don't block the MCP server's event loop.
    job = jobs.create_job(kind=operation_kind, project=proj)
    progress_cb = jobs.make_progress_callback(job)
    folder_str = str(folder)

    if operation_kind == "reindex":
        def _make_coro():
            return indexer.reindex_project(project_name=proj, progress=progress_cb)
    else:
        def _make_coro():
            return indexer.index_folder_into(
                folder_path=folder_str, project_name=proj, progress=progress_cb
            )

    _spawn_indexer_thread(job, _make_coro)

    user_message = (
        f"📚 Indexing started for project **{proj}** ({operation_kind} mode={mode}).\n"
        f"\n"
        f"**Job ID:** `{job.id}`\n"
        f"\n"
        f"Running in the background — you can keep working. A desktop notification "
        f"will fire when it's done. You can also search the project right now; "
        f"results will improve as more files finish embedding."
    )
    return {
        "job_id": job.id,
        "status": job.status,
        "operation": operation_kind,
        "project": proj,
        "mode": mode,
        "user_message": user_message,
        "instruction_to_claude": (
            f"REQUIRED: your next response to the user MUST surface job_id "
            f"'{job.id}' visibly. Show them the `user_message` field above, "
            f"or include the job_id verbatim in your own wording. Then STOP. "
            f"Do not call await_index. Do not loop. Hand control back to the user."
        ),
    }


@mcp.tool(name="vectorise_reindex_project")
async def reindex_project(project: str, ctx: Context) -> dict:
    """Start a re-scan of every source folder this project tracks. RETURNS
    IMMEDIATELY with a job_id. SHA1-incremental: re-embed only new/changed
    files; drop chunks for removed files; leave unchanged files alone. Source
    folders missing on disk are skipped with a warning (existing data preserved).

    **CRITICAL RESPONSE RULE**: After this returns, your next message to the user
    MUST visibly contain the returned `job_id`. Show the tool's `user_message`
    field to the user. Same fire-and-forget pattern as index_project: report
    job_id, STOP, do not loop. The user drives subsequent status checks via
    `index_status(job_id)` when they want an update.
    """
    await _announce_start(ctx, f"Starting reindex job for project '{project}'...")

    if not store.project_exists(project):
        raise ValueError(
            f"Project '{project}' not found. Use vectorise_list_projects to see available."
        )

    job = jobs.create_job(kind="reindex", project=project)
    progress_cb = jobs.make_progress_callback(job)

    def _make_coro():
        return indexer.reindex_project(project_name=project, progress=progress_cb)

    _spawn_indexer_thread(job, _make_coro)

    user_message = (
        f"🔄 Reindex started for project **{project}**.\n"
        f"\n"
        f"**Job ID:** `{job.id}`\n"
        f"\n"
        f"Running in the background — incremental SHA1-based scan; only "
        f"new/changed files re-embed. Desktop notification on completion. "
        f"Search remains available throughout."
    )
    return {
        "job_id": job.id,
        "status": job.status,
        "operation": "reindex",
        "project": project,
        "user_message": user_message,
        "instruction_to_claude": (
            f"REQUIRED: your next response to the user MUST surface job_id "
            f"'{job.id}' visibly. Show the `user_message` field above. Then STOP."
        ),
    }


# ------------------------------------------------------- job status / await ---

@mcp.tool(name="vectorise_index_status")
async def index_status(job_id: str) -> dict:
    """INSTANT snapshot of an indexing job's current state. Use this when the
    user asks 'is it done?' or 'check progress'. Returns immediately — never
    blocks. Reports status (pending|running|completed|failed), progress (a
    rough chunks-done / chunks-estimated pair), the latest progress message
    (e.g. 'Calc 111.pdf done (412/3200 chunks). ~5m remaining.'), and the
    final result dict if completed.

    This is the recommended way to check on a job. Call it once, report the
    state to the user, and stop. Do NOT loop.
    """
    job = jobs.get_job(job_id)
    if not job:
        return {"error": f"Job '{job_id}' not found. Job registry resets on server restart."}
    return job.to_dict()


@mcp.tool(name="vectorise_await_index")
async def await_index(
    job_id: str,
    timeout_sec: float = AWAIT_DEFAULT_TIMEOUT_SEC,
) -> dict:
    """OPTIONAL blocking wait. Holds the tool call open up to `timeout_sec`
    (default 25s, hard-capped at 60s) for the job to complete, then returns
    the latest state. Use this ONLY when the user explicitly says 'wait until
    it's done' — otherwise prefer `index_status` (instant) so you don't tie
    up Claude's loading state.
    """
    job = jobs.get_job(job_id)
    if not job:
        return {"error": f"Job '{job_id}' not found."}
    if job.status not in ("pending", "running"):
        return job.to_dict()
    await jobs.wait_for_job(job, timeout=max(0.5, min(timeout_sec, 60.0)))
    return job.to_dict()


@mcp.tool(name="vectorise_list_jobs")
async def list_jobs(active_only: bool = False) -> list[dict]:
    """List indexing jobs from this server session (resets on restart).

    `active_only=True` filters to jobs still pending or running.
    """
    return [j.to_dict() for j in jobs.list_jobs(active_only=active_only)]


# ------------------------------------------------------------------- search ---

@mcp.tool(name="vectorise_search")
async def search(
    project: str,
    query: str,
    k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    file_glob: str | None = None,
    subdirectory: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    min_similarity: float = 0.0,
) -> list[dict]:
    """Hybrid semantic + keyword search over a project, with cross-encoder reranking.

    Works on the live database — searching a project that's still being indexed
    returns whatever chunks have been committed so far. You don't need to wait
    for an indexing job to finish; partial-corpus answers are valid as soon as
    any file has been embedded.

    Pipeline: vector top-N + BM25 top-N → RRF fusion → optional metadata filter
    → cross-encoder rerank → top-K. Fast (typical <500ms) — no job needed.

    Tunables (raise these for harder/broader queries — Claude is encouraged to do so):
      • k: number of results returned. Default 5; raise to 10-20 for synthesis-heavy
        questions or multi-hop reasoning.
      • candidate_pool: per-side retrieval breadth before reranking. Default 75;
        raise to 150-300 for very large projects, restrictive metadata filters,
        or when initial recall seems weak.

    Optional metadata filters (all ANDed; raise candidate_pool when filters are tight):
      • file_glob: fnmatch pattern, e.g. "*.pdf" or "Q1_*". Matches display_name
        first (de-collided), then basename.
      • subdirectory: substring of source path, e.g. "reports/2025".
      • page_min, page_max: PDF page-number range. Non-PDF chunks excluded by these.
      • min_similarity: minimum cosine similarity in [0, 1] before reranking.

    Returns top-K chunks: text, source_file (absolute path; may not exist anymore),
    display_name (suffixed when collisions happened), page (PDFs), chunk_index,
    rerank_score (higher = more relevant), vector_similarity (cosine in [0, 1]).
    """
    if not store.project_exists(project):
        raise ValueError(
            f"Project '{project}' not found. Use vectorise_list_projects to see available, "
            f"or index_project to create one."
        )

    flt = store.SearchFilter(
        file_glob=file_glob,
        subdirectory=subdirectory,
        page_min=page_min,
        page_max=page_max,
        min_similarity=min_similarity,
    )

    query_embedding = embedder.embed_query(query)
    s = store.ProjectStore(project)
    try:
        candidates = s.hybrid_search(
            query_embedding, query, n_per_side=candidate_pool, filter=flt
        )
    finally:
        s.close()

    if not candidates:
        return []

    rerank_scores = reranker.rerank(query, [c.text for c in candidates])
    ranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )[:k]

    return [
        {
            "text": hit.text,
            "source_file": hit.source_file,
            "display_name": hit.display_name,
            "page": hit.page,
            "chunk_index": hit.chunk_index,
            "rerank_score": round(float(score), 4),
            "vector_similarity": round(hit.score, 4),
        }
        for hit, score in ranked
    ]


# ------------------------------------------------------------------ runtime --

def run_stdio() -> None:
    """Run FastMCP over stdio. Eagerly warms embedder + reranker (+ OCR if available)
    so first tool call isn't blocked by model download/load.
    """
    # Reduce GIL switch interval so indexer-worker threads don't starve the main
    # asyncio loop that handles MCP requests. Default is 5ms; 1ms keeps tool
    # calls (especially status reads) responsive even under heavy embedding load.
    import sys
    sys.setswitchinterval(0.001)

    # Silence the noisy "Token indices sequence length is longer than max" warning
    # from transformers — it fires repeatedly during chunk-size measurement on
    # paragraphs without sentence punctuation, and contends with the logging lock.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*Token indices sequence length is longer.*",
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Warming embedder + reranker (downloads on first run)...")
    try:
        embedder.warm_up()
        reranker.warm_up()
        if ocr.is_available():
            logger.info("Warming OCR engine.")
            ocr.warm_up()
        logger.info("Models ready.")
    except Exception:
        logger.exception("Model warmup failed; will retry on first tool call.")
    mcp.run()
