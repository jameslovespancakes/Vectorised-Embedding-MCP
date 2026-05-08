"""Smoke test: background job pattern.

Drives the MCP tool functions directly (not through a real MCP transport) to
verify:
  • index_project returns a job_id immediately, status='pending'/'running'.
  • Background task runs to completion.
  • await_index blocks then returns the latest job state.
  • index_status is instant.
  • list_jobs reflects history.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE_DIR = HERE / "sample_docs"
PROJECT = "smoke_test_jobs"

from vectorise_mcp import jobs, store
from vectorise_mcp.server import (
    await_index,
    index_project,
    index_status,
    list_jobs as srv_list_jobs,
)


class _FakeCtx:
    """Stand-in for MCP Context — just absorbs progress/info calls."""
    async def report_progress(self, **kw): pass
    async def info(self, msg): pass


async def main() -> int:
    db = store.project_path(PROJECT)
    if db.exists():
        db.unlink()

    ctx = _FakeCtx()

    print("=== STEP 1: launch index_project ===")
    t0 = time.time()
    launch = await index_project(
        folder_path=str(SAMPLE_DIR),
        ctx=ctx,
        project=PROJECT,
        mode="auto",
    )
    launch_elapsed = time.time() - t0
    print(f"   launch took {launch_elapsed:.2f}s (should be <1s)")
    safe = {k: v for k, v in launch.items() if k != "user_message"}
    print(f"   {safe}")
    print(f"   user_message present: {'user_message' in launch} ({len(launch.get('user_message',''))} chars)")
    assert "job_id" in launch
    assert launch["status"] in ("pending", "running")
    job_id = launch["job_id"]

    print("\n=== STEP 2: index_status (instant) ===")
    snap1 = await index_status(job_id=job_id)
    print(f"   {snap1.get('status')}: {snap1.get('message')}")

    print("\n=== STEP 3: await_index loop until completion ===")
    iterations = 0
    while True:
        iterations += 1
        latest = await await_index(job_id=job_id, timeout_sec=5.0)
        print(f"   iter {iterations} status={latest['status']} "
              f"progress={latest['progress']}/{latest['total']} :: {latest['message']}")
        if latest["status"] in ("completed", "failed"):
            break
        if iterations > 30:
            print("   too many iterations — bailing")
            break

    assert latest["status"] == "completed", latest
    print(f"\n   final result: {latest.get('result')}")

    print("\n=== STEP 4: list_jobs ===")
    js = await srv_list_jobs(active_only=False)
    print(f"   {len(js)} job(s):")
    for j in js:
        print(f"     {j['id'][:8]}... kind={j['kind']} status={j['status']} project={j['project']}")

    print("\n=== STEP 5: index_status after completion ===")
    final = await index_status(job_id=job_id)
    print(f"   final status={final['status']} duration={final.get('result', {}).get('duration_sec')}")

    print("\n=== CLEANUP ===")
    print(store.delete_project(PROJECT))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
