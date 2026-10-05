"""Background jobs for long tool calls (clients' request timeouts).

MCP clients give up on a tool call after a fixed time (LibreChat: 30 s by
default), while some calls legitimately take longer — research_route (~75 s),
delegate (minutes). The gateway therefore answers within TOOL_CALL_SYNC_SECS
(default 25): if the call has finished, the result; otherwise
`{"status": "running", "job_id": …}` while the call keeps running in the
background. `get_job_result` then waits a bit and returns the result (or
"running" again). Works with any client, whatever its timeout.

Jobs live in memory (lost on restart), belong to the API key that started them,
and are dropped JOB_TTL_S after completion or when fetched.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import mcp.types as types

JOB_TTL_S = 1800.0
MAX_JOBS_PER_KEY = 20


def sync_budget() -> float:
    try:
        return float(os.environ.get("TOOL_CALL_SYNC_SECS", "25"))
    except ValueError:
        return 25.0


@dataclass
class Job:
    job_id: str
    key_id: str
    tool: str
    task: asyncio.Task
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def _gc(self) -> None:
        now = time.monotonic()
        for jid, j in list(self._jobs.items()):
            if j.finished is not None and now - j.finished > JOB_TTL_S:
                del self._jobs[jid]
            elif j.finished is None and now - j.started > 4 * JOB_TTL_S:  # runaway
                j.task.cancel()
                del self._jobs[jid]

    async def run(self, key_id: str, tool: str, coro: Any, on_background: Any = None) -> types.CallToolResult:
        """Run `coro`; return its result if it finishes within the sync budget,
        else a 'running' handle (the call continues in the background)."""
        task = asyncio.ensure_future(coro)
        done, _ = await asyncio.wait({task}, timeout=sync_budget())
        if done:
            return task.result()
        self._gc()
        if sum(1 for j in self._jobs.values() if j.key_id == key_id and j.finished is None) >= MAX_JOBS_PER_KEY:
            task.cancel()
            return _text({"status": "error", "error": "too many running background jobs for this key"}, error=True)
        if on_background is not None:
            on_background(tool)
        job = Job(secrets.token_hex(8), key_id, tool, task)
        task.add_done_callback(lambda _t, j=job: setattr(j, "finished", time.monotonic()))
        self._jobs[job.job_id] = job
        return self._running(job)

    async def result(self, key_id: str, job_id: str, wait_s: float) -> types.CallToolResult:
        job = self._jobs.get(job_id)
        if job is None or job.key_id != key_id:
            return _text({"status": "unknown", "job_id": job_id,
                          "error": "no such job (finished results are kept 30 min; jobs are lost on gateway restart)"},
                         error=True)
        if not job.task.done():
            await asyncio.wait({job.task}, timeout=max(0.0, min(wait_s, sync_budget())))
        if not job.task.done():
            return self._running(job)
        del self._jobs[job_id]
        try:
            return job.task.result()
        except Exception as e:
            return _text({"status": "error", "job_id": job_id, "error": f"{type(e).__name__}: {e}"}, error=True)

    @staticmethod
    def _running(job: Job) -> types.CallToolResult:
        return _text({
            "status": "running", "job_id": job.job_id, "tool": job.tool,
            "elapsed_s": round(time.monotonic() - job.started),
            "hint": f'Still working. Call get_job_result with {{"job_id": "{job.job_id}"}} to wait for the '
                    f"result (repeat while status is running).",
        })

    async def aclose(self) -> None:
        for j in self._jobs.values():
            j.task.cancel()
        self._jobs.clear()


def _text(obj: dict, error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(isError=error, content=[types.TextContent(type="text", text=json.dumps(obj))])
