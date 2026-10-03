"""Sticky upstream sessions for stateful servers.

The gateway normally opens a fresh upstream session per tool call (stateless,
see backends.py). Servers that keep state per MCP session — Playwright's open
page, for one — lose it between calls that way. For servers flagged
`stateful: true` in the registry, this pool keeps ONE upstream session per
(downstream MCP session, server) alive and routes that session's calls to it,
one at a time (ordering matters for a browser).

Each pooled session lives in its own background task (the MCP client's
transport contexts must be entered and exited in the same task); request
handlers only borrow the ClientSession. A pooled session is closed when it has
been idle for STICKY_SESSION_IDLE_SECS (default 600), when its downstream
session is garbage-collected, when the pool is over STICKY_MAX_SESSIONS (least
recently used first), on an upstream failure, or on gateway shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from typing import Any

import mcp.types as types
from mcp.shared.exceptions import McpError

from gateway.backends import open_upstream_session
from gateway.registry import ServerConfig

logger = logging.getLogger(__name__)

OPEN_TIMEOUT_S = 30.0
REAP_EVERY_S = 30.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class _Entry:
    """One held-open upstream session, owned by its own task."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.session: Any = None
        self.ready = asyncio.Event()
        self.closing = asyncio.Event()
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()
        self.error: BaseException | None = None
        self.task = asyncio.create_task(self._run(), name=f"sticky:{cfg.server_id}")

    async def _run(self) -> None:
        try:
            async with open_upstream_session(self.cfg) as s:
                self.session = s
                self.ready.set()
                await self.closing.wait()
        except BaseException as e:  # incl. cancellation on shutdown
            self.error = e
            if not isinstance(e, asyncio.CancelledError):
                logger.warning("sticky session to %s ended: %s", self.cfg.server_id, e)
        finally:
            self.session = None
            self.ready.set()

    @property
    def alive(self) -> bool:
        return not self.task.done() and not self.closing.is_set()

    def close(self) -> None:
        self.closing.set()


class StickySessionPool:
    def __init__(self, idle_secs: float | None = None, max_sessions: int | None = None) -> None:
        self.idle_secs = idle_secs if idle_secs is not None else _env_float("STICKY_SESSION_IDLE_SECS", 600)
        self.max_sessions = max_sessions if max_sessions is not None else int(_env_float("STICKY_MAX_SESSIONS", 20))
        self._entries: dict[tuple[int, str], _Entry] = {}
        self._tracked: set[int] = set()
        self._reaper: asyncio.Task | None = None

    def _track(self, downstream: Any) -> int:
        """Key for a downstream session; close its upstream sessions when it's collected."""
        key = id(downstream)
        if key not in self._tracked:
            self._tracked.add(key)
            try:
                weakref.finalize(downstream, self._forget, key)
            except TypeError:  # not weak-referenceable: idle TTL still applies
                pass
        return key

    def _forget(self, key: int) -> None:
        self._tracked.discard(key)
        for k in [k for k in self._entries if k[0] == key]:
            self._entries.pop(k).close()

    async def _acquire(self, key: int, cfg: ServerConfig) -> _Entry:
        k = (key, cfg.server_id)
        entry = self._entries.get(k)
        if entry is None or not entry.alive:
            self._evict_if_full()
            entry = self._entries[k] = _Entry(cfg)
            logger.info("sticky session opened: %s (%d open)", cfg.server_id, len(self._entries))
            if self._reaper is None or self._reaper.done():
                self._reaper = asyncio.create_task(self._reap())
        await asyncio.wait_for(entry.ready.wait(), OPEN_TIMEOUT_S)
        if entry.session is None:
            self._entries.pop(k, None)
            raise RuntimeError(f"could not open a session to {cfg.server_id}: {entry.error}")
        return entry

    def _evict_if_full(self) -> None:
        live = sorted(self._entries.items(), key=lambda kv: kv[1].last_used)
        while len(live) >= self.max_sessions:
            k, e = live.pop(0)
            self._entries.pop(k, None)
            e.close()

    async def _reap(self) -> None:
        while self._entries:
            await asyncio.sleep(min(REAP_EVERY_S, max(self.idle_secs / 2, 1)))
            now = time.monotonic()
            for k, e in list(self._entries.items()):
                if not e.alive or (now - e.last_used > self.idle_secs and not e.lock.locked()):
                    self._entries.pop(k, None)
                    e.close()
                    logger.info("sticky session closed (idle): %s", e.cfg.server_id)

    async def call_tool(self, downstream: Any, cfg: ServerConfig, name: str,
                        arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Call through the downstream session's sticky upstream session (serialized).
        If the held session's transport has died (e.g. the server restarted), drop it
        and retry once on a fresh one; protocol errors (McpError) are not retried."""
        key = self._track(downstream)
        entry = await self._acquire(key, cfg)
        try:
            return await self._call(entry, name, arguments)
        except McpError:
            raise
        except Exception as e:
            self._drop(key, cfg, entry)
            logger.info("sticky session to %s failed (%s); retrying on a new one", cfg.server_id, e)
        entry = await self._acquire(key, cfg)
        try:
            return await self._call(entry, name, arguments)
        except Exception:
            self._drop(key, cfg, entry)
            raise

    @staticmethod
    async def _call(entry: _Entry, name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        async with entry.lock:
            entry.last_used = time.monotonic()
            try:
                return await entry.session.call_tool(name, arguments)
            finally:
                entry.last_used = time.monotonic()

    def _drop(self, key: int, cfg: ServerConfig, entry: _Entry) -> None:
        if self._entries.get((key, cfg.server_id)) is entry:
            del self._entries[(key, cfg.server_id)]
        entry.close()

    @property
    def open_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.alive)

    async def aclose(self) -> None:
        entries = list(self._entries.values())
        self._entries.clear()
        for e in entries:
            e.close()
        if entries:
            await asyncio.wait([e.task for e in entries], timeout=10)
        if self._reaper is not None:
            self._reaper.cancel()
