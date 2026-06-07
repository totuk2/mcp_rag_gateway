"""Runtime liveness tracking for upstream MCP servers.

A server can go down *after* startup, but its tools remain ``status='active'`` in
the ToolDb — so retrieval would happily return tools of an unreachable upstream
that then fail when called. ``ServerHealth`` keeps an in-memory set of currently
unreachable servers, refreshed by a background probe loop, and the retriever drops
candidates whose server is down at query time.

No DB writes, no reindex: liveness is volatile runtime state, consulted directly
during retrieval (alongside the existing status filter).
"""

from __future__ import annotations

import asyncio
import logging

from gateway.backends import open_upstream_session
from gateway.registry import Registry

logger = logging.getLogger(__name__)


class ServerHealth:
    """Tracks which upstream servers are currently reachable.

    Empty down-set ⇒ everything is considered up (fail-open). stdio servers are
    treated as always-up: they're spawned per-call, so a broken one fails at call
    time anyway and probing it would just spawn/tear-down a subprocess for no
    useful signal.
    """

    def __init__(self, registry: Registry):
        self._registry = registry
        self._down: set[str] = set()

    def is_up(self, server_id: str) -> bool:
        return server_id not in self._down

    @property
    def down_servers(self) -> set[str]:
        return set(self._down)

    def _set_state(self, server_id: str, up: bool) -> None:
        was_up = server_id not in self._down
        if up and not was_up:
            self._down.discard(server_id)
            logger.info("Upstream %s is back UP", server_id)
        elif not up and was_up:
            self._down.add(server_id)
            logger.warning("Upstream %s is DOWN", server_id)

    async def _probe_one(self, server_id: str, cfg, timeout: float) -> None:
        # stdio = spawned per call; nothing meaningful to probe. Treat as up.
        if cfg.transport == "stdio":
            self._set_state(server_id, True)
            return
        try:
            async with asyncio.timeout(timeout):
                async with open_upstream_session(cfg):
                    pass  # a successful initialize handshake is enough
            self._set_state(server_id, True)
        except Exception as exc:
            logger.debug("Probe failed for %s: %s", server_id, exc)
            self._set_state(server_id, False)

    async def probe_all(self, timeout: float) -> None:
        """Probe every registered server concurrently and update liveness."""
        servers = list(self._registry.servers.items())
        if not servers:
            return
        await asyncio.gather(
            *(self._probe_one(sid, cfg, timeout) for sid, cfg in servers)
        )


async def health_loop(
    health: ServerHealth, interval: float, timeout: float
) -> None:
    """Re-probe all upstreams every ``interval`` seconds until cancelled."""
    while True:
        await asyncio.sleep(interval)
        try:
            await health.probe_all(timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Health probe cycle failed")
