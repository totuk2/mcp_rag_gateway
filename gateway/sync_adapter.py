"""Sync tool metadata from upstream MCP servers into the local ToolDb.

Uses the existing open_upstream_session() from gateway.backends to connect
to each server, calls list_tools(), and stores the results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from gateway.backends import open_upstream_session
from gateway.registry import Registry
from gateway.tool_db import ToolDb
from gateway.tool_record import ToolRecord, ToolType

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Summary of one sync run."""

    servers_synced: int = 0
    servers_failed: int = 0
    tools_added: int = 0
    tools_updated: int = 0
    tools_removed: int = 0
    errors: list[str] = field(default_factory=list)


def _infer_tool_type(tool_name: str, description: str) -> ToolType:
    """Best-guess tool type from name/description."""
    lower = f"{tool_name} {description}".lower()
    if any(w in lower for w in ("read", "get", "list", "fetch", "query", "search", "find", "check")):
        return "read"
    if any(w in lower for w in ("write", "create", "set", "update", "put", "post", "add", "set_config")):
        return "write"
    if any(w in lower for w in ("delete", "remove", "drop", "admin", "configure", "restart")):
        return "admin"
    return "action"


class SyncAdapter:
    """Pulls tool metadata from upstream MCP servers into ToolDb."""

    def __init__(self, registry: Registry, tool_db: ToolDb):
        self._registry = registry
        self._tool_db = tool_db

    async def full_sync(self) -> SyncResult:
        """Sync every server in the registry."""
        result = SyncResult()
        for server_id, cfg in self._registry.servers.items():
            try:
                seen_ids: set[str] = set()
                async with open_upstream_session(cfg) as session:
                    tr = await session.list_tools()
                    for tool in tr.tools:
                        record = self._tool_from_mcp(tool, server_id, cfg.transport)
                        existing = self._tool_db.get_tool(record.tool_id)
                        self._tool_db.upsert_tool(record)
                        seen_ids.add(record.tool_id)
                        if existing:
                            result.tools_updated += 1
                        else:
                            result.tools_added += 1
                # Stale detection: tools from this server not seen are deprecated
                before = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")
                all_server_tools = {
                    t.tool_id
                    for t in self._tool_db.list_tools(server_id=server_id)
                }
                removed = all_server_tools - seen_ids
                for tid in removed:
                    self._tool_db.delete_tool(tid)
                    result.tools_removed += 1
                result.servers_synced += 1
                logger.info(
                    "Synced server=%s tools=%d removed=%d",
                    server_id,
                    len(seen_ids),
                    len(removed),
                )
            except Exception as exc:
                logger.exception("Sync failed for server %s", server_id)
                result.servers_failed += 1
                result.errors.append(f"{server_id}: {exc}")

        # Reconcile: drop tools belonging to servers no longer in the registry.
        # (full_sync above only visits registered servers, so a removed server's
        # tools would otherwise be orphaned in the DB forever. Down-but-still-
        # registered servers are kept — they're hidden at query time by liveness.)
        registry_ids = set(self._registry.servers.keys())
        for rec in self._tool_db.get_all_tools():
            if rec.server_id not in registry_ids:
                self._tool_db.delete_tool(rec.tool_id)
                result.tools_removed += 1
                logger.info("Pruned tool %s (server %s not in registry)", rec.tool_id, rec.server_id)
        return result

    async def incremental_sync(self) -> SyncResult:
        """Same as full_sync for now — MCP servers don't offer delta sync.

        This can be optimised later with timestamps / etags if upstreams
        support them, but for homelab scale a full sync is cheap.
        """
        return await self.full_sync()

    def _tool_from_mcp(
        self,
        tool: object,  # mcp.types.Tool
        server_id: str,
        transport: str,
    ) -> ToolRecord:
        """Convert an mcp.types.Tool to a ToolRecord."""
        # Duck-type to avoid hard import dependency on mcp.types.Tool
        # (mcp library may change).  All fields accessed via getattr.
        tool_name: str = getattr(tool, "name", "unknown")
        description: str = getattr(tool, "description", "")
        input_schema: dict = getattr(tool, "inputSchema", getattr(tool, "input_schema", {}))
        # If description is empty, extract first sentence of input schema as hint
        if not description and isinstance(input_schema, dict):
            props = input_schema.get("properties", {})
            if props:
                description = f"Inputs: {', '.join(props.keys())}"

        tid = ToolRecord.compose_tool_id(server_id, tool_name)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")

        return ToolRecord(
            tool_id=tid,
            tool_name=tool_name,
            description=description,
            server_id=server_id,
            server_name=server_id,  # fallback — server_name can be enriched later
            transport=transport,
            input_schema=input_schema,
            tool_type=_infer_tool_type(tool_name, description),
            last_seen_at=now,
        )