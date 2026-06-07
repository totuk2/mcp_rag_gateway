"""Tool metadata record — source of truth for one upstream MCP tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


ToolType = Literal["read", "write", "admin", "action", "query"]
ToolStatus = Literal["active", "disabled", "deprecated"]


@dataclass(frozen=True)
class ToolRecord:
    """Persistent metadata for one tool from an upstream MCP server.

    Matches the schema defined in spec §6.3.A (Tool Registry).
    """

    tool_id: str
    tool_name: str
    description: str
    server_id: str
    server_name: str
    transport: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    permission_scope: str = ""
    risk_level: str = "low"
    tool_type: ToolType = "action"
    status: ToolStatus = "active"
    version: str = "0.0.1"
    last_seen_at: str = ""
    last_indexed_at: str = ""

    @classmethod
    def compose_tool_id(cls, server_id: str, tool_name: str) -> str:
        return f"{server_id}__{tool_name}"