"""Minimal stdio MCP server for local testing (one `ping` tool)."""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def ping() -> str:
    """Return pong."""
    return "pong"


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())
