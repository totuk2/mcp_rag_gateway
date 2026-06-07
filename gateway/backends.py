"""Connect to one upstream MCP server (stdio, SSE, or Streamable HTTP)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from gateway.registry import ServerConfig


@asynccontextmanager
async def open_upstream_session(cfg: ServerConfig) -> AsyncIterator[ClientSession]:
    if cfg.transport == "stdio":
        if not cfg.command:
            raise ValueError(f"stdio server {cfg.server_id!r} needs command")
        params = StdioServerParameters(
            command=cfg.command,
            args=list(cfg.args),
            env=cfg.env,
            cwd=cfg.cwd,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return

    if not cfg.url:
        raise ValueError(f"server {cfg.server_id!r} ({cfg.transport}) needs url")

    if cfg.transport == "sse":
        async with sse_client(cfg.url, headers=cfg.headers or {}) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return

    if cfg.transport == "streamable_http":
        headers = cfg.headers or {}
        timeout = httpx.Timeout(30.0, read=300.0)
        async with create_mcp_http_client(headers, timeout) as http_client:
            async with streamable_http_client(cfg.url, http_client=http_client) as streams:
                read, write, _get_id = streams
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        return

    raise AssertionError(f"Unhandled transport {cfg.transport}")
