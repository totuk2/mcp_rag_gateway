"""Upstream MCP server definitions (transport + connection parameters)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from gateway.merge import assert_safe_server_id

TransportName = Literal["stdio", "sse", "streamable_http"]


@dataclass(frozen=True)
class ServerConfig:
    server_id: str
    transport: TransportName
    # stdio
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    # http-based
    url: str | None = None
    headers: dict[str, str] | None = None
    # Catalog metadata (optional, any transport): a hand-written one-sentence
    # description wins over the LLM/upstream one; categories are forced into
    # the dynamic tool taxonomy and onto this server's tools.
    description: str | None = None
    categories: tuple[str, ...] = ()
    # Server keeps per-session state (e.g. Playwright's open page): reuse one
    # upstream session per downstream MCP session instead of one per call.
    stateful: bool = False
    # Rewrite oversized flight-search results into a short digest (gateway/compactors.py).
    compact_results: bool = False


@dataclass(frozen=True)
class Registry:
    servers: dict[str, ServerConfig]


def _catalog_meta(raw: dict[str, Any]) -> dict[str, Any]:
    desc = raw.get("description")
    cats = raw.get("categories") or ()
    return {
        "description": str(desc).strip() if desc else None,
        "categories": tuple(str(c) for c in cats),
        "stateful": bool(raw.get("stateful", False)),
        "compact_results": bool(raw.get("compact_results", False)),
    }


def _one_server(server_id: str, raw: dict[str, Any]) -> ServerConfig:
    transport = raw["transport"]
    if transport not in ("stdio", "sse", "streamable_http"):
        raise ValueError(f"Unknown transport {transport!r} for server {server_id!r}")
    if transport == "stdio":
        return ServerConfig(
            server_id=server_id,
            transport="stdio",
            command=raw["command"],
            args=tuple(raw.get("args") or ()),
            env=dict(raw["env"]) if raw.get("env") else None,
            cwd=raw.get("cwd"),
            **_catalog_meta(raw),
        )
    url = raw.get("url")
    if not url:
        raise ValueError(f"Server {server_id!r} ({transport}) requires url")
    headers = dict(raw["headers"]) if raw.get("headers") else None
    return ServerConfig(
        server_id=server_id,
        transport=transport,  # type: ignore[arg-type]
        url=url,
        headers=headers,
        **_catalog_meta(raw),
    )


def load_registry(path: Path) -> Registry:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Registry must be a YAML mapping")
    servers_raw = doc.get("servers") or {}
    servers: dict[str, ServerConfig] = {}
    for sid, cfg in servers_raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"servers[{sid!r}] must be a mapping")
        sid_str = str(sid)
        assert_safe_server_id(sid_str)
        servers[sid_str] = _one_server(sid_str, cfg)
    return Registry(servers=servers)


def load_registries(primary: Path, generated: Path | None = None) -> Registry:
    """Merge the hand-written registry with the provision.py-generated one.

    Generated entries load first; hand-written entries win on id conflict.
    A missing generated file is fine (provision.py simply hasn't run).
    """
    servers: dict[str, ServerConfig] = {}
    if generated is not None and generated.exists():
        servers.update(load_registry(generated).servers)
    servers.update(load_registry(primary).servers)
    return Registry(servers=servers)
