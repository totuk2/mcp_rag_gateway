"""MCP Server implementation: merge upstreams and enforce AccessPolicy."""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import mcp.types as types
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import Server
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from gateway.backends import open_upstream_session
from gateway.context import get_policy
from gateway.merge import (
    gateway_resource_uri,
    merged_prompt_name,
    merged_tool_name,
    parse_gateway_resource_uri,
    split_merged_name,
)
from gateway.registry import Registry

if TYPE_CHECKING:
    from gateway.policy import AccessPolicy
    from tool_rag.retriever import Retriever

logger = logging.getLogger(__name__)

# Per-session discovered tools: session object → {merged_tool_name: types.Tool}
# WeakKeyDictionary auto-cleans when the session is GC'd (session ends/times out).
_session_tools: weakref.WeakKeyDictionary[Any, dict[str, types.Tool]] = weakref.WeakKeyDictionary()

# Name of the always-listed meta-tool that exposes Tool-RAG discovery in-band.
# Has no "__" so it never collides with a merged {server_id}__{tool} name.
FIND_TOOLS_NAME = "find_tools"

# Companion meta-tool: executes any tool discovered via find_tools.
# Always listed alongside find_tools so clients with a static execution registry
# (e.g. LibreChat) can call discovered tools without needing tools/list_changed support.
RUN_TOOL_NAME = "run_tool"

# Surfaced to clients at MCP `initialize` so the agent knows the catalog is
# hidden and how to discover tools. Only set when Tool-RAG (the retriever) is on.
GATEWAY_INSTRUCTIONS = (
    "This gateway proxies many upstream MCP servers but hides its full tool "
    "catalog to keep your context small. Two meta-tools are always available: "
    f"`{FIND_TOOLS_NAME}` (discover tools by query) and `{RUN_TOOL_NAME}` (execute "
    "a discovered tool). Workflow: FIRST call `find_tools` with a natural-language "
    "`query`; it returns matching tools with `call_name` and `input_schema`. THEN "
    f"call `{RUN_TOOL_NAME}` with the chosen `call_name` and `arguments` matching "
    "that schema."
)


def _clean_schema(schema: Any) -> Any:
    """Recursively strip 'title' from JSON Schema dicts.

    Pydantic injects 'title' at every level of the schema it generates. Most
    OpenAI-compatible providers (including OpenRouter) reject tool input schemas
    that contain 'title' fields, returning 400. Strip them before sending to LLMs.
    """
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema


def _tool_allowed_by_policy(merged_name: str, policy: "AccessPolicy") -> bool:
    try:
        sid, orig = split_merged_name(merged_name)
    except ValueError:
        return False
    return policy.allows_server(sid) and policy.tool_visible(sid, orig)


def _err_tool(msg: str) -> types.CallToolResult:
    return types.CallToolResult(
        isError=True,
        content=[types.TextContent(type="text", text=msg)],
    )


def _find_tools_definition() -> types.Tool:
    """The in-band discovery meta-tool advertised to every key."""
    return types.Tool(
        name=FIND_TOOLS_NAME,
        description=(
            "Discover tools available through this gateway. The full catalog is "
            "hidden to save context, so you MUST call this to find a tool before "
            "using it. Describe what you want to accomplish in `query`; "
            "this returns the most relevant tools with their `call_name` and "
            "`input_schema`. Then execute the chosen tool using `run_tool` with "
            "its `call_name` and matching arguments."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the task you want to perform.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of tools to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )


def _run_tool_definition() -> types.Tool:
    """Execution proxy: runs any tool discovered via find_tools."""
    return types.Tool(
        name=RUN_TOOL_NAME,
        description=(
            "Execute a tool discovered via find_tools. After calling find_tools, "
            "use this to run a specific tool by its `call_name` with the arguments "
            "matching its `input_schema`. This is the required execution path for "
            "tools that are not pre-listed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "call_name": {
                    "type": "string",
                    "description": "The `call_name` returned by find_tools (e.g. `images__fetch_images`).",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool, matching its `input_schema`.",
                },
            },
            "required": ["call_name"],
        },
    )


def build_gateway_server(registry: Registry, retriever: "Retriever | None" = None) -> Server:
    # Advertise discovery instructions only when Tool-RAG is wired up.
    server = Server(
        "homelab-mcp-gateway",
        version="0.1.0",
        instructions=GATEWAY_INSTRUCTIONS if retriever is not None else None,
    )

    async def handle_find_tools(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Semantic tool discovery exposed as an MCP tool (in-band Tool-RAG)."""
        if retriever is None:
            return _err_tool("Tool discovery is not enabled on this gateway.")
        args = arguments or {}
        query = args.get("query")
        if not query or not isinstance(query, str):
            return _err_tool("find_tools requires a non-empty string `query`.")
        try:
            top_k = int(args.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5

        policy = get_policy()
        allowed = list(policy.servers.keys())
        if not allowed:
            # No servers granted to this key -> nothing to discover.
            payload = {"query": query, "count": 0, "results": [],
                       "note": "No servers are granted to this API key."}
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(payload))]
            )

        # Over-fetch: the retriever truncates to its top_k *before* we apply the
        # per-key tool_prefixes filter below, so a prefix-restricted key would
        # otherwise see fewer than top_k tools. Fetch extra headroom, then slice.
        fetch_k = top_k * 3 if any(r.tool_prefixes for r in policy.servers.values()) else top_k
        result = await retriever.retrieve(query=query, top_k=fetch_k, allowed_servers=allowed)

        results = []
        for r in result.results:
            # tool_id is the merged {server_id}__{tool} name == the call_name.
            try:
                sid, orig = split_merged_name(r.tool_id)
            except ValueError:
                continue
            if not policy.tool_visible(sid, orig):
                continue  # honour per-key tool_prefixes allowlists
            results.append({
                "call_name": r.tool_id,
                "server": r.server_name,
                "tool_type": r.tool_type,
                "description": r.description,
                "input_schema": _clean_schema(r.input_schema),
                "score": r.score,
            })
            if len(results) >= top_k:
                break

        payload = {
            "query": query,
            "count": len(results),
            "results": results,
            "instructions": (
                f"To execute a discovered tool, call `{RUN_TOOL_NAME}` with "
                '{"call_name": "<call_name>", "arguments": {<args matching input_schema>}}.'
            ),
            "fallback_used": result.fallback_used,
        }

        # Register discovered tools on this session and notify the client so it
        # re-fetches list_tools() and can call the tools without them having been
        # in the original list (strict clients like LibreChat require this).
        try:
            session = server.request_context.session
            discovered = _session_tools.setdefault(session, {})
            for r in results:
                discovered[r["call_name"]] = types.Tool(
                    name=r["call_name"],
                    description=r["description"],
                    inputSchema=r["input_schema"],
                )
            await session.send_tool_list_changed()
        except Exception:
            logger.debug("Could not send tools/list_changed", exc_info=True)

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))]
        )

    @server.list_tools()
    async def handle_list_tools(_req: types.ListToolsRequest) -> types.ListToolsResult:
        policy = get_policy()
        # The discovery meta-tool is the in-band entry point: it is the ONLY tool
        # a non-admin key sees, so any MCP client (not just frameworks that know
        # to POST /tool-rag/retrieve) can find tools. Admin keys see it plus the
        # full catalog. Omitted entirely when Tool-RAG is disabled.
        base: list[types.Tool] = (
            [_find_tools_definition(), _run_tool_definition()] if retriever is not None else []
        )
        if not policy.admin:
            # Include tools discovered via find_tools in this session so strict
            # MCP clients (e.g. LibreChat) can call them after tools/list_changed.
            try:
                session = server.request_context.session
                extra = [
                    t for name, t in _session_tools.get(session, {}).items()
                    if _tool_allowed_by_policy(name, policy)
                ]
            except LookupError:
                extra = []
            return types.ListToolsResult(tools=base + extra)
        out: list[types.Tool] = list(base)

        async def one(server_id: str) -> list[types.Tool]:
            cfg = registry.servers.get(server_id)
            if not cfg or not policy.allows_server(server_id):
                return []
            try:
                async with open_upstream_session(cfg) as us:
                    tr = await us.list_tools()
                    tools: list[types.Tool] = []
                    for t in tr.tools:
                        if not policy.tool_visible(server_id, t.name):
                            continue
                        tools.append(
                            t.model_copy(
                                update={"name": merged_tool_name(server_id, t.name)},
                                deep=True,
                            )
                        )
                    return tools
            except Exception:
                logger.exception("list_tools failed for server %s", server_id)
                return []

        results = await asyncio.gather(*[one(sid) for sid in policy.servers])
        for part in results:
            out.extend(part)
        return types.ListToolsResult(tools=out)

    async def _dispatch_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Inner dispatch: validate access and call an upstream tool by merged name."""
        policy = get_policy()
        try:
            server_id, orig = split_merged_name(name)
        except ValueError:
            return _err_tool("Invalid tool name (expected server__tool).")
        if server_id not in registry.servers:
            return _err_tool(f"Unknown server {server_id!r}.")
        if not policy.allows_server(server_id) or not policy.tool_visible(server_id, orig):
            return _err_tool("Access denied for this tool.")
        cfg = registry.servers[server_id]
        try:
            async with open_upstream_session(cfg) as us:
                return await us.call_tool(orig, arguments)
        except Exception as e:
            logger.exception("call_tool upstream error")
            return _err_tool(f"Upstream error: {e}")

    async def handle_run_tool(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Execution proxy: runs any tool discovered via find_tools."""
        args = arguments or {}
        call_name = args.get("call_name")
        if not call_name or not isinstance(call_name, str):
            return _err_tool("run_tool requires a non-empty string `call_name`.")
        tool_arguments = args.get("arguments") or {}
        if not isinstance(tool_arguments, dict):
            return _err_tool("`arguments` must be a JSON object.")
        return await _dispatch_tool(call_name, tool_arguments)

    @server.call_tool(validate_input=False)
    async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        if name == FIND_TOOLS_NAME:
            return await handle_find_tools(arguments)
        if name == RUN_TOOL_NAME:
            return await handle_run_tool(arguments)
        return await _dispatch_tool(name, arguments)

    @server.list_resources()
    async def handle_list_resources(_req: types.ListResourcesRequest) -> types.ListResourcesResult:
        policy = get_policy()
        out: list[types.Resource] = []

        async def one(server_id: str) -> list[types.Resource]:
            cfg = registry.servers.get(server_id)
            if not cfg or not policy.allows_server(server_id):
                return []
            try:
                async with open_upstream_session(cfg) as us:
                    lr = await us.list_resources()
                    res: list[types.Resource] = []
                    for r in lr.resources:
                        u = str(r.uri)
                        if not policy.uri_visible(server_id, u):
                            continue
                        new_uri = gateway_resource_uri(server_id, u)
                        res.append(
                            r.model_copy(
                                update={
                                    "uri": new_uri,
                                    "name": merged_tool_name(server_id, r.name),
                                },
                                deep=True,
                            )
                        )
                    return res
            except Exception:
                logger.exception("list_resources failed for server %s", server_id)
                return []

        parts = await asyncio.gather(*[one(sid) for sid in policy.servers])
        for p in parts:
            out.extend(p)
        return types.ListResourcesResult(resources=out)

    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> Iterable[ReadResourceContents]:
        policy = get_policy()
        try:
            server_id, original_uri = parse_gateway_resource_uri(uri)
        except ValueError as e:
            raise McpError(
                types.ErrorData(code=types.INVALID_PARAMS, message=f"Invalid gateway resource URI: {e}")
            ) from e
        if server_id not in registry.servers:
            raise McpError(
                types.ErrorData(code=types.INVALID_PARAMS, message=f"Unknown server {server_id!r}")
            )
        if not policy.allows_server(server_id) or not policy.uri_visible(server_id, original_uri):
            raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message="Access denied for this resource"))
        cfg = registry.servers[server_id]
        async with open_upstream_session(cfg) as us:
            rr = await us.read_resource(AnyUrl(original_uri))
            return rr.contents

    @server.list_prompts()
    async def handle_list_prompts(_req: types.ListPromptsRequest) -> types.ListPromptsResult:
        policy = get_policy()
        out: list[types.Prompt] = []

        async def one(server_id: str) -> list[types.Prompt]:
            cfg = registry.servers.get(server_id)
            if not cfg or not policy.allows_server(server_id):
                return []
            try:
                async with open_upstream_session(cfg) as us:
                    pr = await us.list_prompts()
                    prompts: list[types.Prompt] = []
                    for p in pr.prompts:
                        if not policy.prompt_visible(server_id, p.name):
                            continue
                        prompts.append(
                            p.model_copy(
                                update={"name": merged_prompt_name(server_id, p.name)},
                                deep=True,
                            )
                        )
                    return prompts
            except Exception:
                logger.exception("list_prompts failed for server %s", server_id)
                return []

        parts = await asyncio.gather(*[one(sid) for sid in policy.servers])
        for p in parts:
            out.extend(p)
        return types.ListPromptsResult(prompts=out)

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        policy = get_policy()
        try:
            server_id, orig = split_merged_name(name)
        except ValueError:
            raise McpError(
                types.ErrorData(code=types.INVALID_PARAMS, message="Invalid prompt name (expected server__prompt).")
            ) from None
        if server_id not in registry.servers:
            raise McpError(
                types.ErrorData(code=types.INVALID_PARAMS, message=f"Unknown server {server_id!r}")
            )
        if not policy.allows_server(server_id) or not policy.prompt_visible(server_id, orig):
            raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message="Access denied for this prompt"))
        cfg = registry.servers[server_id]
        async with open_upstream_session(cfg) as us:
            return await us.get_prompt(orig, arguments)

    return server
