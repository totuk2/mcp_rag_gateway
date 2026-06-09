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
    from tool_rag.planner import Planner

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

# Batch execution: run several discovered tools concurrently in one call.
RUN_TOOLS_NAME = "run_tools"

# Lazy schema fetch: get a single tool's full input_schema on demand (pairs with
# find_tools include_schema=false), and register it for strict clients.
DESCRIBE_TOOL_NAME = "describe_tool"

# Optional planning meta-tool: returns a structured multi-step plan (LLM-backed).
# Only listed when a planner is configured.
PLAN_TOOL_NAME = "plan"

# Default cap on concurrent upstream calls in run_tools (overridable per call and
# via TOOL_RAG_MAX_PARALLEL). Bounds stdio subprocess spawns / upstream load.
DEFAULT_MAX_PARALLEL = 8


def _gateway_instructions(planner_on: bool) -> str:
    """MCP `initialize` instructions; surfaced so the agent knows the catalog is
    hidden and how to discover/execute tools. Only set when Tool-RAG is on."""
    base = (
        "This gateway proxies many upstream MCP servers but hides its full tool "
        "catalog to keep your context small. Meta-tools available: "
        f"`{FIND_TOOLS_NAME}` (discover tools by query), `{RUN_TOOL_NAME}` (execute "
        f"one discovered tool), `{RUN_TOOLS_NAME}` (execute several in parallel), and "
        f"`{DESCRIBE_TOOL_NAME}` (fetch a tool's full input_schema on demand). "
        "Workflow: FIRST call `find_tools` with a natural-language `query`; it "
        "returns matching tools with `call_name` (and `input_schema` unless you set "
        f"`include_schema=false`, in which case call `{DESCRIBE_TOOL_NAME}` for the "
        f"schema). THEN call `{RUN_TOOL_NAME}`/`{RUN_TOOLS_NAME}` with the chosen "
        "`call_name`(s) and `arguments` matching the schema."
    )
    if planner_on:
        base += (
            f" For multi-step tasks, call `{PLAN_TOOL_NAME}` with a `query` to get a "
            f"structured plan (ordered steps + parallel groups); fill in arguments and "
            f"execute each group with `{RUN_TOOLS_NAME}`."
        )
    return base


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
                "include_schema": {
                    "type": "boolean",
                    "description": (
                        "If false, omit each tool's input_schema for a lighter "
                        "name+description shortlist; fetch the schema later with "
                        f"`{DESCRIBE_TOOL_NAME}`. Defaults to true."
                    ),
                    "default": True,
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


def _run_tools_definition() -> types.Tool:
    """Batch execution proxy: run several discovered tools concurrently."""
    return types.Tool(
        name=RUN_TOOLS_NAME,
        description=(
            "Execute several tools discovered via find_tools concurrently, in one "
            "call. Use for independent or parallelizable steps. Each call runs "
            "independently — one failure does not abort the others. Returns a result "
            "per call in input order."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "description": "Tools to run concurrently.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Optional caller-supplied id echoed back in the result.",
                            },
                            "call_name": {
                                "type": "string",
                                "description": "The `call_name` returned by find_tools.",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Arguments for the tool, matching its `input_schema`.",
                            },
                        },
                        "required": ["call_name"],
                    },
                },
                "max_concurrency": {
                    "type": "integer",
                    "description": "Optional cap on parallelism (clamped to the server limit).",
                },
            },
            "required": ["calls"],
        },
    )


def _describe_tool_definition() -> types.Tool:
    """Fetch one tool's full input_schema on demand (two-phase discovery)."""
    return types.Tool(
        name=DESCRIBE_TOOL_NAME,
        description=(
            "Fetch the full `input_schema` (and metadata) for a single tool by its "
            "`call_name`. Use after a find_tools call made with include_schema=false, "
            "before executing the tool."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "call_name": {
                    "type": "string",
                    "description": "The `call_name` returned by find_tools.",
                },
            },
            "required": ["call_name"],
        },
    )


def _plan_definition() -> types.Tool:
    """Optional LLM-backed planning meta-tool."""
    return types.Tool(
        name=PLAN_TOOL_NAME,
        description=(
            "Produce a structured, multi-step plan for a task: it discovers relevant "
            "tools and returns ordered steps with dependencies and parallel groups. "
            "The plan is advisory — fill in concrete arguments and execute each group "
            f"yourself via `{RUN_TOOLS_NAME}`. It does not run anything."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the task to plan.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max candidate tools to consider (default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    )


def build_gateway_server(
    registry: Registry,
    retriever: "Retriever | None" = None,
    planner: "Planner | None" = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
) -> Server:
    # Advertise discovery instructions only when Tool-RAG is wired up.
    server = Server(
        "homelab-mcp-gateway",
        version="0.1.0",
        instructions=_gateway_instructions(planner is not None) if retriever is not None else None,
    )

    async def _gather_candidates(query: str, top_k: int) -> tuple[list[dict[str, Any]], bool]:
        """Run the policy-scoped retriever and return result dicts (with full schema)
        plus the fallback flag. Shared by find_tools and plan."""
        assert retriever is not None
        policy = get_policy()
        allowed = list(policy.servers.keys())
        if not allowed:
            return [], False
        # Over-fetch: the retriever truncates to its top_k *before* we apply the
        # per-key tool_prefixes filter below, so a prefix-restricted key would
        # otherwise see fewer than top_k tools. Fetch extra headroom, then slice.
        fetch_k = top_k * 3 if any(r.tool_prefixes for r in policy.servers.values()) else top_k
        result = await retriever.retrieve(query=query, top_k=fetch_k, allowed_servers=allowed)
        out: list[dict[str, Any]] = []
        for r in result.results:
            # tool_id is the merged {server_id}__{tool} name == the call_name.
            try:
                sid, orig = split_merged_name(r.tool_id)
            except ValueError:
                continue
            if not policy.tool_visible(sid, orig):
                continue  # honour per-key tool_prefixes allowlists
            out.append({
                "call_name": r.tool_id,
                "server": r.server_name,
                "tool_type": r.tool_type,
                "description": r.description,
                "input_schema": _clean_schema(r.input_schema),
                "score": r.score,
            })
            if len(out) >= top_k:
                break
        return out, result.fallback_used

    async def _register_session_tools(results: list[dict[str, Any]], lazy: bool) -> None:
        """Register discovered tools on this session + notify the client so strict
        clients (e.g. LibreChat) can call them. In lazy mode use a placeholder
        schema; describe_tool fills the real one later."""
        try:
            session = server.request_context.session
            discovered = _session_tools.setdefault(session, {})
            for r in results:
                discovered[r["call_name"]] = types.Tool(
                    name=r["call_name"],
                    description=r["description"],
                    inputSchema={"type": "object"} if lazy else r["input_schema"],
                )
            await session.send_tool_list_changed()
        except Exception:
            logger.debug("Could not send tools/list_changed", exc_info=True)

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
        include_schema = args.get("include_schema", True)
        if not isinstance(include_schema, bool):
            include_schema = True

        policy = get_policy()
        if not policy.servers:
            payload = {"query": query, "count": 0, "results": [],
                       "note": "No servers are granted to this API key."}
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(payload))]
            )

        results, fallback_used = await _gather_candidates(query, top_k)

        # Register tools for the session so strict clients can call them. In lazy
        # mode register a placeholder schema; describe_tool fills it in later.
        await _register_session_tools(results, lazy=not include_schema)

        if not include_schema:
            results = [{k: v for k, v in r.items() if k != "input_schema"} for r in results]
            instructions = (
                f"Schemas omitted. Call `{DESCRIBE_TOOL_NAME}` with a `call_name` to "
                f"get its input_schema, then `{RUN_TOOL_NAME}`/`{RUN_TOOLS_NAME}` to execute."
            )
        else:
            instructions = (
                f"To execute a discovered tool, call `{RUN_TOOL_NAME}` with "
                '{"call_name": "<call_name>", "arguments": {<args matching input_schema>}}; '
                f"use `{RUN_TOOLS_NAME}` to run several in parallel."
            )

        payload = {
            "query": query,
            "count": len(results),
            "results": results,
            "instructions": instructions,
            "fallback_used": fallback_used,
        }
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
        base: list[types.Tool] = []
        if retriever is not None:
            base = [
                _find_tools_definition(),
                _run_tool_definition(),
                _run_tools_definition(),
                _describe_tool_definition(),
            ]
            if planner is not None:
                base.append(_plan_definition())
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

    def _result_to_json(res: types.CallToolResult) -> dict[str, Any]:
        """Flatten a CallToolResult into a JSON-able summary for run_tools. On
        failure always populate `error` so callers see one consistent shape."""
        texts = "\n".join(c.text for c in res.content if isinstance(c, types.TextContent))
        out: dict[str, Any] = {"ok": not res.isError}
        if res.isError:
            out["error"] = texts or "tool returned an error"
            return out
        if res.structuredContent is not None:
            out["structured"] = res.structuredContent
        if texts:
            out["text"] = texts
        non_text = [c.type for c in res.content if not isinstance(c, types.TextContent)]
        if non_text:
            out["content_types"] = non_text
        return out

    async def handle_run_tools(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Batch execution proxy: run several discovered tools concurrently, with a
        bounded concurrency and per-call error isolation."""
        args = arguments or {}
        calls = args.get("calls")
        if not isinstance(calls, list) or not calls:
            return _err_tool("run_tools requires a non-empty `calls` array.")
        try:
            requested = int(args.get("max_concurrency", max_parallel))
        except (TypeError, ValueError):
            requested = max_parallel
        limit = max(1, min(requested, max_parallel))
        sem = asyncio.Semaphore(limit)

        async def _one(idx: int, call: Any) -> dict[str, Any]:
            tag: dict[str, Any] = {"index": idx}
            if isinstance(call, dict) and call.get("id") is not None:
                tag["id"] = call["id"]
            if not isinstance(call, dict) or not isinstance(call.get("call_name"), str):
                return {**tag, "ok": False, "error": "each call needs a string `call_name`."}
            tag["call_name"] = call["call_name"]
            tool_args = call.get("arguments") or {}
            if not isinstance(tool_args, dict):
                return {**tag, "ok": False, "error": "`arguments` must be a JSON object."}
            async with sem:
                try:
                    res = await _dispatch_tool(call["call_name"], tool_args)
                except Exception as e:  # defensive; _dispatch_tool already catches upstream
                    return {**tag, "ok": False, "error": f"Execution error: {e}"}
            return {**tag, **_result_to_json(res)}

        results = await asyncio.gather(
            *[_one(i, c) for i, c in enumerate(calls)], return_exceptions=False
        )
        payload = {"count": len(results), "max_concurrency": limit, "results": list(results)}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))]
        )

    async def handle_describe_tool(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Two-phase discovery: return a single tool's full input_schema on demand,
        and register it for the session so strict clients can then call it."""
        if retriever is None:
            return _err_tool("Tool discovery is not enabled on this gateway.")
        args = arguments or {}
        call_name = args.get("call_name")
        if not call_name or not isinstance(call_name, str):
            return _err_tool("describe_tool requires a non-empty string `call_name`.")
        if not _tool_allowed_by_policy(call_name, get_policy()):
            return _err_tool("Access denied for this tool.")
        rec = retriever.get_tool(call_name)
        if rec is None:
            return _err_tool(f"Unknown tool {call_name!r}.")
        schema = _clean_schema(rec.input_schema)
        payload = {
            "call_name": rec.tool_id,
            "server": rec.server_name,
            "tool_type": rec.tool_type,
            "description": rec.description,
            "input_schema": schema,
        }
        await _register_session_tools(
            [{"call_name": rec.tool_id, "description": rec.description, "input_schema": schema}],
            lazy=False,
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))]
        )

    async def handle_plan(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """LLM-backed planning: discover candidate tools, ask the planner for a
        structured multi-step plan, and return it (no execution)."""
        if retriever is None or planner is None:
            return _err_tool("Planning is not enabled on this gateway.")
        args = arguments or {}
        query = args.get("query")
        if not query or not isinstance(query, str):
            return _err_tool("plan requires a non-empty string `query`.")
        try:
            top_k = int(args.get("top_k", 10))
        except (TypeError, ValueError):
            top_k = 10
        if not get_policy().servers:
            return _err_tool("No servers are granted to this API key.")
        candidates, _ = await _gather_candidates(query, top_k)
        if not candidates:
            payload = {"query": query, "steps": [], "notes": "No relevant tools found.",
                       "missing": []}
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(payload))]
            )
        try:
            plan = await planner.plan(query, candidates)
        except Exception as e:
            logger.exception("planner error")
            return _err_tool(f"Planner error: {e}")
        payload = {"query": query, **plan,
                   "instructions": (
                       f"Each step lists its `args`/`required`; for the exact input_schema "
                       f"call `{DESCRIBE_TOOL_NAME}` with the step's `call_name`. Fill in "
                       f"`arguments` against the schema, then execute each `group` (steps "
                       f"sharing a group have no inter-dependencies) via `{RUN_TOOLS_NAME}`."
                   )}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))]
        )

    @server.call_tool(validate_input=False)
    async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        if name == FIND_TOOLS_NAME:
            return await handle_find_tools(arguments)
        if name == RUN_TOOL_NAME:
            return await handle_run_tool(arguments)
        if name == RUN_TOOLS_NAME:
            return await handle_run_tools(arguments)
        if name == DESCRIBE_TOOL_NAME:
            return await handle_describe_tool(arguments)
        if name == PLAN_TOOL_NAME:
            return await handle_plan(arguments)
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
