"""MCP Server implementation: merge upstreams and enforce AccessPolicy."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import mcp.types as types
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import Server
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from gateway.backends import open_upstream_session
from gateway.compactors import compact
from gateway.jobs import JobStore
from gateway.results import RESULT_MAX_CHARS, ProjectionError, ResultStore, view
from gateway.context import get_policy
from gateway.merge import (
    gateway_resource_uri,
    merged_prompt_name,
    merged_tool_name,
    parse_gateway_resource_uri,
    split_merged_name,
)
from gateway.recipes import RecipeError, RecipeStore, site_of, urls_in
from gateway.registry import Registry
from tool_rag.agent import AgentTool, agent_enabled, run_agent

if TYPE_CHECKING:
    from gateway.policy import AccessPolicy
    from gateway.session_pool import StickySessionPool
    from gateway.skills import SkillStore
    from tool_rag.catalog import ToolCatalog
    from tool_rag.retriever import Retriever
    from tool_rag.planner import Planner

logger = logging.getLogger(__name__)

# Per-session discovered tools: session object → {merged_tool_name: types.Tool}
# WeakKeyDictionary auto-cleans when the session is GC'd (session ends/times out).
_session_tools: weakref.WeakKeyDictionary[Any, dict[str, types.Tool]] = weakref.WeakKeyDictionary()

# Per-session sites whose recipes were already injected (inject once per site).
_session_recipe_sites: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()

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

# Catalog meta-tool: general questions about what's available (domains,
# categories, "do I have tools for X?"). Listed when a catalog is wired up.
BROWSE_TOOLS_NAME = "browse_tools"

# Skills: markdown playbooks (skills/<name>/SKILL.md) fetched on demand.
GET_SKILL_NAME = "get_skill"

# Sub-agent: the gateway runs a whole multi-step task with the planner model
# (TOOL_RAG_AGENT=on + planner configured). See tool_rag/agent.py.
DELEGATE_NAME = "delegate"

# Site recipes: procedural memory for browser automation (gateway/recipes.py).
GET_SITE_RECIPES_NAME = "get_site_recipes"
SAVE_SITE_RECIPE_NAME = "save_site_recipe"

# Long calls (past TOOL_CALL_SYNC_SECS) continue in the background; fetch with this.
GET_JOB_RESULT_NAME = "get_job_result"

# Read back a stored full result (compacted / oversized / projected calls) by result_id.
GET_RESULT_NAME = "get_result"

_FIELDS_SCHEMA = {
    "type": "array", "items": {"type": "string"},
    "description": ("Return only these fields of a JSON result (paths like \"query\", \"a.b\", "
                    "\"items[].price\", \"items[0].title\"). Saves context on big results."),
}

# Default cap on concurrent upstream calls in run_tools (overridable per call and
# via TOOL_RAG_MAX_PARALLEL). Bounds stdio subprocess spawns / upstream load.
DEFAULT_MAX_PARALLEL = 8

# Default cap (chars) on the *teaser* description returned in the find_tools
# shortlist. Long upstream descriptions (e.g. arxiv search_papers' ~50 lines of
# query-construction guidance) bloat the agent's context for no selection benefit;
# the full text stays in describe_tool + the embedding. 0 = no truncation.
DEFAULT_SHORTLIST_DESC_CHARS = 300

# plan multi-intent candidate gathering (decompose -> retrieve per intent -> union).
# See docs/superpowers/specs/2026-06-14-plan-multi-intent-candidates-design.md.
DEFAULT_PLAN_MAX_INTENTS = 6        # max sub-intents kept from decomposition
DEFAULT_PLAN_PER_INTENT_K = 5       # retrieval depth per intent (and per base query)
DEFAULT_PLAN_TOP_K = 30             # default union cap (the plan `top_k` arg)


def _union_candidates(
    lists: list[list[dict[str, Any]]], cap: int
) -> list[dict[str, Any]]:
    """Merge per-intent candidate lists into one ranked list.

    Dedup by `call_name` keeping the entry with the max `score`; sort by score
    descending with a deterministic tie-break on `call_name`; truncate to `cap`
    (cap <= 0 means no truncation). Pure and deterministic — no I/O."""
    best: dict[str, dict[str, Any]] = {}
    for lst in lists:
        for c in lst:
            name = c.get("call_name")
            if not name:
                continue
            prev = best.get(name)
            if prev is None or c.get("score", 0) > prev.get("score", 0):
                best[name] = c
    merged = sorted(
        best.values(),
        key=lambda c: (-c.get("score", 0), c.get("call_name") or ""),
    )
    return merged[:cap] if cap > 0 else merged


def _shortlist_description(text: str) -> str:
    """Truncate a tool description to a short teaser for the find_tools shortlist.

    Presentational only (call-time; no reindex). Prefers the first paragraph — for
    front-loaded descriptions this is the summary sentence — then falls back to a
    sentence boundary, then a hard cap. Cap from TOOL_RAG_SHORTLIST_DESC_CHARS
    (0 disables). describe_tool still serves the full description on demand."""
    text = (text or "").strip()
    try:
        cap = int(os.environ.get("TOOL_RAG_SHORTLIST_DESC_CHARS", DEFAULT_SHORTLIST_DESC_CHARS))
    except (TypeError, ValueError):
        cap = DEFAULT_SHORTLIST_DESC_CHARS
    if cap <= 0 or len(text) <= cap:
        return text
    # First paragraph (e.g. arxiv: the lead summary sentence) if it fits.
    para = text.split("\n\n", 1)[0].strip()
    if len(para) <= cap:
        return para
    # Otherwise cut at the last sentence boundary that isn't too aggressive.
    cut = para[:cap]
    dot = cut.rfind(". ")
    if dot >= cap // 2:
        return cut[: dot + 1]
    return cut.rstrip() + "…"


def _gateway_instructions(planner_on: bool, catalog_on: bool = False, skills_on: bool = False,
                          agent_on: bool = False, recipes_on: bool = False) -> str:
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
    if catalog_on:
        base += (
            f" For general questions — what kinds of tools exist, which categories, "
            f"whether anything covers a problem — call `{BROWSE_TOOLS_NAME}` (no arguments "
            f"for an overview, `category` for one category or domain, `query` to check "
            f"coverage of a problem) before `{FIND_TOOLS_NAME}`."
        )
    if skills_on:
        base += (
            f" Some task types have playbooks: when a response mentions a `related_skill`, "
            f"call `{GET_SKILL_NAME}` and follow it — it says which tools to use and what to "
            f"try when the first attempt finds nothing. Don't give up after one empty search."
        )
    base += (
        f" Long calls (research, delegate) may return {{\"status\": \"running\", \"job_id\": ...}} "
        f"instead of a result: they keep running — call `{GET_JOB_RESULT_NAME}` with the job_id "
        f"(repeat while it says running) instead of retrying the call."
    )
    if recipes_on:
        base += (
            f" Before automating a website with a browser, call `{GET_SITE_RECIPES_NAME}` for it "
            f"(saved recipes are also appended automatically when you navigate to a known site). "
            f"After you successfully get the data from a site, call `{SAVE_SITE_RECIPE_NAME}` "
            f"with the URL pattern and steps that worked, so the next visit is fast."
        )
    if agent_on:
        base += (
            f" For long research tasks you can hand the whole task to `{DELEGATE_NAME}`, which "
            f"runs it with tools on the gateway side and returns the result (may take minutes)."
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


def _find_tools_definition(domains: list[str] | None = None) -> types.Tool:
    """The in-band discovery meta-tool advertised to every key. `domains` (the
    key's visible servers) is appended so the agent knows what to expect."""
    description = (
        "Discover tools available through this gateway. The full catalog is "
        "hidden to save context, so you MUST call this to find a tool before "
        "using it. Describe what you want to accomplish in `query`; "
        "this returns the most relevant tools with their `call_name` and "
        "`input_schema`. Then execute the chosen tool using `run_tool` with "
        "its `call_name` and matching arguments."
    )
    if domains:
        description += (
            f" Available domains: {', '.join(domains)} (call `{BROWSE_TOOLS_NAME}` "
            "for categories and descriptions)."
        )
    return types.Tool(
        name=FIND_TOOLS_NAME,
        description=description,
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
                "full": {
                    "type": "boolean",
                    "description": "Return the raw result even if the gateway would compact it (large flight searches).",
                },
                "fields": _FIELDS_SCHEMA,
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
                            "full": {
                                "type": "boolean",
                                "description": "Raw result instead of the gateway's compact digest.",
                            },
                            "fields": _FIELDS_SCHEMA,
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
                    "description": "Max candidate tools (after per-intent union) to consider (default 30).",
                    "default": 30,
                },
            },
            "required": ["query"],
        },
    )


def _browse_tools_definition(summary: str) -> types.Tool:
    """Catalog meta-tool; `summary` is the key's compact catalog overview."""
    return types.Tool(
        name=BROWSE_TOOLS_NAME,
        description=(
            "Answer general questions about which tools you have, without loading "
            "them: call with no arguments for an overview (domains, categories, "
            "counts); with `category` to inspect one category or domain (\"do I "
            "have a tool in category X?\"); with `query` to check whether any tool "
            "covers a problem (\"do I have tools for X?\" -> verdict strong/weak/"
            f"none). Then use `{FIND_TOOLS_NAME}` to get specific tools. {summary}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "A category name (or a domain/server name) to inspect.",
                },
                "query": {
                    "type": "string",
                    "description": "A problem or capability to check coverage for, in natural language.",
                },
            },
        },
    )


def _get_skill_definition(skills: list[dict[str, str]]) -> types.Tool:
    """Playbook fetch; lists the key's visible skills in the description."""
    listing = "; ".join(f"`{s['name']}` — {s['description']}" for s in skills)
    return types.Tool(
        name=GET_SKILL_NAME,
        description=(
            "Fetch a playbook (step-by-step procedure) for a type of task: which tools to "
            "use, in what order, and what to try when results are thin. Available: " + listing
        ),
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Skill name."}},
            "required": ["name"],
        },
    )


def _delegate_definition() -> types.Tool:
    """Gateway-side sub-agent for long multi-step tasks."""
    return types.Tool(
        name=DELEGATE_NAME,
        description=(
            "Hand a whole research task to a gateway-side agent that discovers and runs "
            "tools on its own (following a matching playbook) and returns a final answer "
            "with the steps it took. Use for long, multi-step research (e.g. finding flights "
            "on a hard route); it can take a few minutes. Read-only: it never books or pays."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The full task, with all constraints (dates, places, budget)."},
                "skill": {"type": "string", "description": "Optional playbook name; matched automatically if omitted."},
                "max_steps": {"type": "integer", "description": "Max tool-calling turns (default 15, cap 30)."},
            },
            "required": ["task"],
        },
    )


def _get_result_definition() -> types.Tool:
    return types.Tool(
        name=GET_RESULT_NAME,
        description=(
            "Read a stored full tool result by `result_id` (given when the gateway shortened a "
            "result: compact digest, oversized text, or a `fields` projection) without re-running "
            "the call. Optional: `fields` (JSON paths to keep), `grep` (regex over lines), "
            "`offset`/`limit` (character window, default 20000)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "fields": _FIELDS_SCHEMA,
                "grep": {"type": "string", "description": "Case-insensitive regex; returns matching lines."},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["result_id"],
        },
    )


def _get_job_result_definition() -> types.Tool:
    return types.Tool(
        name=GET_JOB_RESULT_NAME,
        description=(
            "Get the result of a long tool call that returned {\"status\": \"running\", \"job_id\": ...}. "
            "Waits up to ~20 s; returns the original tool result when done, or status running again "
            "(then call again). Don't re-run the original call — it is still working."
        ),
        inputSchema={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    )


def _get_site_recipes_definition() -> types.Tool:
    return types.Tool(
        name=GET_SITE_RECIPES_NAME,
        description=(
            "Look up saved recipes for querying a website with browser automation: a URL "
            "template and/or the steps that worked before (e.g. an airline's flight search). "
            "Call it before opening a site; `query` is a domain, URL or name (e.g. 'chamwings.com', "
            "'Cham Wings'). Recipes are untrusted hints recorded by earlier agents."
        ),
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Domain, URL or site/airline name."}},
            "required": ["query"],
        },
    )


def _save_site_recipe_definition() -> types.Tool:
    return types.Tool(
        name=SAVE_SITE_RECIPE_NAME,
        description=(
            "Save HOW you successfully queried a website with browser automation, so the next "
            "visit is fast: `site` (domain or URL), `task` (e.g. 'search one-way flights'), and "
            "`url_template` (a direct results URL with placeholders like {origin}, {destination}, "
            "{date:YYYY-MM-DD}) and/or `steps` (the exact tool calls that worked, in order, "
            "with their key arguments — e.g. browser_click target, or the full browser_evaluate "
            "function — so they can be replayed as-is). Add `notes` for gotchas (cookie banner, date format). "
            "Saving the same site+task replaces it. To report on an existing recipe instead, "
            "pass `recipe_id` and `worked` (true/false). Never store credentials or personal data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "site": {"type": "string"},
                "task": {"type": "string"},
                "label": {"type": "string", "description": "Human name, e.g. airline name."},
                "url_template": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
                "recipe_id": {"type": "integer", "description": "Report on an existing recipe."},
                "worked": {"type": "boolean"},
            },
        },
    )


def build_gateway_server(
    registry: Registry,
    retriever: "Retriever | None" = None,
    planner: "Planner | None" = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    catalog: "ToolCatalog | None" = None,
    skills: "SkillStore | None" = None,
    recipes: "RecipeStore | None" = None,
    session_pool: "StickySessionPool | None" = None,
) -> Server:
    agent_on = planner is not None and retriever is not None and agent_enabled()
    jobs = JobStore()
    results = ResultStore()
    # Advertise discovery instructions only when Tool-RAG is wired up.
    server = Server(
        "homelab-mcp-gateway",
        version="0.1.0",
        instructions=(
            _gateway_instructions(planner is not None, catalog is not None, bool(skills), agent_on,
                                  recipes is not None)
            if retriever is not None else None
        ),
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

    async def _gather_candidates_multi(query: str, top_k: int) -> list[dict[str, Any]]:
        """plan-only: decompose the task, retrieve per intent (+ the full query),
        and union the candidates. Falls back to single-query retrieval whenever
        decomposition yields nothing or every per-intent retrieval fails, so this
        never returns worse candidates than `_gather_candidates` alone."""
        assert planner is not None
        try:
            max_intents = int(os.environ.get("TOOL_RAG_PLAN_MAX_INTENTS", DEFAULT_PLAN_MAX_INTENTS))
        except (TypeError, ValueError):
            max_intents = DEFAULT_PLAN_MAX_INTENTS
        try:
            per_intent_k = int(os.environ.get("TOOL_RAG_PLAN_PER_INTENT_K", DEFAULT_PLAN_PER_INTENT_K))
        except (TypeError, ValueError):
            per_intent_k = DEFAULT_PLAN_PER_INTENT_K

        try:
            intents = await planner.decompose(query)
        except Exception:
            logger.debug("plan decompose failed; single-query candidates", exc_info=True)
            intents = []
        intents = [s for s in intents if isinstance(s, str) and s.strip()][:max_intents]
        if not intents:
            single, _ = await _gather_candidates(query, top_k)
            return single

        queries = [query] + intents  # full query always included as a base intent
        settled = await asyncio.gather(
            *[_gather_candidates(q, per_intent_k) for q in queries],
            return_exceptions=True,
        )
        lists: list[list[dict[str, Any]]] = []
        for s in settled:
            if isinstance(s, BaseException):
                logger.debug("per-intent retrieval failed", exc_info=s)
                continue
            cands, _fallback = s
            lists.append(cands)
        merged = _union_candidates(lists, top_k)
        if not merged:
            single, _ = await _gather_candidates(query, top_k)
            return single
        return merged

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

        # Advertise a short teaser description on the shortlist (full text stays in
        # describe_tool + the embedding). Done here, not in _gather_candidates, so
        # the planner's candidate input is untouched.
        for r in results:
            r["description"] = _shortlist_description(r["description"])

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
        if skills:
            hits = await skills.match(query, policy)
            if hits:
                payload["related_skill"] = hits[0][0].brief()
                payload["instructions"] += (
                    f" A playbook exists for this kind of task: call `{GET_SKILL_NAME}` with "
                    f'{{"name": "{hits[0][0].name}"}} and follow it.'
                )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))]
        )

    async def handle_get_skill(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Return a playbook visible to this key."""
        if not skills:
            return _err_tool("No skills are configured on this gateway.")
        name = (arguments or {}).get("name")
        if not name or not isinstance(name, str):
            return _err_tool("get_skill requires a string `name`.")
        skill = skills.get(name, get_policy())
        if skill is None:
            avail = ", ".join(s.name for s in skills.visible(get_policy())) or "none"
            return _err_tool(f"Unknown skill {name!r}. Available: {avail}.")
        return types.CallToolResult(content=[types.TextContent(
            type="text", text=f"# Skill: {skill.name}\n\n{skill.body}")])

    def _agent_tools() -> list[AgentTool]:
        """Tools for the delegate sub-agent: the meta-tool cores, run inside the
        caller's request (same policy checks as direct calls). No nested delegate."""
        async def find(args: dict[str, Any]) -> str:
            q = args.get("query")
            if not isinstance(q, str) or not q:
                return "ERROR: find_tools needs a `query` string."
            res, _ = await _gather_candidates(q, int(args.get("top_k") or 6))
            for r in res:
                r["description"] = _shortlist_description(r["description"])
            return json.dumps(res, ensure_ascii=False)

        async def describe(args: dict[str, Any]) -> str:
            res = await handle_describe_tool(args)
            text = "\n".join(c.text for c in res.content if isinstance(c, types.TextContent))
            return f"ERROR: {text}" if res.isError else text

        async def run(args: dict[str, Any]) -> str:
            name = args.get("call_name")
            if not isinstance(name, str) or name in _META_TOOL_NAMES:
                return "ERROR: run_tool needs the `call_name` of an upstream tool from find_tools."
            tool_args = args.get("arguments") or {}
            if not isinstance(tool_args, dict):
                return "ERROR: `arguments` must be an object."
            res = await _dispatch_tool(name, tool_args, full=args.get("full") is True,
                                       fields=args.get("fields") if isinstance(args.get("fields"), list) else None)
            text = "\n".join(c.text for c in res.content if isinstance(c, types.TextContent))
            if res.isError:
                return f"ERROR: {text or 'tool returned an error'}"
            # Text once (FastMCP also mirrors it into structuredContent — don't send both).
            return text or json.dumps(res.structuredContent, ensure_ascii=False)

        async def skill(args: dict[str, Any]) -> str:
            res = await handle_get_skill(args)
            text = "\n".join(c.text for c in res.content if isinstance(c, types.TextContent))
            return f"ERROR: {text}" if res.isError else text

        async def via(handler, args: dict[str, Any]) -> str:
            res = await handler(args)
            text = "\n".join(c.text for c in res.content if isinstance(c, types.TextContent))
            return f"ERROR: {text}" if res.isError else text

        obj = {"type": "object"}
        extra: list[AgentTool] = [AgentTool(
            "get_result", _get_result_definition().description or "", _get_result_definition().inputSchema,
            lambda a: via(handle_get_result, a))]
        if recipes is not None:
            extra.append(AgentTool(
                "get_site_recipes", "Saved recipes (URL template / steps) for querying a website; call before browsing a site.",
                {**obj, "properties": {"query": {"type": "string"}}, "required": ["query"]},
                lambda a: via(handle_get_site_recipes, a)))
            if recipes.can_write(get_policy()):
                extra.append(AgentTool(
                    "save_site_recipe", _save_site_recipe_definition().description or "",
                    _save_site_recipe_definition().inputSchema, lambda a: via(handle_save_site_recipe, a)))
        return extra + [
            AgentTool("find_tools", "Find tools for a capability (natural-language query). Returns call_name, description, input_schema.",
                      {**obj, "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}, find),
            AgentTool("describe_tool", "Full input_schema and description of one tool.",
                      {**obj, "properties": {"call_name": {"type": "string"}}, "required": ["call_name"]}, describe),
            AgentTool("run_tool", "Execute a tool by call_name with arguments matching its input_schema.",
                      {**obj, "properties": {"call_name": {"type": "string"}, "arguments": {"type": "object"},
                                         "full": {"type": "boolean", "description": "raw result instead of the compact digest"},
                                         "fields": _FIELDS_SCHEMA},
                       "required": ["call_name", "arguments"]}, run),
            AgentTool("get_skill", "Fetch a playbook by name.",
                      {**obj, "properties": {"name": {"type": "string"}}, "required": ["name"]}, skill),
        ]

    async def handle_delegate(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Run a whole task with the gateway-side sub-agent (tool_rag/agent.py)."""
        if not agent_on:
            return _err_tool("The delegate agent is not enabled on this gateway.")
        args = arguments or {}
        task = args.get("task")
        if not task or not isinstance(task, str):
            return _err_tool("delegate requires a non-empty string `task`.")
        policy = get_policy()
        if not policy.servers:
            return _err_tool("No servers are granted to this API key.")
        chosen = None
        if skills:
            if isinstance(args.get("skill"), str) and args["skill"]:
                chosen = skills.get(args["skill"], policy)
            else:
                hits = await skills.match(task, policy)
                chosen = hits[0][0] if hits else None
        try:
            max_steps = int(args.get("max_steps") or 15)
        except (TypeError, ValueError):
            max_steps = 15

        progress = None
        try:
            ctx = server.request_context
            token = ctx.meta.progressToken if ctx.meta else None
        except LookupError:
            token = None
        if token is not None:
            async def progress(done: int, total: int, message: str) -> None:
                try:
                    await ctx.session.send_progress_notification(token, done, total, message)
                except Exception:
                    logger.debug("progress notification failed", exc_info=True)

        # Recipes for sites the task names go into the prompt up front (and count as
        # shown for this session, so they aren't injected again on navigate).
        site_ctx = None
        if recipes is not None:
            named = {s for d in re.findall(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", task) if (s := site_of(d))}
            found = [r for s in sorted(named) for r in recipes.find(s)]
            if found:
                _mark_recipes_shown({r["site"] for r in found})
                site_ctx = RecipeStore.as_context(found)
        result = await run_agent(planner, task, _agent_tools(), chosen.body if chosen else None,
                                 chosen.name if chosen else None, max_steps, progress, site_ctx)
        logger.info("delegate: %s after %d tool calls (skill=%s)", result.stopped_reason, len(result.steps), result.skill)
        return types.CallToolResult(content=[types.TextContent(
            type="text", text=json.dumps(result.to_dict(), ensure_ascii=False))])

    async def handle_get_site_recipes(arguments: dict[str, Any] | None) -> types.CallToolResult:
        if recipes is None:
            return _err_tool("Site recipes are not enabled on this gateway.")
        q = (arguments or {}).get("query")
        if not q or not isinstance(q, str):
            return _err_tool("get_site_recipes requires a string `query`.")
        found = recipes.find(q)
        if found:
            _mark_recipes_shown({r["site"] for r in found})
            text = RecipeStore.as_context(found)
        else:
            text = (f"No saved recipe for {q!r}. Explore the site; once you get the data, save how "
                    f"with `{SAVE_SITE_RECIPE_NAME}`.")
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    async def handle_save_site_recipe(arguments: dict[str, Any] | None) -> types.CallToolResult:
        if recipes is None:
            return _err_tool("Site recipes are not enabled on this gateway.")
        a = arguments or {}
        policy = get_policy()
        try:
            if a.get("recipe_id") is not None and not a.get("steps") and not a.get("url_template"):
                rec = recipes.report(policy, int(a["recipe_id"]), bool(a.get("worked", True)))
                msg = "Recorded."
            else:
                steps = a.get("steps")
                if isinstance(steps, str):
                    steps = [steps]
                rec = recipes.save(policy, str(a.get("site") or ""), str(a.get("task") or ""), steps,
                                   str(a.get("url_template") or ""), str(a.get("label") or ""),
                                   str(a.get("notes") or ""))
                msg = "Saved; it will be shown the next time an agent visits this site."
        except (RecipeError, ValueError, TypeError) as e:
            return _err_tool(f"Not saved: {e}")
        logger.info("site recipe %s: %s / %s by %s", rec["id"], rec["site"], rec["task"], policy.key_id)
        return types.CallToolResult(content=[types.TextContent(
            type="text", text=json.dumps({"message": msg, "recipe": RecipeStore.brief(rec)}, ensure_ascii=False))])

    def _mark_recipes_shown(sites: set[str]) -> set[str]:
        """Record sites whose recipes this session has seen; returns the ones that are new."""
        try:
            seen = _session_recipe_sites.setdefault(server.request_context.session, set())
        except LookupError:
            return set(sites)
        new = set(sites) - seen
        seen.update(sites)
        return new

    def _with_recipes(res: types.CallToolResult, arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Append saved recipes for any site this successful call just touched (once per session)."""
        if recipes is None or res.isError:
            return res
        sites = {s for u in urls_in(arguments) if (s := site_of(u))}
        new = _mark_recipes_shown(sites) if sites else set()
        if not new:
            return res
        extra: list[types.TextContent] = []
        found = [r for site in sorted(new) for r in recipes.find(site)]
        if found:
            extra.append(types.TextContent(type="text", text=RecipeStore.as_context(found)))
        unknown = sorted(new - {r["site"] for r in found})
        if unknown and recipes.can_write(get_policy()):
            # Nudge (once per session per site): the agent is the one who learns how the
            # site works, so ask it to record that once it has the data.
            extra.append(types.TextContent(type="text", text=(
                f"[No saved recipe for {', '.join(unknown)} yet. Once you have the data you came for, "
                f"call `{SAVE_SITE_RECIPE_NAME}` with the URL pattern and steps that worked, so the "
                f"next visit is fast.]")))
        if not extra:
            return res
        return res.model_copy(update={"content": [*res.content, *extra]})

    async def handle_browse_tools(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Catalog questions: overview / one category or domain / coverage of a
        problem. Summaries only for non-admin keys (see tool_rag/catalog.py)."""
        if catalog is None:
            return _err_tool("The tool catalog is not enabled on this gateway.")
        args = arguments or {}
        category = args.get("category")
        query = args.get("query")
        if category is not None and not isinstance(category, str):
            return _err_tool("`category` must be a string.")
        if query is not None and not isinstance(query, str):
            return _err_tool("`query` must be a string.")
        if category and query:
            return _err_tool("Pass either `category` or `query`, not both.")
        policy = get_policy()
        if query:
            payload = await catalog.coverage(policy, query)
        elif category:
            payload = catalog.category(policy, category)
        else:
            payload = catalog.overview(policy)
            payload["hint"] = (
                f"Call `{BROWSE_TOOLS_NAME}` with `category` or `query` to drill in, "
                f"or `{FIND_TOOLS_NAME}` with a task description to get specific tools."
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
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
            # Per-key catalog summary in the descriptions, so the agent knows what
            # to expect before searching (list_tools runs per request + policy).
            base = [
                _find_tools_definition(catalog.domain_names(policy) if catalog else None),
                _run_tool_definition(),
                _run_tools_definition(),
                _describe_tool_definition(),
            ]
            if catalog is not None:
                base.append(_browse_tools_definition(catalog.summary_text(policy)))
            visible_skills = skills.visible(policy) if skills else []
            if visible_skills:
                base.append(_get_skill_definition([s.brief() for s in visible_skills]))
            base.append(_get_job_result_definition())
            base.append(_get_result_definition())
            if recipes is not None:
                base.append(_get_site_recipes_definition())
                if recipes.can_write(policy):
                    base.append(_save_site_recipe_definition())
            if agent_on:
                base.append(_delegate_definition())
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

    async def _dispatch_tool(name: str, arguments: dict[str, Any] | None, full: bool = False,
                             fields: list[str] | None = None) -> types.CallToolResult:
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
        if retriever is not None:
            rec = retriever.get_tool(name)
            if rec is not None and rec.status == "quarantined":
                return _err_tool("This tool is new on an external server and pending admin review; it can't be "
                                 "used until approved (GET /tool-rag/reviews).")
        cfg = registry.servers[server_id]
        # Stateful servers (e.g. Playwright's open page) reuse one upstream session per
        # downstream MCP session; everything else gets a fresh session per call.
        downstream = None
        if cfg.stateful and session_pool is not None:
            try:
                downstream = server.request_context.session
            except LookupError:
                downstream = None
        try:
            if downstream is not None:
                res = await session_pool.call_tool(downstream, cfg, orig, arguments)
            else:
                async with open_upstream_session(cfg) as us:
                    res = await us.call_tool(orig, arguments)
        except Exception as e:
            logger.exception("call_tool upstream error")
            return _err_tool(f"Upstream error: {e}")
        res = _shape_result(name, orig, cfg, arguments, res, full, fields)
        return _with_recipes(res, arguments)

    def _shape_result(name: str, orig: str, cfg: Any, arguments: dict[str, Any] | None,
                      res: types.CallToolResult, full: bool, fields: list[str] | None) -> types.CallToolResult:
        """Projection (`fields`), compact digests and oversize truncation. Whenever the
        result is shortened, the full text is kept under a result_id (get_result)."""
        if res.isError or not res.content or not all(isinstance(c, types.TextContent) for c in res.content):
            return res
        text = "\n".join(c.text for c in res.content)
        key_id = get_policy().key_id

        def out(body: str, note: str) -> types.CallToolResult:
            return types.CallToolResult(content=[types.TextContent(type="text", text=body),
                                                 types.TextContent(type="text", text=note)])

        if fields:
            try:
                body, total = view(text, fields=fields, limit=RESULT_MAX_CHARS)
            except ProjectionError as e:
                return out(text if len(text) <= RESULT_MAX_CHARS else text[:RESULT_MAX_CHARS], f"[fields ignored: {e}]")
            rid = results.put(key_id, name, text)
            return out(body, f"[projected {len(text)} -> {total} chars; full result: get_result "
                             f'{{"result_id": "{rid}"}}]')
        if full:
            return res
        if cfg.compact_results:
            digest = compact(orig, arguments, text)
            if digest is not None:
                rid = results.put(key_id, name, text)
                logger.info("compacted %s result: %d -> %d chars (%s)", name, len(text), len(digest), rid)
                try:
                    d = json.loads(digest)
                    d["result_id"] = rid
                    d["note"] = (f"Compacted by the gateway: {d.get('shown')} of {d.get('total')} options. "
                                 f"Details without re-running: get_result with result_id {rid} "
                                 f"(fields / grep / offset).")
                    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(d, ensure_ascii=False))])
                except (json.JSONDecodeError, AttributeError):
                    return out(digest, f'[full result: get_result {{"result_id": "{rid}"}}]')
        if len(text) > RESULT_MAX_CHARS:
            rid = results.put(key_id, name, text)
            logger.info("truncated %s result: %d -> %d chars (%s)", name, len(text), RESULT_MAX_CHARS, rid)
            return out(text[:RESULT_MAX_CHARS],
                       f"[truncated: showing {RESULT_MAX_CHARS} of {len(text)} chars. Rest: get_result "
                       f'{{"result_id": "{rid}", "offset": {RESULT_MAX_CHARS}}} (or use fields / grep)]')
        return res

    async def handle_get_result(arguments: dict[str, Any] | None) -> types.CallToolResult:
        a = arguments or {}
        rid = a.get("result_id")
        if not isinstance(rid, str) or not rid:
            return _err_tool("get_result requires a string `result_id`.")
        stored = results.get(get_policy().key_id, rid)
        if stored is None:
            return _err_tool(f"Unknown or expired result_id {rid!r} (results are kept 30 min, lost on restart).")
        fields = a.get("fields") if isinstance(a.get("fields"), list) else None
        try:
            body, total = view(stored.text, fields=fields, grep=a.get("grep") or None,
                               offset=int(a.get("offset") or 0), limit=int(a.get("limit") or 20000))
        except (ProjectionError, ValueError, TypeError) as e:
            return _err_tool(f"get_result: {e}")
        end = int(a.get("offset") or 0) + len(body)
        more = f" Next: offset {end}." if end < total else ""
        return types.CallToolResult(content=[
            types.TextContent(type="text", text=body),
            types.TextContent(type="text", text=f"[{stored.tool} result {rid}: chars {int(a.get('offset') or 0)}-{end} of {total}.{more}]"),
        ])

    async def handle_run_tool(arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Execution proxy: runs any tool discovered via find_tools."""
        args = arguments or {}
        call_name = args.get("call_name")
        if not call_name or not isinstance(call_name, str):
            return _err_tool("run_tool requires a non-empty string `call_name`.")
        tool_arguments = args.get("arguments") or {}
        if not isinstance(tool_arguments, dict):
            return _err_tool("`arguments` must be a JSON object.")
        fields = args.get("fields") if isinstance(args.get("fields"), list) else None
        return await _dispatch_tool(call_name, tool_arguments, full=args.get("full") is True, fields=fields)

    def _result_to_json(res: types.CallToolResult) -> dict[str, Any]:
        """Flatten a CallToolResult into a JSON-able summary for run_tools. On
        failure always populate `error` so callers see one consistent shape."""
        texts = "\n".join(c.text for c in res.content if isinstance(c, types.TextContent))
        out: dict[str, Any] = {"ok": not res.isError}
        if res.isError:
            out["error"] = texts or "tool returned an error"
            return out
        # Text is the canonical content; structuredContent usually mirrors it (FastMCP),
        # so include it only when there's no text — never both (doubles the size).
        if texts:
            out["text"] = texts
        elif res.structuredContent is not None:
            out["structured"] = res.structuredContent
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
                    res = await _dispatch_tool(call["call_name"], tool_args, full=call.get("full") is True,
                                               fields=call.get("fields") if isinstance(call.get("fields"), list) else None)
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
            top_k = int(args.get("top_k", DEFAULT_PLAN_TOP_K))
        except (TypeError, ValueError):
            top_k = DEFAULT_PLAN_TOP_K
        if not get_policy().servers:
            return _err_tool("No servers are granted to this API key.")
        candidates = await _gather_candidates_multi(query, top_k)
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

    _META_TOOL_NAMES = {
        FIND_TOOLS_NAME, RUN_TOOL_NAME, RUN_TOOLS_NAME, DESCRIBE_TOOL_NAME, PLAN_TOOL_NAME,
        BROWSE_TOOLS_NAME, GET_SKILL_NAME, DELEGATE_NAME, GET_SITE_RECIPES_NAME, SAVE_SITE_RECIPE_NAME,
        GET_JOB_RESULT_NAME, GET_RESULT_NAME,
    }

    @server.call_tool(validate_input=False)
    async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        # One line per call so logs show which meta-tool (or upstream tool) ran —
        # CallToolRequest alone is generic. Keeps find_tools/run_tools/plan visible.
        logger.info("call_tool: %s (%s)", name, "meta" if name in _META_TOOL_NAMES else "upstream")
        if name == GET_JOB_RESULT_NAME:
            job_id = (arguments or {}).get("job_id")
            if not isinstance(job_id, str) or not job_id:
                return _err_tool("get_job_result requires a string `job_id`.")
            return await jobs.result(get_policy().key_id, job_id, 20.0)
        # Anything slower than the sync budget continues as a background job.
        return await jobs.run(get_policy().key_id, name, _route_call(name, arguments))

    async def _route_call(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
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
        if name == BROWSE_TOOLS_NAME:
            return await handle_browse_tools(arguments)
        if name == GET_SKILL_NAME:
            return await handle_get_skill(arguments)
        if name == DELEGATE_NAME:
            return await handle_delegate(arguments)
        if name == GET_SITE_RECIPES_NAME:
            return await handle_get_site_recipes(arguments)
        if name == GET_RESULT_NAME:
            return await handle_get_result(arguments)
        if name == SAVE_SITE_RECIPE_NAME:
            return await handle_save_site_recipe(arguments)
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
