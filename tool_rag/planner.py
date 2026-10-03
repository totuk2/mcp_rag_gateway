"""Optional LLM-backed planning for the gateway.

Turns a natural-language task + a set of candidate tools (from the retriever)
into a structured, multi-step plan: ordered steps with dependencies and parallel
groups. The plan is advisory — the client model fills concrete arguments and
executes each group via `run_tools`; the gateway never auto-executes it.

Self-hosted: points at any OpenAI-shaped chat-completions endpoint (e.g. your own
Ollama), mirroring the ApiEmbedder pattern. Configured via TOOL_RAG_PLANNER
(off|llm). Default off — no behaviour change unless enabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a planning assistant for an MCP tool gateway. Given a user TASK and a "
    "list of AVAILABLE TOOLS, produce a concise execution plan. Respond with ONLY a "
    "JSON object, no prose, matching exactly:\n"
    '{"steps": [{"id": "s1", "call_name": "<one of the available call_names>", '
    '"arguments_hint": {"<arg>": "<what to put here>"}, "depends_on": ["s0"], '
    '"group": 0, "rationale": "<why>"}], "notes": "<overall guidance>", '
    '"missing": ["<capability not covered by any available tool>"]}\n'
    "Rules: only use call_name values that appear in AVAILABLE TOOLS; never invent "
    "tools. `group` is an integer — steps sharing a group have no inter-dependencies "
    "and may run in parallel; `depends_on` lists ids of steps that must finish first. "
    "In arguments_hint describe what each argument should contain (placeholders, not "
    "real secret values). If the task needs something no tool provides, list it in "
    "`missing`. Keep the plan minimal."
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a user TASK into short, independent search phrases — one per distinct "
    "capability or tool the task needs (e.g. 'search arxiv for papers', 'take a "
    "screenshot of a web page'). Each phrase is a standalone query used to look up the "
    "right tool, so name the action and object, not the whole task. Respond with ONLY "
    'a JSON object: {"intents": ["phrase", ...]}. Use 1-6 phrases; fewer is fine for '
    "simple tasks. Do not invent capabilities the task does not mention."
)


_TAXONOMY_SYSTEM_PROMPT = (
    "You maintain the category taxonomy for a catalog of tools exposed to AI agents. "
    "Given the DOMAINS (tool servers with their tools), produce categories that let an "
    "agent ask 'what kinds of tools do I have?' or 'do I have a tool for X?'. Respond "
    'with ONLY a JSON object: {"categories": [{"name": "kebab-case-name", '
    '"description": "<one short sentence: what tools in it do>"}]}. Rules: '
    "categories describe capabilities (e.g. web-browsing, research-papers, "
    "circuit-simulation), not server names; a category may span several servers; "
    "aim for 5-20 categories, fewer for a small catalog; every category must fit at "
    "least one listed tool. If PREVIOUS categories are given, KEEP their exact names "
    "when they still fit, add new ones only for capabilities they do not cover, and "
    "drop ones no tool fits anymore. REQUIRED categories must be included as named."
)

_TAGGING_SYSTEM_PROMPT = (
    "You assign catalog categories to the tools of one tool server. Respond with ONLY "
    'a JSON object: {"description": "<one sentence: what this server is for>", '
    '"tags": {"<tool name>": ["category", ...]}}. Rules: give every listed tool 1-3 '
    "category names chosen ONLY from CATEGORIES (exact names); prefer the most specific "
    "fitting ones. The description summarizes the server's purpose for an agent "
    "deciding whether to use it (no marketing, no tool lists)."
)


# Extra attempts for a rate-limited / transiently failing chat call.
_CHAT_RETRIES = 4

_COVERAGE_SYSTEM_PROMPT = (
    "You judge whether an AI agent's available tools cover a problem. Input: the "
    "agent's QUESTION (any language), its CATALOG (domains = tool servers with "
    "descriptions and categories), and the TOP SEARCH MATCHES from a semantic search "
    "(may be noisy or miss things). Respond with ONLY a JSON object: "
    '{"verdict": "strong|weak|none", "domains": ["<domain>"], "categories": '
    '["<category>"], "reason": "<one short sentence>", "suggested_query": '
    '"<short English search phrase naming the action and object>"}. '
    "strong = a domain is built for this capability; weak = only partially or "
    "indirectly — in particular, if the only fit is a GENERIC tool (web-browser "
    "automation, running arbitrary code) that could do it by hand on some website or "
    "script, the verdict is at most weak, never strong; none = nothing fits. "
    "Use only domain and category names from CATALOG. Judge by what the domains do, "
    "not by search scores alone."
)


def _compact_tool(c: dict[str, Any]) -> dict[str, Any]:
    """Trim a candidate to the fields the planner needs (keeps the prompt small)."""
    schema = c.get("input_schema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    arg_names = sorted(props.keys()) if isinstance(props, dict) else []
    required = schema.get("required") if isinstance(schema, dict) else None
    return {
        "call_name": c.get("call_name"),
        "tool_type": c.get("tool_type"),
        "description": (c.get("description") or "").strip()[:400],
        "args": arg_names,
        "required": required if isinstance(required, list) else [],
    }


class Planner(ABC):
    """Abstract task -> plan generator."""

    @abstractmethod
    async def plan(self, query: str, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Return {"steps": [...], "notes": str, "missing": [...]} for the task."""

    @abstractmethod
    async def decompose(self, query: str) -> list[str]:
        """Split a task into short per-capability search phrases (best-effort)."""

    async def derive_taxonomy(
        self, domains: Sequence[dict[str, Any]], previous: Sequence[dict[str, str]], required: Sequence[str]
    ) -> list[dict[str, Any]] | None:
        """Catalog categories [{name, description}] for the current tool set, or
        None when unavailable. Unvalidated — the caller validates."""
        return None

    async def assign_tags(
        self, server_id: str, server_hint: str, tools: Sequence[dict[str, str]], categories: Sequence[dict[str, str]]
    ) -> dict[str, Any] | None:
        """{"description": str, "tags": {tool_name: [category, ...]}} for one
        server, or None when unavailable. Unvalidated — the caller validates."""
        return None

    async def judge_coverage(
        self, question: str, catalog: dict[str, Any], matches: Sequence[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """{"verdict", "domains", "categories", "reason", "suggested_query"} or
        None when unavailable. Unvalidated — the caller validates."""
        return None


class LlmPlanner(Planner):
    """Calls an OpenAI-shaped chat-completions endpoint to generate the plan."""

    def __init__(
        self,
        url: str = "",
        model: str = "",
        api_key: str = "",
        temperature: float | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._url = url or os.environ.get("TOOL_RAG_PLANNER_URL", "")
        self._model = model or os.environ.get("TOOL_RAG_PLANNER_MODEL", "")
        self._api_key = api_key or os.environ.get("TOOL_RAG_PLANNER_API_KEY", "")
        if temperature is None:
            try:
                temperature = float(os.environ.get("TOOL_RAG_PLANNER_TEMPERATURE", "0.1"))
            except ValueError:
                temperature = 0.1
        self._temperature = temperature
        self._timeout = timeout

    async def plan(self, query: str, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tools = [_compact_tool(c) for c in candidates]
        valid_names = {t["call_name"] for t in tools}
        user = (
            f"TASK:\n{query}\n\nAVAILABLE TOOLS (JSON):\n{json.dumps(tools)}\n\n"
            "Return the plan JSON now."
        )
        content = await self._chat([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
        raw = _parse_json_object(content)
        return _normalize_plan(raw, valid_names, candidates)

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        """POST an OpenAI-shaped chat-completions request and return message content."""
        data = await self._post({"messages": messages, "response_format": {"type": "json_object"}})
        return data["choices"][0]["message"]["content"]

    async def chat_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                         tool_choice: str = "auto") -> dict[str, Any]:
        """Tool-calling chat turn (OpenAI `tools` format); returns the assistant message
        dict (`content` and/or `tool_calls`). Backs the `delegate` sub-agent."""
        body: dict[str, Any] = {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        data = await self._post(body)
        return data["choices"][0]["message"]

    async def _post(self, extra: dict[str, Any]) -> dict[str, Any]:
        if not self._url:
            raise RuntimeError("LlmPlanner: TOOL_RAG_PLANNER_URL not configured")
        if not self._model:
            raise RuntimeError("LlmPlanner: TOOL_RAG_PLANNER_MODEL not configured")
        import httpx

        body: dict[str, Any] = {"model": self._model, "temperature": self._temperature, **extra}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Hosted endpoints (e.g. OpenRouter) rate-limit bursts — the catalog
            # refresh fires several calls at once. Retry 429/5xx briefly,
            # honouring a small Retry-After.
            for attempt in range(_CHAT_RETRIES + 1):
                resp = await client.post(self._url, json=body, headers=headers)
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == _CHAT_RETRIES:
                    break
                try:
                    delay = min(float(resp.headers.get("retry-after", "")), 10.0)
                except ValueError:
                    delay = 2.0 * 2 ** attempt  # 2, 4, 8, 16 s
                await asyncio.sleep(delay)
            resp.raise_for_status()
            return resp.json()

    async def decompose(self, query: str) -> list[str]:
        """Return short per-capability search phrases for the task, or [] on any
        failure (parse, network, or config) — callers fall back to single-query
        retrieval, so decomposition must never break planning."""
        try:
            content = await self._chat([
                {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"TASK:\n{query}\n\nReturn the intents JSON now."},
            ])
            raw = _parse_json_object(content)
        except Exception:
            logger.debug("planner decompose failed; returning no intents", exc_info=True)
            return []
        intents = raw.get("intents")
        if not isinstance(intents, list):
            return []
        return [s.strip() for s in intents if isinstance(s, str) and s.strip()]

    async def derive_taxonomy(
        self, domains: Sequence[dict[str, Any]], previous: Sequence[dict[str, str]], required: Sequence[str]
    ) -> list[dict[str, Any]] | None:
        """Best-effort like decompose: any failure -> None (caller keeps the
        previous taxonomy / falls back), so a refresh never breaks startup."""
        user = (
            f"DOMAINS (JSON):\n{json.dumps(list(domains), ensure_ascii=False)}\n\n"
            f"PREVIOUS categories (JSON):\n{json.dumps(list(previous), ensure_ascii=False)}\n\n"
            f"REQUIRED category names: {json.dumps(list(required))}\n\n"
            "Return the categories JSON now."
        )
        try:
            raw = _parse_json_object(await self._chat([
                {"role": "system", "content": _TAXONOMY_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]))
        except Exception:
            logger.warning("catalog: taxonomy derivation failed", exc_info=True)
            return None
        cats = raw.get("categories")
        return cats if isinstance(cats, list) else None

    async def assign_tags(
        self, server_id: str, server_hint: str, tools: Sequence[dict[str, str]], categories: Sequence[dict[str, str]]
    ) -> dict[str, Any] | None:
        user = (
            f"SERVER: {server_id}\n"
            f"SERVER INFO: {server_hint or '(none)'}\n\n"
            f"CATEGORIES (JSON):\n{json.dumps(list(categories), ensure_ascii=False)}\n\n"
            f"TOOLS (JSON):\n{json.dumps(list(tools), ensure_ascii=False)}\n\n"
            "Return the JSON now."
        )
        try:
            raw = _parse_json_object(await self._chat([
                {"role": "system", "content": _TAGGING_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]))
        except Exception:
            logger.warning("catalog: tagging failed for server %s", server_id, exc_info=True)
            return None
        return raw if isinstance(raw.get("tags"), dict) else None

    async def judge_coverage(
        self, question: str, catalog: dict[str, Any], matches: Sequence[dict[str, Any]]
    ) -> dict[str, Any] | None:
        user = (
            f"QUESTION: {question}\n\n"
            f"CATALOG (JSON):\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
            f"TOP SEARCH MATCHES (JSON):\n{json.dumps(list(matches), ensure_ascii=False)}\n\n"
            "Return the JSON now."
        )
        try:
            raw = _parse_json_object(await self._chat([
                {"role": "system", "content": _COVERAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]))
        except Exception:
            logger.warning("catalog: coverage judgement failed", exc_info=True)
            return None
        return raw if raw.get("verdict") in ("strong", "weak", "none") else None


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse the model's content as a JSON object, tolerating stray prose/fences."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise RuntimeError("planner did not return valid JSON")


def _normalize_plan(
    raw: dict[str, Any],
    valid_names: set[str],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the model's plan against the real candidate set: drop steps that
    reference unknown tools (record them in `missing`) and stamp each step with the
    tool_type/server so the caller can see which steps are writes."""
    type_by_name = {c["call_name"]: c.get("tool_type") for c in candidates}
    server_by_name = {c["call_name"]: c.get("server") for c in candidates}
    # Carry each tool's arg names/required into the step so the client can fill
    # arguments without guessing (the authoritative full schema is one
    # describe_tool call away). Cheap — we already hold the candidate schemas.
    spec_by_name: dict[str, dict[str, Any]] = {}
    for c in candidates:
        compact = _compact_tool(c)
        spec_by_name[c["call_name"]] = {"args": compact["args"], "required": compact["required"]}
    steps_raw = raw.get("steps")
    steps_in = steps_raw if isinstance(steps_raw, list) else []
    missing_raw = raw.get("missing")
    missing = list(missing_raw) if isinstance(missing_raw, list) else []
    steps_out: list[dict[str, Any]] = []
    for i, s in enumerate(steps_in):
        if not isinstance(s, dict):
            continue
        call_name = s.get("call_name")
        if call_name not in valid_names:
            if call_name:
                missing.append(f"unknown tool referenced: {call_name}")
            continue
        spec = spec_by_name.get(call_name, {"args": [], "required": []})
        steps_out.append({
            "id": s.get("id") or f"s{i}",
            "call_name": call_name,
            "server": server_by_name.get(call_name),
            "tool_type": type_by_name.get(call_name),
            "args": spec["args"],
            "required": spec["required"],
            "arguments_hint": s.get("arguments_hint") or {},
            "depends_on": s.get("depends_on") if isinstance(s.get("depends_on"), list) else [],
            "group": s.get("group") if isinstance(s.get("group"), int) else 0,
            "rationale": s.get("rationale") or "",
        })
    return {
        "steps": steps_out,
        "notes": raw.get("notes") if isinstance(raw.get("notes"), str) else "",
        "missing": missing,
    }


def create_planner() -> Planner | None:
    """Factory: instantiate the planner based on TOOL_RAG_PLANNER env var.

    Returns None when disabled (the default), leaving the gateway unchanged.
    """
    kind = os.environ.get("TOOL_RAG_PLANNER", "off").lower()
    if kind in ("llm", "on", "1", "true"):
        return LlmPlanner()
    return None
