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
        if not self._url:
            raise RuntimeError("LlmPlanner: TOOL_RAG_PLANNER_URL not configured")
        if not self._model:
            raise RuntimeError("LlmPlanner: TOOL_RAG_PLANNER_MODEL not configured")
        import httpx

        tools = [_compact_tool(c) for c in candidates]
        valid_names = {t["call_name"] for t in tools}
        user = (
            f"TASK:\n{query}\n\nAVAILABLE TOOLS (JSON):\n{json.dumps(tools)}\n\n"
            "Return the plan JSON now."
        )
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        raw = _parse_json_object(content)
        return _normalize_plan(raw, valid_names, candidates)


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
        steps_out.append({
            "id": s.get("id") or f"s{i}",
            "call_name": call_name,
            "server": server_by_name.get(call_name),
            "tool_type": type_by_name.get(call_name),
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
