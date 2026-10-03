"""Gateway-side sub-agent behind the `delegate` meta-tool.

A bounded tool-calling loop on the planner model: the caller hands over a whole
task ("find the cheapest way from Gdańsk to Aleppo next week"), the sub-agent
discovers and runs tools until it can answer, guided by a skill (playbook)
when one is given or matches the task. Useful when the client model gives up
too early or lacks the patience for multi-step research.

The tools are injected by the caller (gateway/server.py) as async functions
that run inside the original request, so every upstream call is checked
against the calling key's policy exactly like a direct call. No nesting:
the sub-agent cannot call `delegate`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from gateway.recipes import site_of, urls_in

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 15
MAX_STEPS_CAP = 30
DEFAULT_TIMEOUT_S = 240.0
# Tool results are truncated before going back to the model (context budget).
RESULT_CHARS = 9000

_SYSTEM = (
    "You are a research sub-agent inside an MCP tool gateway. Complete the TASK by "
    "calling tools, then answer.\n"
    "- Discover tools with `find_tools` (natural-language query), execute them with "
    "`run_tool` (call_name + arguments matching input_schema); `describe_tool` gives a "
    "full schema; `get_skill` returns a playbook.\n"
    "- Be persistent: if a search returns nothing, try alternatives (other tools, nearby "
    "places, other dates, other phrasings) before concluding something is impossible.\n"
    "- Prefer one well-chosen composite tool over many small calls. Run independent "
    "calls in the same turn.\n"
    "- Never book, buy, pay or submit anything; read-only research only.\n"
    "- Websites: before browsing a site, check `get_site_recipes` (known recipes are also "
    "appended to results when you navigate). After you successfully get the data from a "
    "site by browsing, call `save_site_recipe` with the URL pattern and steps that worked; "
    "if a recipe failed, report it (recipe_id, worked=false) and save the fix.\n"
    "- When done, reply WITHOUT tool calls: the final answer in the language of the TASK, "
    "in Markdown, with sources/links, and a short note on what could not be checked.\n"
    "You have at most {steps} tool turns and about {seconds} seconds."
)

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class AgentResult:
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "answered"
    skill: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "stopped_reason": self.stopped_reason, "skill": self.skill,
                "steps": self.steps}


def agent_enabled() -> bool:
    return os.environ.get("TOOL_RAG_AGENT", "off").lower() in ("1", "true", "on", "llm")


def _timeout() -> float:
    try:
        return float(os.environ.get("TOOL_RAG_AGENT_TIMEOUT", DEFAULT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _summary(text: str, n: int = 160) -> str:
    return " ".join(text.split())[:n]


async def run_agent(
    planner: Any,
    task: str,
    tools: list[AgentTool],
    skill_body: str | None = None,
    skill_name: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    progress: Callable[[int, int, str], Awaitable[None]] | None = None,
    site_recipes: str | None = None,
) -> AgentResult:
    max_steps = max(1, min(int(max_steps), MAX_STEPS_CAP))
    budget = _timeout()
    deadline = time.monotonic() + budget
    by_name = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    system = _SYSTEM.format(steps=max_steps, seconds=int(budget))
    if skill_body:
        system += f"\n\n# Playbook: {skill_name}\n\n{skill_body}"
    if site_recipes:
        system += f"\n\n# Known site recipes (start from these)\n\n{site_recipes}"
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    result = AgentResult(answer="", skill=skill_name)
    # Site-recipe bookkeeping: sites browsed successfully, sites that already had a
    # recipe (injected into a result) and sites the agent saved a recipe for.
    browsed: set[str] = set()
    known: set[str] = set()
    saved: set[str] = set()

    async def call_llm(tool_choice: str = "auto") -> dict[str, Any]:
        left = deadline - time.monotonic()
        return await asyncio.wait_for(planner.chat_tools(messages, specs, tool_choice), timeout=max(left, 20))

    async def execute(call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        tool = by_name.get(name)
        if tool is None and "__" in name and "run_tool" in by_name:
            # Models often call a discovered upstream tool directly by its call_name;
            # route it through run_tool instead of wasting a turn on an error.
            args, name, tool = {"call_name": name, "arguments": args}, "run_tool", by_name["run_tool"]
        step = {"tool": name, "args": args}
        if tool is None:
            out, ok = f"Unknown tool {name!r}. Available: {', '.join(by_name)}.", False
        else:
            try:
                left = deadline - time.monotonic()
                out = await asyncio.wait_for(tool.fn(args), timeout=max(left, 5))
                ok = not out.startswith("ERROR:")
            except asyncio.TimeoutError:
                out, ok = "ERROR: tool call exceeded the remaining time budget.", False
            except Exception as e:
                out, ok = f"ERROR: {type(e).__name__}: {e}", False
        step.update(ok=ok, summary=_summary(out))
        result.steps.append(step)
        if ok and name == "run_tool":
            sites = {s for u in urls_in(args.get("arguments") or {}) if (s := site_of(u))}
            browsed.update(sites)
            if "[Saved site recipes" in out:
                known.update(sites)
        elif ok and name == "save_site_recipe" and (s := site_of(str(args.get("site") or ""))):
            saved.add(s)
        if len(out) > RESULT_CHARS:
            out = out[:RESULT_CHARS] + f"\n…[truncated {len(out) - RESULT_CHARS} chars]"
        return {"role": "tool", "tool_call_id": call.get("id", ""), "content": out}

    async def _record_recipes(final_msg: dict[str, Any]) -> None:
        """Successful browsing of a site with no recipe -> one forced save turn, so
        the next visit is fast even if the model forgot (the answer is kept)."""
        todo = sorted(browsed - known - saved)
        if not todo or "save_site_recipe" not in by_name or time.monotonic() > deadline - 15:
            return
        messages.append({"role": "assistant", "content": final_msg.get("content") or ""})
        messages.append({"role": "user", "content": (
            f"Before finishing: you successfully got data from {', '.join(todo)} by browsing. Call "
            f"save_site_recipe for each site: the task, a url_template if a results URL is reusable "
            f"(with placeholders), and the steps that worked as the EXACT tool calls with their key "
            f"arguments (e.g. the full browser_evaluate function, the click target) so they can be "
            f"replayed as-is. Describe HOW, not the answer.")})
        try:
            msg = await asyncio.wait_for(planner.chat_tools(messages, [by_name["save_site_recipe"].spec()],
                                                            "required"), timeout=45)
            for call in msg.get("tool_calls") or []:
                await execute(call)
        except Exception:
            logger.info("delegate: recipe save turn failed", exc_info=True)

    for turn in range(1, max_steps + 1):
        if time.monotonic() >= deadline - 10:
            result.stopped_reason = "time_budget"
            break
        try:
            msg = await call_llm()
        except Exception as e:
            logger.exception("delegate: model call failed")
            result.answer = f"Sub-agent stopped: model call failed ({type(e).__name__}: {e})."
            result.stopped_reason = "model_error"
            return result
        calls = msg.get("tool_calls") or []
        if not calls:
            result.answer = (msg.get("content") or "").strip()
            await _record_recipes(msg)
            return result
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        if progress is not None:
            names = ", ".join((c.get("function") or {}).get("name", "?") for c in calls)
            await progress(turn, max_steps, f"step {turn}: {names}")
        messages.extend(await asyncio.gather(*(execute(c) for c in calls)))
    else:
        result.stopped_reason = "max_steps"

    # Out of steps/time: ask for the best answer from what was gathered.
    messages.append({"role": "user", "content": "Stop calling tools now. Give your best final answer "
                     "from what you found so far, and say what remains unchecked."})
    try:
        msg = await asyncio.wait_for(planner.chat_tools(messages, specs, "none"), timeout=45)
        result.answer = (msg.get("content") or "").strip()
        await _record_recipes(msg)
    except Exception as e:
        result.answer = f"Sub-agent stopped ({result.stopped_reason}) and could not summarize: {e}"
    return result
