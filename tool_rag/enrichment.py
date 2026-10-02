"""Dynamic tool catalog: server profiles, category taxonomy, per-tool tags.

Backs the `browse_tools` meta-tool (tool_rag/catalog.py), which lets an agent
ask general questions ("what kinds of tools do I have?", "do I have anything
for X?") instead of only specific find_tools queries.

Nothing here is hand-maintained except optional manifest metadata:
  - the taxonomy (categories + descriptions) is derived by the planner LLM from
    the *current* tool set and re-derived whenever a server or tool is added,
    removed, or changes its description;
  - every tool gets 1-3 tags from that taxonomy;
  - every server gets a one-sentence description.

Precedence: manifest `description`/`categories` (registry) win; the LLM fills
the rest; the upstream's own `initialize` serverInfo/instructions are the
fallback. Without an LLM (planner off / failing) the catalog degrades to
manifest categories + domains (servers) only.

Change detection is by fingerprint: the catalog fingerprint hashes every
(tool_id, name+description hash) plus manifest metadata. Unchanged → no-op
without LLM calls, so a plain restart costs nothing. Changed → re-derive the
taxonomy (previous names are fed back so they stay stable) and re-tag only the
tools that need it (all of them only when the set of category names changed).
Results are computed first and written at the end, so a timeout or failure
leaves the previous catalog intact.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from gateway.registry import Registry, ServerConfig
from gateway.tool_db import ToolDb
from gateway.tool_record import ToolRecord

if TYPE_CHECKING:
    from tool_rag.planner import Planner

logger = logging.getLogger(__name__)

# Category names: kebab-case (also enforced on manifest `categories` by provision.py).
CATEGORY_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_CATEGORIES = 20
MAX_TAGS_PER_TOOL = 3
# Concurrent per-server tagging calls.
_TAG_CONCURRENCY = 3
# Whole refresh (all LLM calls) budget; on timeout the previous catalog stays.
DEFAULT_REFRESH_TIMEOUT = 90.0
# Per-tool description chars sent to the LLM (names carry most of the signal).
_LLM_DESC_CHARS = 200

_refresh_lock = asyncio.Lock()


@dataclass
class RefreshResult:
    changed: bool
    reason: str
    taxonomy_version: int | None = None
    categories: int = 0
    tags_changed: int = 0
    llm_calls: int = 0
    # Some LLM call failed (or the refresh timed out): retry later.
    incomplete: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")


def _hash(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _server_parts(cfg: ServerConfig, tools: list[ToolRecord]) -> list[str]:
    parts = [f"S{cfg.server_id}\0{cfg.description or ''}\0{','.join(cfg.categories)}"]
    parts += [f"T{t.tool_id}\0{t.content_fingerprint}" for t in sorted(tools, key=lambda t: t.tool_id)]
    return parts


def catalog_fingerprint(registry: Registry, by_server: dict[str, list[ToolRecord]]) -> str:
    parts: list[str] = []
    for sid in sorted(by_server):
        parts += _server_parts(registry.servers[sid], by_server[sid])
    return _hash(parts)


def _server_fingerprint(cfg: ServerConfig, tools: list[ToolRecord]) -> str:
    return _hash(_server_parts(cfg, tools))


def _first_sentence(text: str, cap: int = 200) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    out = m.group(1) if m else text
    return out if len(out) <= cap else out[: cap - 1].rstrip() + "…"


def _upstream_hint(profile: dict[str, Any]) -> str:
    title = (profile.get("upstream_title") or "").strip()
    instr = _first_sentence(profile.get("upstream_instructions") or "", 300)
    return " — ".join(p for p in (title, instr) if p)


def _fallback_description(sid: str, tools: list[ToolRecord], profile: dict[str, Any]) -> tuple[str, str]:
    instr = _first_sentence(profile.get("upstream_instructions") or "")
    if instr:
        return instr, "upstream"
    names = ", ".join(t.tool_name for t in tools[:6])
    more = f", … (+{len(tools) - 6})" if len(tools) > 6 else ""
    return f"{len(tools)} tools: {names}{more}", "fallback"


def _validate_taxonomy(raw: Any) -> list[dict[str, str]]:
    """Keep well-formed, unique kebab-case categories (max MAX_CATEGORIES)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(raw, list):
        return out
    for c in raw:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip().lower()
        if not CATEGORY_NAME.match(name) or name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "description": _first_sentence(str(c.get("description") or ""), 160)})
        if len(out) >= MAX_CATEGORIES:
            break
    return out


def _validate_tags(raw: Any, allowed: set[str]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    tags: list[str] = []
    for t in raw:
        name = str(t).strip().lower()
        if name in allowed and name not in tags:
            tags.append(name)
    return tuple(tags[:MAX_TAGS_PER_TOOL])


def _tools_payload(tools: list[ToolRecord]) -> list[dict[str, str]]:
    return [{"name": t.tool_name, "description": _first_sentence(t.description, _LLM_DESC_CHARS)} for t in tools]


async def refresh_catalog(
    registry: Registry,
    tool_db: ToolDb,
    planner: "Planner | None",
    force: bool = False,
    timeout: float | None = None,
) -> RefreshResult:
    """Recompute the catalog if the tool set changed (or `force`). Call after
    every sync and before reindexing (tags are part of the embedding text)."""
    if timeout is None:
        try:
            timeout = float(os.environ.get("TOOL_RAG_CATALOG_REFRESH_TIMEOUT", DEFAULT_REFRESH_TIMEOUT))
        except ValueError:
            timeout = DEFAULT_REFRESH_TIMEOUT
    async with _refresh_lock:
        by_server: dict[str, list[ToolRecord]] = defaultdict(list)
        for t in tool_db.get_all_tools():
            if t.server_id in registry.servers:
                by_server[t.server_id].append(t)
        by_server = dict(by_server)
        fp = catalog_fingerprint(registry, by_server)
        current = tool_db.get_current_taxonomy()
        tag_cache = tool_db.get_all_tool_tags()
        profiles = tool_db.list_server_profiles()

        up_to_date = (
            current is not None
            and current["catalog_fingerprint"] == fp
            and all(
                (c := tag_cache.get(t.tool_id)) is not None and c["taxonomy_version"] == current["version"]
                for tools in by_server.values() for t in tools
            )
            and all(profiles.get(sid, {}).get("description") for sid in by_server)
        )
        if up_to_date and not force:
            logger.info("catalog unchanged (fingerprint %s, taxonomy v%s)", fp, current["version"])
            return RefreshResult(changed=False, reason="unchanged", taxonomy_version=current["version"],
                                 categories=len(current["categories"]))
        try:
            computed = await asyncio.wait_for(
                _compute(registry, by_server, current, tag_cache, profiles, planner, fp, force), timeout
            )
        except asyncio.TimeoutError:
            logger.warning("catalog refresh timed out after %.0fs; keeping the previous catalog", timeout)
            return RefreshResult(changed=False, reason="timeout",
                                 taxonomy_version=current["version"] if current else None, incomplete=True)
        return _apply(tool_db, registry, by_server, current, tag_cache, profiles, fp, computed)


async def _compute(
    registry: Registry,
    by_server: dict[str, list[ToolRecord]],
    current: dict[str, Any] | None,
    tag_cache: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    planner: "Planner | None",
    fp: str,
    force: bool,
) -> dict[str, Any]:
    llm_calls = 0
    # Set when an LLM call that should have happened failed: the result is still
    # applied (best-effort), but without its fingerprint, so the next refresh
    # (restart / resync / POST /tool-rag/catalog/refresh) retries.
    incomplete = False
    seeds = sorted({c for sid in by_server for c in registry.servers[sid].categories})
    prev_cats: list[dict[str, str]] = current["categories"] if current else []

    # 1. Taxonomy — one global call, only when the tool set changed.
    categories: list[dict[str, str]] = []
    if planner is not None and (force or current is None or current["catalog_fingerprint"] != fp):
        domains = []
        for sid, tools in sorted(by_server.items()):
            cfg = registry.servers[sid]
            domains.append({
                "server": sid,
                "about": cfg.description or _upstream_hint(profiles.get(sid, {})),
                "tools": _tools_payload(tools),
            })
        llm_calls += 1
        categories = _validate_taxonomy(await planner.derive_taxonomy(domains, prev_cats, seeds))
        incomplete = not categories
    if not categories:
        categories = [dict(c) for c in prev_cats]  # LLM off/failed: keep what we had
    names = {c["name"] for c in categories}
    # Stability: the LLM may add categories but never drop one that unchanged
    # tools still carry (it sometimes does, e.g. after an unrelated removal);
    # categories only disappear when no current tool is tagged with them.
    cur_ver = current["version"] if current else None
    still_used = {
        x
        for tools in by_server.values() for t in tools
        if (c := tag_cache.get(t.tool_id)) is not None
        and c["taxonomy_version"] == cur_ver and c["fingerprint"] == t.content_fingerprint
        for x in c["tags"]
    }
    for c in prev_cats:
        if c["name"] in still_used and c["name"] not in names:
            categories.append(dict(c))
            names.add(c["name"])
    for s in seeds:  # manifest categories are always present
        if s not in names:
            owners = ", ".join(sid for sid in sorted(by_server) if s in registry.servers[sid].categories)
            categories.append({"name": s, "description": f"Tools from {owners}."})
            names.add(s)
    # A new category may fit existing tools -> re-tag everything. Removals alone
    # don't invalidate the remaining tags (they're filtered below); only tools
    # left with no tag at all get re-tagged.
    retag_all = current is None or bool(names - {c["name"] for c in prev_cats})

    # 2. Per-server description + tags — re-tag only what needs it.
    sem = asyncio.Semaphore(_TAG_CONCURRENCY)
    calls = 0

    async def one(sid: str, tools: list[ToolRecord]) -> tuple[str, dict[str, tuple[str, ...]], tuple[str, str, str], bool]:
        nonlocal calls
        cfg = registry.servers[sid]
        prof = profiles.get(sid, {})
        sfp = _server_fingerprint(cfg, tools)

        def cached_tags(t: ToolRecord) -> tuple[str, ...]:
            """Still-valid cached tags (same content, current taxonomy), filtered
            to surviving category names; () when the tool must be re-tagged."""
            c = tag_cache.get(t.tool_id)
            if c is None or c["taxonomy_version"] != cur_ver or c["fingerprint"] != t.content_fingerprint:
                return ()
            return tuple(x for x in c["tags"] if x in names)

        tags: dict[str, tuple[str, ...]] = {}
        need: list[ToolRecord] = []
        for t in tools:
            kept = () if (force or retag_all) else cached_tags(t)
            if kept:
                tags[t.tool_id] = kept
            else:
                need.append(t)
        need_desc = not cfg.description and (force or prof.get("fingerprint") != sfp or prof.get("source") != "llm")

        out = None
        failed = False
        if planner is not None and names and (need or need_desc):
            hint = cfg.description or _upstream_hint(prof)
            async with sem:
                calls += 1
                out = await planner.assign_tags(sid, hint, _tools_payload(tools), categories)
            failed = out is None
        llm_tags = out.get("tags") if out else None
        forced = tuple(c for c in cfg.categories if c in names)
        for t in need:
            got = _validate_tags(llm_tags.get(t.tool_name), names) if isinstance(llm_tags, dict) else ()
            if not got and t.tool_id in tag_cache:
                got = tuple(x for x in tag_cache[t.tool_id]["tags"] if x in names)  # keep old on LLM miss
            tags[t.tool_id] = got or forced[:MAX_TAGS_PER_TOOL]
        if forced:  # manifest categories always apply to this server's tools
            for tid, tg in tags.items():
                if not set(tg) & set(forced):
                    tags[tid] = (forced[0],) + tuple(x for x in tg if x != forced[0])[: MAX_TAGS_PER_TOOL - 1]

        # Description only changes when the server did (stable across re-tagging).
        if cfg.description:
            desc = (cfg.description, "manifest", sfp)
        elif not need_desc:
            desc = (prof.get("description", ""), prof.get("source", ""), prof.get("fingerprint", ""))
        elif out and isinstance(out.get("description"), str) and out["description"].strip():
            desc = (_first_sentence(out["description"], 200), "llm", sfp)
        else:
            d, src = _fallback_description(sid, tools, prof)
            desc = (d, src, sfp)
        return sid, tags, desc, failed

    settled = await asyncio.gather(*(one(sid, tools) for sid, tools in sorted(by_server.items())))
    llm_calls += calls
    all_tags: dict[str, tuple[str, ...]] = {}
    descriptions: dict[str, tuple[str, str, str]] = {}
    for sid, tags, desc, failed in settled:
        all_tags.update(tags)
        descriptions[sid] = desc
        incomplete = incomplete or failed

    # 3. Drop categories no tool carries anymore (e.g. their only server left).
    used = {x for tg in all_tags.values() for x in tg}
    categories = [c for c in categories if c["name"] in used]
    return {"categories": categories, "tags": all_tags, "descriptions": descriptions,
            "llm_calls": llm_calls, "incomplete": incomplete}


def _apply(
    tool_db: ToolDb,
    registry: Registry,
    by_server: dict[str, list[ToolRecord]],
    current: dict[str, Any] | None,
    tag_cache: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    fp: str,
    computed: dict[str, Any],
) -> RefreshResult:
    categories = sorted(computed["categories"], key=lambda c: c["name"])
    names = {c["name"] for c in categories}
    if computed["incomplete"]:
        fp = ""  # never matches -> the next refresh retries the failed LLM calls
        logger.warning("catalog: some LLM calls failed; partial result applied, will retry on next refresh")
    if current is not None and names == {c["name"] for c in current["categories"]}:
        version = current["version"]
        tool_db.update_taxonomy(version, fp, categories)
    else:
        version = tool_db.save_taxonomy(fp, categories)

    tags: dict[str, tuple[str, ...]] = computed["tags"]
    all_tools = [t for tools in by_server.values() for t in tools]
    tool_db.save_tool_tags({t.tool_id: (t.content_fingerprint, version, tags.get(t.tool_id, ())) for t in all_tools})
    live_ids = {t.tool_id for t in all_tools}
    gone = [tid for tid in tag_cache if tid not in live_ids]
    if gone:
        tool_db.delete_tool_tags(gone)

    # Mirror tags onto the tool rows (embedding text). A fresh last_seen_at makes
    # `incremental` reindexing pick the change up too.
    now = _now()
    changed = 0
    for t in all_tools:
        new = tags.get(t.tool_id, ())
        if tuple(t.tags) != new:
            tool_db.upsert_tool(replace(t, tags=new, last_seen_at=now))
            changed += 1

    for sid, (desc, source, sfp) in computed["descriptions"].items():
        tool_db.save_server_profile(sid, desc, source, sfp)
    for sid in profiles:
        if sid not in registry.servers:
            tool_db.delete_server_profile(sid)

    logger.info(
        "catalog refreshed: taxonomy v%s (%d categories: %s), %d tool tag changes, %d LLM calls",
        version, len(categories), ", ".join(sorted(names)), changed, computed["llm_calls"],
    )
    return RefreshResult(changed=True, reason="refreshed", taxonomy_version=version,
                         categories=len(categories), tags_changed=changed, llm_calls=computed["llm_calls"],
                         incomplete=computed["incomplete"])
