"""Policy-scoped tool catalog views for general questions about available tools.

Backs the `browse_tools` meta-tool and GET /tool-rag/catalog:
  overview()  — "what kinds of tools do I have?": domains (servers) with a
                description + categories, and categories with counts.
  category()  — "do I have a tool in category X?": one category (or domain).
  coverage()  — "do I have tools for problem X?": a strong/weak/none verdict
                (planner LLM over the catalog + search matches, else score
                thresholds) with the covering domains and categories.
  summary_text() — compact overview injected into meta-tool descriptions.

Everything is computed from the ToolDb at call time (cheap at homelab scale)
and only counts tools the key can see: granted servers, `tool_prefixes`,
status=active, and servers currently up. Non-admin keys get summaries only —
never a per-category tool listing — unless TOOL_RAG_CATALOG_LIST_TOOLS=1
(the spec forbids exposing the full catalog to non-admin keys). Categories and
descriptions come from tool_rag/enrichment.py.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from gateway.merge import split_merged_name
from gateway.tool_db import ToolDb
from gateway.tool_record import ToolRecord

if TYPE_CHECKING:
    from gateway.health import ServerHealth
    from gateway.policy import AccessPolicy
    from tool_rag.planner import Planner
    from tool_rag.retriever import Retriever

# Max tools listed by category() when listing is allowed.
CATEGORY_LIST_LIMIT = 25
# coverage(): retrieval depth, matches echoed back.
COVERAGE_TOP_K = 10
COVERAGE_TOP_MATCHES = 3
# Fallback (no planner) final-score thresholds (ranker: semantic + 0.25 keyword
# + 0.15 metadata). The semantic term is a sigmoid'd cross-encoder score when
# reranked, else bi-encoder cosine — different scales, so separate defaults.
# Reranked values from the live catalog: clear hits 0.45-1.0, unrelated <=0.29,
# but oblique/non-English hits as low as 0.17 — hence the LLM judge first.
_THRESHOLDS = {
    True: {"strong": 0.45, "weak": 0.3},
    False: {"strong": 0.85, "weak": 0.7},
}
# summary_text() budget (~200 tokens).
SUMMARY_MAX_CHARS = 900


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _thresholds(reranked: bool) -> dict[str, float]:
    suffix = "" if reranked else "_NORERANK"
    base = _THRESHOLDS[reranked]
    return {
        "strong": _env_float(f"TOOL_RAG_CATALOG_STRONG{suffix}", base["strong"]),
        "weak": _env_float(f"TOOL_RAG_CATALOG_WEAK{suffix}", base["weak"]),
    }


def _teaser(text: str, cap: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= cap else text[: cap - 1].rstrip() + "…"


class ToolCatalog:
    def __init__(self, tool_db: ToolDb, server_health: "ServerHealth | None" = None,
                 retriever: "Retriever | None" = None, planner: "Planner | None" = None) -> None:
        self._tool_db = tool_db
        self._server_health = server_health
        self._retriever = retriever
        self._planner = planner

    # ------------------------------------------------------------------
    # Scoping
    # ------------------------------------------------------------------

    def _visible(self, policy: "AccessPolicy | None", rec: ToolRecord) -> bool:
        if rec.status != "active":
            return False
        if self._server_health is not None and not self._server_health.is_up(rec.server_id):
            return False
        if policy is None:  # TOOL_RAG_WITHOUT_AUTH: no policy in context by design
            return True
        return policy.allows_server(rec.server_id) and policy.tool_visible(rec.server_id, rec.tool_name)

    def _visible_tools(self, policy: "AccessPolicy | None") -> list[ToolRecord]:
        return [t for t in self._tool_db.get_all_tools() if self._visible(policy, t)]

    @staticmethod
    def _may_list_tools(policy: "AccessPolicy | None") -> bool:
        if policy is not None and policy.admin:
            return True
        return os.environ.get("TOOL_RAG_CATALOG_LIST_TOOLS", "").lower() in ("1", "true")

    def _taxonomy(self) -> dict[str, str]:
        tax = self._tool_db.get_current_taxonomy()
        return {c["name"]: c.get("description", "") for c in (tax["categories"] if tax else [])}

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def overview(self, policy: "AccessPolicy | None") -> dict[str, Any]:
        tools = self._visible_tools(policy)
        profiles = self._tool_db.list_server_profiles()
        taxonomy = self._taxonomy()
        by_domain: dict[str, list[ToolRecord]] = defaultdict(list)
        cat_tools: dict[str, int] = defaultdict(int)
        cat_domains: dict[str, set[str]] = defaultdict(set)
        for t in tools:
            by_domain[t.server_id].append(t)
            for c in t.tags:
                cat_tools[c] += 1
                cat_domains[c].add(t.server_id)

        domains = []
        for sid in sorted(by_domain, key=lambda s: (-len(by_domain[s]), s)):
            counts: dict[str, int] = defaultdict(int)
            for t in by_domain[sid]:
                for c in t.tags:
                    counts[c] += 1
            domains.append({
                "domain": sid,
                "description": profiles.get(sid, {}).get("description", ""),
                "categories": sorted(counts, key=lambda c: (-counts[c], c)),
                "tool_count": len(by_domain[sid]),
            })
        categories = [
            {
                "name": c,
                "description": taxonomy.get(c, ""),
                "tool_count": cat_tools[c],
                "domains": sorted(cat_domains[c]),
            }
            for c in sorted(cat_tools, key=lambda c: (-cat_tools[c], c))
        ]
        return {"total_tools": len(tools), "domains": domains, "categories": categories}

    def category(self, policy: "AccessPolicy | None", name: str) -> dict[str, Any]:
        """One category — or one domain, since agents often ask by server name."""
        key = (name or "").strip().lower()
        ov = self.overview(policy)
        hit = next((c for c in ov["categories"] if c["name"] == key), None)
        dom = next((d for d in ov["domains"] if d["domain"].lower() == key), None)
        if hit is None and dom is None:
            close = [c["name"] for c in ov["categories"] if key and (key in c["name"] or c["name"] in key)]
            return {
                "found": False,
                "category": name,
                "similar": close,
                "available_categories": [c["name"] for c in ov["categories"]],
                "hint": "No such category or domain. Call browse_tools with `query` "
                        "to check whether any tool covers it.",
            }
        out: dict[str, Any] = {"found": True}
        if hit is not None:
            out.update({"kind": "category", **hit})
            members = [t for t in self._visible_tools(policy) if key in t.tags]
        else:
            out.update({"kind": "domain", **dom})
            members = [t for t in self._visible_tools(policy) if t.server_id == dom["domain"]]
        if self._may_list_tools(policy):
            members.sort(key=lambda t: t.tool_id)
            out["tools"] = [
                {"call_name": t.tool_id, "description": _teaser(t.description, 160)}
                for t in members[:CATEGORY_LIST_LIMIT]
            ]
            if len(members) > CATEGORY_LIST_LIMIT:
                out["tools_truncated"] = len(members) - CATEGORY_LIST_LIMIT
        out["hint"] = ("Call find_tools with a query describing your task to get specific tools "
                       "(call_name + input_schema).")
        return out

    async def coverage(self, policy: "AccessPolicy | None", query: str) -> dict[str, Any]:
        """Do the visible tools cover `query`? Aggregated, not a tool listing.

        With a planner configured the verdict is an LLM judgement over the domain
        catalog + top search matches: measured on the live catalog, absolute
        search scores alone separate poorly (oblique or non-English questions about
        covered capabilities score like unrelated ones). Without a planner (or if
        the call fails) the verdict falls back to score thresholds."""
        assert self._retriever is not None
        allowed: list[str] | None = None
        over = False
        if policy is not None:
            allowed = list(policy.servers.keys())
            if not allowed:
                return {"query": query, "verdict": "none", "reason": "No servers are granted to this API key."}
            over = any(r.tool_prefixes for r in policy.servers.values())
        # Wait out a reranker cold start: comparable scores for the threshold
        # fallback and better-ordered matches for the judge.
        result = await self._retriever.retrieve(
            query=query, top_k=COVERAGE_TOP_K * 3 if over else COVERAGE_TOP_K, allowed_servers=allowed,
            wait_for_rerank=True,
        )
        tags_by_id = {t.tool_id: t.tags for t in self._visible_tools(policy)}
        candidates: list[dict[str, Any]] = []
        for r in result.results:
            if r.tool_id not in tags_by_id:  # not visible to this key (prefixes, down, ...)
                continue
            candidates.append({"call_name": r.tool_id, "domain": split_merged_name(r.tool_id)[0],
                               "categories": list(tags_by_id[r.tool_id]), "score": r.score,
                               "description": _teaser(r.description, 120)})
            if len(candidates) >= COVERAGE_TOP_K:
                break

        out: dict[str, Any] = {"query": query, "best_score": round(max((c["score"] for c in candidates), default=0.0), 4),
                               "reranked": result.reranked}
        judged = await self._judge(policy, query, candidates)
        if judged is not None:
            verdict, domains, categories = judged["verdict"], judged["domains"], judged["categories"]
            matches = [c for c in candidates if c["domain"] in domains] if verdict != "none" else []
            out.update({"verdict": verdict, "verdict_source": "llm", "reason": judged["reason"]})
            if judged["suggested_query"] and verdict != "none":
                out["suggested_query"] = judged["suggested_query"]
        else:
            th = _thresholds(result.reranked)
            matches = [c for c in candidates if c["score"] >= th["weak"]]
            best = max((m["score"] for m in matches), default=0.0)
            verdict = "strong" if best >= th["strong"] else ("weak" if matches else "none")
            domains = list(dict.fromkeys(m["domain"] for m in matches))
            categories = list(dict.fromkeys(c for m in matches for c in m["categories"]))
            out.update({"verdict": verdict, "verdict_source": "scores"})
        search = out.get("suggested_query") or query
        out.update({
            "domains": domains,
            "categories": categories,
            "top_matches": [{k: m[k] for k in ("call_name", "domain", "score")} for m in matches[:COVERAGE_TOP_MATCHES]],
            "hint": {
                "strong": f"Relevant tools exist: call find_tools with query {search!r} to get their input_schema.",
                "weak": f"Only partial/indirect coverage: call find_tools with query {search!r} to inspect.",
                "none": "No tool available to this key appears to cover this.",
            }[verdict],
        })
        return out

    async def _judge(self, policy: "AccessPolicy | None", query: str,
                     candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Planner LLM verdict, validated against what this key can see."""
        if self._planner is None:
            return None
        ov = self.overview(policy)
        catalog = {
            "domains": [{k: d[k] for k in ("domain", "description", "categories")} for d in ov["domains"]],
            "categories": [{k: c[k] for k in ("name", "description")} for c in ov["categories"]],
        }
        matches = [{k: c[k] for k in ("call_name", "description", "score")} for c in candidates[:8]]
        raw = await self._planner.judge_coverage(query, catalog, matches)
        if raw is None:
            return None
        known_domains = {d["domain"] for d in ov["domains"]}
        known_cats = {c["name"] for c in ov["categories"]}
        domains = [d for d in raw.get("domains") or [] if d in known_domains]
        verdict = raw["verdict"]
        if verdict != "none" and not domains:
            verdict = "weak"  # claims coverage but names nothing this key has
        return {
            "verdict": verdict,
            "domains": domains,
            "categories": [c for c in raw.get("categories") or [] if c in known_cats],
            "reason": str(raw.get("reason") or "")[:300],
            "suggested_query": str(raw.get("suggested_query") or "")[:200],
        }

    def summary_text(self, policy: "AccessPolicy | None", max_chars: int = SUMMARY_MAX_CHARS) -> str:
        """Compact per-key overview for meta-tool descriptions (~200 tokens)."""
        ov = self.overview(policy)
        if not ov["domains"]:
            return "No tools are currently available to this key."
        head = f"Your catalog: {ov['total_tools']} tools in {len(ov['domains'])} domains. "
        cats = ", ".join(f"{c['name']} ({c['tool_count']})" for c in ov["categories"])
        cats_part = f" Categories: {cats}." if cats else ""
        for desc_cap in (70, 35, 0):  # shrink descriptions before dropping domains
            items = [
                f"{d['domain']} ({d['tool_count']})" + (f" — {_teaser(d['description'], desc_cap)}"
                                                       if desc_cap and d["description"] else "")
                for d in ov["domains"]
            ]
            text = head + "Domains: " + "; ".join(items) + "." + cats_part
            if len(text) <= max_chars:
                return text
        text = head + "Domains: " + ", ".join(f"{d['domain']} ({d['tool_count']})" for d in ov["domains"]) + "."
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def domain_names(self, policy: "AccessPolicy | None") -> list[str]:
        return sorted({t.server_id for t in self._visible_tools(policy)})
