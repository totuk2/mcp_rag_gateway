"""Starlette route handlers for the Tool-RAG API.

Endpoints (spec section 8):
  POST /tool-rag/retrieve   —  semantic tool retrieval
  POST /tool-rag/reindex    —  trigger index rebuild
  GET  /tool-rag/health     —  index + DB health
  GET  /tool-rag/metrics    —  index / sync metrics
  GET  /tool-rag/catalog    —  catalog overview / ?category= / ?query= coverage
  POST /tool-rag/catalog/refresh — force catalog recompute (admin)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from gateway.context import current_policy
from gateway.merge import split_merged_name
from gateway.tool_db import ToolDb
from gateway.tool_record import ToolType
from tool_rag.embedder import Embedder, create_embedder
from tool_rag.indexer import ToolRagIndexer
from tool_rag.ranker import Ranker
from tool_rag.retriever import Retriever

if TYPE_CHECKING:
    from gateway.health import ServerHealth
    from gateway.registry import Registry
    from tool_rag.catalog import ToolCatalog
    from tool_rag.planner import Planner

logger = logging.getLogger(__name__)


class ToolRagRouter:
    """Starlette-compatible route handler collection for Tool-RAG."""

    def __init__(
        self,
        tool_db: ToolDb,
        embedder: Embedder | None = None,
        indexer: ToolRagIndexer | None = None,
        retriever: Retriever | None = None,
        server_health: "ServerHealth | None" = None,
        catalog: "ToolCatalog | None" = None,
        registry: "Registry | None" = None,
        planner: "Planner | None" = None,
    ):
        self._tool_db = tool_db
        self._catalog = catalog
        self._registry = registry
        self._planner = planner
        self._embedder = embedder or create_embedder()
        self._indexer = indexer or ToolRagIndexer(self._embedder, self._tool_db)
        self._retriever = retriever or Retriever(
            self._embedder, self._indexer, self._tool_db, server_health=server_health
        )
        self._started_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # POST /tool-rag/retrieve
    # ------------------------------------------------------------------

    async def retrieve(self, request: Request) -> JSONResponse:
        body = await request.json()
        query = body.get("query", "")
        if not query:
            return JSONResponse({"detail": "query is required"}, status_code=400)

        top_k = int(body.get("top_k", 5))
        body_servers = body.get("allowed_servers")
        include_schema = body.get("include_schema", True)
        if not isinstance(include_schema, bool):
            include_schema = True
        permission_scope = body.get("permission_scope")
        raw_type = body.get("tool_type")
        tool_type: ToolType | None = raw_type if raw_type in ("read", "write", "admin", "action", "query") else None

        # Policy scoping: when auth is on, the caller's AccessPolicy is in context.
        # Restrict to the key's granted servers (narrowing any body-supplied
        # allowed_servers, never broadening) and apply tool_prefixes — mirroring
        # the in-band find_tools meta-tool. When TOOL_RAG_WITHOUT_AUTH=1 there is
        # no policy in context by design, so enforcement is intentionally skipped.
        policy = current_policy.get()
        if policy is not None:
            granted = set(policy.servers.keys())
            allowed_servers = (
                [s for s in body_servers if s in granted] if body_servers else list(granted)
            )
            if not allowed_servers:
                # No servers granted (or body narrowed everything away) -> nothing.
                return JSONResponse({"query": query, "results": [], "fallback_used": False})
            # Over-fetch so the post-retrieval tool_prefixes filter doesn't undercount.
            over = any(r.tool_prefixes for r in policy.servers.values())
        else:
            allowed_servers = body_servers
            over = False

        fetch_k = top_k * 3 if over else top_k
        result = await self._retriever.retrieve(
            query=query,
            top_k=fetch_k,
            allowed_servers=allowed_servers,
            permission_scope=permission_scope,
            tool_type=tool_type,
        )

        results = []
        for r in result.results:
            if policy is not None:
                try:
                    sid, orig = split_merged_name(r.tool_id)
                except ValueError:
                    continue
                if not policy.tool_visible(sid, orig):
                    continue  # honour per-key tool_prefixes allowlists
            entry = {
                "tool_id": r.tool_id,
                "tool_name": r.tool_name,
                "server_name": r.server_name,
                "score": r.score,
                "reason": r.reason,
                "description": r.description,
                "status": r.status,
                "tool_type": r.tool_type,
            }
            if include_schema:
                entry["input_schema"] = r.input_schema
            results.append(entry)
            if len(results) >= top_k:
                break

        return JSONResponse({
            "query": result.query,
            "results": results,
            "fallback_used": result.fallback_used,
            "include_schema": include_schema,
        })

    # ------------------------------------------------------------------
    # GET /tool-rag/tool/{tool_id}  —  on-demand schema fetch (lazy two-phase)
    # ------------------------------------------------------------------

    async def describe(self, request: Request) -> JSONResponse:
        tool_id = request.path_params.get("tool_id", "")
        if not tool_id:
            return JSONResponse({"detail": "tool_id is required"}, status_code=400)
        # Policy-scope: when auth is on, the tool's server must be granted + visible.
        policy = current_policy.get()
        if policy is not None:
            try:
                sid, orig = split_merged_name(tool_id)
            except ValueError:
                return JSONResponse({"detail": "invalid tool_id"}, status_code=400)
            if not (policy.allows_server(sid) and policy.tool_visible(sid, orig)):
                return JSONResponse({"detail": "access denied for this tool"}, status_code=403)
        rec = self._tool_db.get_tool(tool_id)
        if rec is None:
            return JSONResponse({"detail": f"unknown tool {tool_id}"}, status_code=404)
        return JSONResponse({
            "tool_id": rec.tool_id,
            "tool_name": rec.tool_name,
            "server_name": rec.server_name,
            "description": rec.description,
            "tool_type": rec.tool_type,
            "input_schema": rec.input_schema,
            "status": rec.status,
        })

    # ------------------------------------------------------------------
    # POST /tool-rag/reindex
    # ------------------------------------------------------------------

    async def reindex(self, request: Request) -> JSONResponse:
        body = await request.json()
        mode = body.get("mode", "incremental")

        if mode == "full":
            count = self._indexer.full_reindex()
        elif mode == "incremental":
            count = self._indexer.incremental_reindex()
        else:
            return JSONResponse({"detail": "mode must be 'full' or 'incremental'"}, status_code=400)

        return JSONResponse({
            "mode": mode,
            "tools_indexed": count,
            "index_size": self._indexer.size,
            "db_size": self._tool_db.count_tools(),
        })

    # ------------------------------------------------------------------
    # GET /tool-rag/catalog  —  same views as the browse_tools meta-tool
    # ------------------------------------------------------------------

    async def catalog(self, request: Request) -> JSONResponse:
        if self._catalog is None:
            return JSONResponse({"detail": "catalog not enabled"}, status_code=404)
        # Policy-scoped like retrieve; None under TOOL_RAG_WITHOUT_AUTH by design.
        policy = current_policy.get()
        category = request.query_params.get("category")
        query = request.query_params.get("query")
        if category and query:
            return JSONResponse({"detail": "pass either category or query, not both"}, status_code=400)
        if query:
            return JSONResponse(await self._catalog.coverage(policy, query))
        if category:
            return JSONResponse(self._catalog.category(policy, category))
        return JSONResponse(self._catalog.overview(policy))

    # ------------------------------------------------------------------
    # POST /tool-rag/catalog/refresh  —  force recompute (admin; LLM cost)
    # ------------------------------------------------------------------

    async def catalog_refresh(self, _request: Request) -> JSONResponse:
        policy = current_policy.get()
        if policy is None or not policy.admin:
            return JSONResponse({"detail": "admin key required"}, status_code=403)
        if self._registry is None:
            return JSONResponse({"detail": "catalog not enabled"}, status_code=404)
        from tool_rag.enrichment import refresh_catalog

        res = await refresh_catalog(self._registry, self._tool_db, self._planner, force=True)
        reindexed = self._indexer.incremental_reindex() if res.tags_changed else 0
        return JSONResponse({**res.__dict__, "reindexed": reindexed})

    # ------------------------------------------------------------------
    # Tool reviews (quarantine on review_changes servers) — admin only
    # ------------------------------------------------------------------

    @staticmethod
    def _admin() -> bool:
        policy = current_policy.get()
        return policy is not None and policy.admin

    async def reviews(self, request: Request) -> JSONResponse:
        if not self._admin():
            return JSONResponse({"detail": "admin key required"}, status_code=403)
        state = request.query_params.get("state", "pending")
        out = []
        for r in self._tool_db.list_reviews(None if state == "all" else state):
            cur = self._tool_db.get_tool(r["tool_id"])
            out.append({
                "tool_id": r["tool_id"], "server_id": r["server_id"], "kind": r["kind"], "state": r["state"],
                "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                "approved": None if cur is None or r["kind"] == "new" else
                            {"description": cur.description, "input_schema": cur.input_schema},
                "proposed": r["pending"],
            })
        return JSONResponse({"count": len(out), "reviews": out})

    async def reviews_decide(self, request: Request) -> JSONResponse:
        """POST /tool-rag/reviews/approve|reject  {"tool_ids": [...]} or {"all": true}."""
        if not self._admin():
            return JSONResponse({"detail": "admin key required"}, status_code=403)
        action = request.path_params["action"]
        body = await request.json() if (await request.body()) else {}
        pending = {r["tool_id"]: r for r in self._tool_db.list_reviews("pending")}
        ids = list(pending) if body.get("all") else [t for t in body.get("tool_ids") or [] if t in pending]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")
        done = []
        for tid in ids:
            if action == "approve":
                rec = self._tool_db.get_tool(tid)
                p = pending[tid]["pending"]
                if rec is not None:
                    from dataclasses import replace
                    self._tool_db.upsert_tool(replace(
                        rec, description=p.get("description", rec.description),
                        input_schema=p.get("input_schema", rec.input_schema),
                        status="active", last_seen_at=now))
                self._tool_db.delete_review(tid)
            elif action == "reject":
                self._tool_db.set_review_state(tid, "rejected")
            else:
                return JSONResponse({"detail": "action must be approve or reject"}, status_code=400)
            done.append(tid)
        reindexed = self._indexer.incremental_reindex() if done and action == "approve" else 0
        return JSONResponse({"action": action, "tool_ids": done, "reindexed": reindexed})

    # ------------------------------------------------------------------
    # GET /tool-rag/health
    # ------------------------------------------------------------------

    async def health(self, _request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "index_size": self._indexer.size,
            "db_size": self._tool_db.count_tools(),
            "started_at": self._started_at,
        })

    # ------------------------------------------------------------------
    # GET /tool-rag/metrics
    # ------------------------------------------------------------------

    async def metrics(self, _request: Request) -> JSONResponse:
        return JSONResponse({
            "tools_in_index": self._indexer.size,
            "tools_in_db": self._tool_db.count_tools(),
            "active_servers": self._tool_db.count_servers(),
            "stale_entries": self._tool_db.get_stale_count(),
            "pending_reviews": len(self._tool_db.list_reviews("pending")),
            "started_at": self._started_at,
        })
