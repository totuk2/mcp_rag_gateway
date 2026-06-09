"""Starlette route handlers for the Tool-RAG API.

Endpoints (spec section 8):
  POST /tool-rag/retrieve   —  semantic tool retrieval
  POST /tool-rag/reindex    —  trigger index rebuild
  GET  /tool-rag/health     —  index + DB health
  GET  /tool-rag/metrics    —  index / sync metrics
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
    ):
        self._tool_db = tool_db
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
            "started_at": self._started_at,
        })
