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
        allowed_servers = body.get("allowed_servers")
        permission_scope = body.get("permission_scope")
        raw_type = body.get("tool_type")
        tool_type: ToolType | None = raw_type if raw_type in ("read", "write", "admin", "action", "query") else None

        result = await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            allowed_servers=allowed_servers,
            permission_scope=permission_scope,
            tool_type=tool_type,
        )

        return JSONResponse({
            "query": result.query,
            "results": [
                {
                    "tool_id": r.tool_id,
                    "tool_name": r.tool_name,
                    "server_name": r.server_name,
                    "score": r.score,
                    "reason": r.reason,
                    "description": r.description,
                    "input_schema": r.input_schema,
                    "status": r.status,
                    "tool_type": r.tool_type,
                }
                for r in result.results
            ],
            "fallback_used": result.fallback_used,
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
