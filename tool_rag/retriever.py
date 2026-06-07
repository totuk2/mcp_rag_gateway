"""Top-K retrieval orchestrator for Tool-RAG.

Pipeline:
  1. Embed query
  2. FAISS search for top-K*2 candidates
  3. Apply metadata filters (section 7.7)
  4. Rank via Ranker
  5. Return top-K with scores + reasons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from tool_rag.embedder import Embedder
from tool_rag.indexer import ToolRagIndexer
from tool_rag.ranker import Ranker
from gateway.tool_db import ToolDb
from gateway.tool_record import ToolRecord, ToolStatus, ToolType

if TYPE_CHECKING:
    from gateway.health import ServerHealth

logger = logging.getLogger(__name__)


@dataclass
class ScoredTool:
    """One retrieval result with score and explanation."""

    tool_id: str
    tool_name: str
    server_name: str
    score: float
    reason: str
    description: str
    input_schema: dict[str, Any]
    status: str
    tool_type: str


@dataclass
class RetrievalResult:
    """Full response from a retrieve() call."""

    query: str
    results: list[ScoredTool]
    fallback_used: bool = False


class Retriever:
    """Orchestrates query -> embed -> search -> filter -> rank -> top-K."""

    def __init__(
        self,
        embedder: Embedder,
        indexer: ToolRagIndexer,
        tool_db: ToolDb,
        ranker: Ranker | None = None,
        server_health: "ServerHealth | None" = None,
    ):
        self._embedder = embedder
        self._indexer = indexer
        self._tool_db = tool_db
        self._ranker = ranker or Ranker()
        self._server_health = server_health

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_servers: Sequence[str] | None = None,
        permission_scope: str | None = None,
        tool_type: ToolType | None = None,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline."""
        # 1. Embed
        query_vec = self._embedder.embed(query)

        # 2. FAISS search (fetch more than top_k for filtering headroom)
        candidates = self._indexer.search(query_vec, top_k=top_k * 2)

        if not candidates:
            logger.info("No index candidates for query=%r, falling back to DB scan", query)
            return self._fallback(query, top_k, allowed_servers, permission_scope, tool_type)

        # 3. Fetch full records + filter + rank
        scored: list[tuple[float, ToolRecord]] = []
        for tid, semantic_score in candidates:
            record = self._tool_db.get_tool(tid)
            if record is None:
                continue

            # Filter by mandatory filters (section 7.7)
            if allowed_servers and record.server_id not in allowed_servers:
                continue
            if permission_scope and record.permission_scope != permission_scope:
                continue
            if tool_type and record.tool_type != tool_type:
                continue
            if record.status != "active":
                continue
            if self._server_health and not self._server_health.is_up(record.server_id):
                continue  # upstream currently unreachable — don't surface its tools

            final_score = self._ranker.compute_score(
                query=query,
                record=record,
                semantic_score=semantic_score,
                allowed_servers=allowed_servers,
                permission_scope=permission_scope,
                tool_type=tool_type,
            )
            scored.append((final_score, record))

        # 4. Sort by score desc, tie-break by tool_id
        scored.sort(key=lambda x: (-x[0], x[1].tool_id))
        top = scored[:top_k]

        results = [
            ScoredTool(
                tool_id=record.tool_id,
                tool_name=record.tool_name,
                server_name=record.server_name,
                score=round(score, 4),
                reason=self._ranker.reason(query, record, score),
                description=record.description,
                input_schema=record.input_schema,
                status=record.status,
                tool_type=record.tool_type,
            )
            for score, record in top
        ]

        return RetrievalResult(query=query, results=results)

    def _fallback(
        self,
        query: str,
        top_k: int,
        allowed_servers: Sequence[str] | None,
        permission_scope: str | None,
        tool_type: ToolType | None,
    ) -> RetrievalResult:
        """Fallback: keyword scan of all tools when vector index is empty."""
        tools = self._tool_db.get_all_tools()
        scored: list[tuple[float, ToolRecord]] = []
        for record in tools:
            if allowed_servers and record.server_id not in allowed_servers:
                continue
            if permission_scope and record.permission_scope != permission_scope:
                continue
            if tool_type and record.tool_type != tool_type:
                continue
            if record.status != "active":
                continue
            if self._server_health and not self._server_health.is_up(record.server_id):
                continue
            kw = self._ranker._keyword_score(query, record)  # noqa
            scored.append((kw, record))
        scored.sort(key=lambda x: (-x[0], x[1].tool_id))
        top = scored[:top_k]
        results = [
            ScoredTool(
                tool_id=record.tool_id,
                tool_name=record.tool_name,
                server_name=record.server_name,
                score=round(score, 4),
                reason=f"Fallback keyword match for '{query}'.",
                description=record.description,
                input_schema=record.input_schema,
                status=record.status,
                tool_type=record.tool_type,
            )
            for score, record in top
        ]
        return RetrievalResult(query=query, results=results, fallback_used=True)
