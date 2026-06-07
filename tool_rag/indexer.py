"""FAISS-based vector index for tool embeddings.

Backed by ``IndexIDMap(IndexFlatIP)`` so individual tool vectors can be removed
(plain ``IndexFlatIP`` has no removal, which previously caused the index file to
accumulate one orphaned vector per tool per re-index).

Each tool gets a stable int64 id (``tool_id`` string -> int), persisted alongside
the index. Re-indexing a tool is remove-then-add of the same id, so the on-disk
size stays bounded.

Methods:
  full_reindex()        — rebuild index from all tools in ToolDb.
  incremental_reindex() — only (re-)embed tools whose last_indexed_at < last_seen_at.
  index_tool()          — embed + add/update a single tool.
  remove_tool()         — delete a tool from the index.
  search()              — query embedding -> top-K candidates (unfiltered IDs + scores).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tool_rag.embedder import Embedder
from gateway.tool_db import ToolDb
from gateway.tool_record import ToolRecord

logger = logging.getLogger(__name__)

# Bump when the on-disk meta/index layout changes. An older/missing version
# forces a clean rebuild instead of loading an incompatible index.
META_VERSION = 2


def _composite_text(record: ToolRecord) -> str:
    """Build the composite embedding text per spec section 7.5."""
    inp = list(record.input_schema.get("properties", {}).keys()) if record.input_schema else []
    tags = ", ".join(record.tags)
    return (
        f"Tool name: {record.tool_name}\n"
        f"Description: {record.description}\n"
        f"Type: {record.tool_type}\n"
        f"Server: {record.server_id}\n"
        f"Input: {', '.join(inp)}\n"
        f"Tags: {tags}"
    )


class ToolRagIndexer:
    """Manages a FAISS index mapping tool embeddings <-> tool_ids."""

    def __init__(
        self,
        embedder: Embedder,
        tool_db: ToolDb,
        index_path: str | Path = "tool_rag.index",
        meta_path: str | Path = "tool_rag.meta",
    ):
        self._embedder = embedder
        self._tool_db = tool_db
        self._index_path = Path(index_path)
        self._meta_path = Path(meta_path)
        import faiss
        self._faiss = faiss
        self._index = None
        # Stable bidirectional map: tool_id (str) <-> faiss int64 id.
        self._tool_to_id: dict[str, int] = {}
        self._id_to_tool: dict[int, str] = {}
        self._next_id: int = 0
        self._dim = embedder.dimension
        self._load_or_create()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_create(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            try:
                self._load()
                logger.info("Loaded index (%d vectors, dim=%d)", self._index.ntotal, self._dim)
                return
            except Exception as exc:
                # Stale format (old positional index), corruption, or version
                # bump -> discard and rebuild. Never read a half-loaded index.
                logger.warning("Failed to load index (%s), rebuilding from scratch", exc)
        self._create_empty()

    def _create_empty(self) -> None:
        self._index = self._faiss.IndexIDMap(self._faiss.IndexFlatIP(self._dim))
        self._tool_to_id = {}
        self._id_to_tool = {}
        self._next_id = 0

    def _save(self) -> None:
        self._faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps({
                "version": META_VERSION,
                "tool_to_id": self._tool_to_id,
                "next_id": self._next_id,
                "dim": self._dim,
            })
        )

    def _load(self) -> None:
        # Validate meta (incl. version) BEFORE touching the index file, so an
        # incompatible on-disk index is never loaded into this object.
        meta = json.loads(self._meta_path.read_text())
        if meta.get("version") != META_VERSION:
            raise ValueError(f"index meta version {meta.get('version')} != {META_VERSION}")
        self._tool_to_id = {str(k): int(v) for k, v in meta["tool_to_id"].items()}
        self._id_to_tool = {v: k for k, v in self._tool_to_id.items()}
        self._next_id = int(meta["next_id"])
        self._dim = int(meta["dim"])
        self._index = self._faiss.read_index(str(self._index_path))

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------

    def _id_for(self, tool_id: str) -> int:
        """Return the stable int id for a tool, allocating one if new."""
        iid = self._tool_to_id.get(tool_id)
        if iid is None:
            iid = self._next_id
            self._next_id += 1
            self._tool_to_id[tool_id] = iid
            self._id_to_tool[iid] = tool_id
        return iid

    def index_tool(self, record: ToolRecord) -> None:
        """Embed and add/update one tool in the index (leak-free)."""
        text = _composite_text(record)
        vec = self._embedder.embed(text)
        vec_np = np.array([vec], dtype=np.float32)
        tid = record.tool_id
        if tid in self._tool_to_id:
            # Re-index: drop the old vector for this id, then re-add under it.
            iid = self._tool_to_id[tid]
            self._index.remove_ids(np.array([iid], dtype=np.int64))
        else:
            iid = self._id_for(tid)
        self._index.add_with_ids(vec_np, np.array([iid], dtype=np.int64))

    def remove_tool(self, tool_id: str) -> None:
        """Remove a tool's vector from the index and forget its id."""
        iid = self._tool_to_id.pop(tool_id, None)
        if iid is not None:
            self._id_to_tool.pop(iid, None)
            self._index.remove_ids(np.array([iid], dtype=np.int64))

    def search(self, query_vec: list[float], top_k: int = 20) -> list[tuple[str, float]]:
        """Query index. Returns [(tool_id, similarity_score)] sorted descending."""
        if self._index.ntotal == 0:
            return []
        q = np.array([query_vec], dtype=np.float32)
        scores, ids = self._index.search(q, min(top_k, self._index.ntotal))
        results: list[tuple[str, float]] = []
        for score, iid in zip(scores[0], ids[0]):
            if iid == -1:
                continue
            tid = self._id_to_tool.get(int(iid))
            if tid is not None:
                results.append((tid, float(score)))
        return results

    # ------------------------------------------------------------------
    # Full / incremental reindex
    # ------------------------------------------------------------------

    def full_reindex(self) -> int:
        """Rebuild entire index from scratch. Returns count of indexed tools."""
        self._create_empty()
        all_tools = self._tool_db.get_all_tools()
        if not all_tools:
            self._save()  # persist the empty index (e.g. after all servers removed)
            logger.info("Full reindex: no tools to index.")
            return 0

        texts = [_composite_text(t) for t in all_tools]
        vecs = self._embedder.embed_many(texts)
        vec_np = np.array(vecs, dtype=np.float32)
        ids = np.array([self._id_for(t.tool_id) for t in all_tools], dtype=np.int64)
        self._index.add_with_ids(vec_np, ids)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")
        for record in all_tools:
            self._tool_db.upsert_tool(replace(record, last_indexed_at=now))

        self._save()
        logger.info("Full reindex: %d tools -> %s", len(all_tools), self._index_path)
        return len(all_tools)

    def incremental_reindex(self) -> int:
        """Re-index tools updated since last indexed. Returns count."""
        all_tools = self._tool_db.get_all_tools()
        to_index = [t for t in all_tools if t.last_seen_at > t.last_indexed_at]
        if not to_index:
            return 0

        for record in to_index:
            self.index_tool(record)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")
        for record in to_index:
            self._tool_db.upsert_tool(replace(record, last_indexed_at=now))

        self._save()
        logger.info("Incremental reindex: %d tools", len(to_index))
        return len(to_index)

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
