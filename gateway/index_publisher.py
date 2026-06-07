"""Bridge between ToolDb changes and the Tool-RAG indexer.

On sync completion, diffs ToolDb vs indexed set and updates the vector index.
Eventual-consistency model.
"""

from __future__ import annotations

import logging

from gateway.tool_db import ToolDb
from tool_rag.indexer import ToolRagIndexer

logger = logging.getLogger(__name__)


class IndexPublisher:
    """Publishes ToolDb diffs to the Tool-RAG indexer."""

    def __init__(self, tool_db: ToolDb, indexer: ToolRagIndexer):
        self._tool_db = tool_db
        self._indexer = indexer

    def publish_sync_results(self) -> int:
        """Diff ToolDb vs indexed state and update the index.

        Returns count of tools (re-)indexed.
        """
        count = self._indexer.incremental_reindex()
        logger.info("Publisher: %d tools (re-)indexed", count)
        return count
