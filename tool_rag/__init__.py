"""Tool-RAG: semantic tool retrieval for MCP Gateway.

Components:
  Embedder    — text → vector (local or API)
  Indexer     — manage FAISS index of tool embeddings
  Ranker      — score + filter candidates
  Retriever   — query → top-K tools
"""

from __future__ import annotations

from tool_rag.embedder import Embedder, LocalEmbedder
from tool_rag.indexer import ToolRagIndexer
from tool_rag.ranker import Ranker
from tool_rag.retriever import Retriever, RetrievalResult

__all__ = [
    "Embedder",
    "LocalEmbedder",
    "ToolRagIndexer",
    "Ranker",
    "Retriever",
    "RetrievalResult",
]