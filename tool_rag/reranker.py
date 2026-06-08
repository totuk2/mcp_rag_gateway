"""Optional cross-encoder reranker for Tool-RAG retrieval.

A bi-encoder (the embedder) scores query and tool independently, so spurious
surface overlap can win (e.g. an image tool whose description mentions "http://"
outranking a docs tool for a "Streamable HTTP" query). A cross-encoder scores
each (query, tool) pair *jointly* in one forward pass — far better precision —
but only on a shortlist, so it runs as a second stage over the FAISS candidates.

Configurable via TOOL_RAG_RERANKER env var (off|local). Default off.
"""

from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Sequence

logger = logging.getLogger(__name__)

# Small, multilingual (mMARCO, 14 languages) cross-encoder — pairs well with a
# multilingual embedder like Qwen3-Embedding. Override via TOOL_RAG_RERANKER_MODEL.
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Hard cap on pairs scored per query, so cross-encoder cost stays bounded at
# catalog scale regardless of the caller's top_k / over-fetch.
MAX_RERANK_CANDIDATES = 50


def _sigmoid(x: float) -> float:
    """Map a cross-encoder logit to a relevance probability in (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class Reranker(ABC):
    """Abstract (query, document) pair scorer."""

    @abstractmethod
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score in [0, 1] per document, aligned to input order."""


class LocalReranker(Reranker):
    """sentence-transformers CrossEncoder, lazy-loaded."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            logger.info("Loading reranker model %s …", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Reranker model loaded.")

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        self._load()
        pairs = [(query, doc) for doc in documents]
        raw = self._model.predict(pairs)  # type: ignore[union-attr]
        # CrossEncoder.predict returns logits; squash to [0,1] so the score can
        # feed the ranker's semantic_score slot cleanly.
        return [_sigmoid(float(s)) for s in raw]


def create_reranker() -> Reranker | None:
    """Factory: instantiate reranker based on TOOL_RAG_RERANKER env var.

    Returns None when disabled (the default), leaving the pipeline unchanged.
    """
    kind = os.environ.get("TOOL_RAG_RERANKER", "off").lower()
    if kind in ("local", "on", "1", "true"):
        model = os.environ.get("TOOL_RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        return LocalReranker(model)
    return None
