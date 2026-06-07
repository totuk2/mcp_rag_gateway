"""Text → vector embedding interface and implementations.

Two strategies:
  - LocalEmbedder  — sentence-transformers (all-MiniLM-L6-v2), offline homelab.
  - ApiEmbedder    — remote endpoint (e.g. OpenAI, Ollama) via httpx.

Configurable via TOOL_RAG_EMBEDDER env var (local|api).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Sequence

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract text embedder."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector."""

    @abstractmethod
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""


class LocalEmbedder(Embedder):
    """sentence-transformers local embedder (all-MiniLM-L6-v2 → 384 dims)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            logger.info("Loading embedding model %s …", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded (dim=%d).", self.dimension)

    def embed(self, text: str) -> list[float]:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()  # type: ignore[union-attr]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        emb = self._model.encode(list(texts), normalize_embeddings=True)  # type: ignore[union-attr]
        return [e.tolist() for e in emb]

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]


class ApiEmbedder(Embedder):
    """Remote HTTP API embedder.

    Expects a POST endpoint that accepts JSON:
        {"model": "...", "input": [text1, text2, ...]}
    and returns:
        {"data": [{"embedding": [float, ...]}, ...]}
    Compatible with OpenAI / Ollama / LiteLLM proxy shapes.
    """

    def __init__(
        self,
        url: str = "",
        model: str = "",
        api_key: str = "",
        dim: int = 384,
    ):
        import httpx

        self._url = url or os.environ.get("TOOL_RAG_EMBED_URL", "")
        self._model = model or os.environ.get("TOOL_RAG_EMBED_MODEL", "text-embedding-3-small")
        self._api_key = api_key or os.environ.get("TOOL_RAG_EMBED_API_KEY", "")
        self._dim = dim
        self._client = httpx.Client(timeout=30.0)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        if not self._url:
            raise RuntimeError("ApiEmbedder: TOOL_RAG_EMBED_URL not configured")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {"model": self._model, "input": list(texts)}
        resp = self._client.post(self._url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI: data[0].embedding, Ollama: embeddings[i]
        if "data" in data:
            return [d["embedding"] for d in data["data"]]
        if "embeddings" in data:
            return list(data["embeddings"])
        raise RuntimeError(f"Unknown embedding response shape: {list(data.keys())}")

    @property
    def dimension(self) -> int:
        return self._dim

    def close(self) -> None:
        self._client.close()


def create_embedder() -> Embedder:
    """Factory: instantiate embedder based on TOOL_RAG_EMBEDDER env var."""
    kind = os.environ.get("TOOL_RAG_EMBEDDER", "local").lower()
    if kind == "api":
        return ApiEmbedder()
    return LocalEmbedder()