"""Text → vector embedding interface and implementations.

Two strategies:
  - LocalEmbedder  — sentence-transformers (all-MiniLM-L6-v2), offline homelab.
  - ApiEmbedder    — remote OpenAI-shaped endpoint (e.g. OpenAI, Ollama) via httpx.

Configurable via TOOL_RAG_EMBEDDER env var (local|url). `api` is accepted as a
legacy alias for `url`.
"""

from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Sequence

logger = logging.getLogger(__name__)


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    """Unit-normalize a vector so inner-product == cosine similarity.

    The index is IndexFlatIP and the ranker/`reason()` thresholds assume
    semantic_score ∈ [0, 1]; remote APIs (notably Ollama) return raw,
    unnormalized vectors, so we normalize here to keep the pipeline's cosine
    assumption intact regardless of embedder.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


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

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identity of the model+endpoint, persisted with the index so a
        model change (even at the same dimension) forces a clean rebuild."""


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

    @property
    def model_id(self) -> str:
        return f"local:{self._model_name}"


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
        dim: int | None = None,
    ):
        import httpx

        self._url = url or os.environ.get("TOOL_RAG_EMBED_URL", "")
        self._model = model or os.environ.get("TOOL_RAG_EMBED_MODEL", "text-embedding-3-small")
        self._api_key = api_key or os.environ.get("TOOL_RAG_EMBED_API_KEY", "")
        # Dimension: explicit arg > TOOL_RAG_EMBED_DIM env > lazy probe (one
        # network call). Env avoids a hard dependency on the endpoint at boot.
        env_dim = os.environ.get("TOOL_RAG_EMBED_DIM", "")
        self._dim: int | None = dim if dim is not None else (int(env_dim) if env_dim else None)
        self._client = httpx.Client(timeout=30.0)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
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
            vectors = [d["embedding"] for d in data["data"]]
        elif "embeddings" in data:
            vectors = list(data["embeddings"])
        else:
            raise RuntimeError(f"Unknown embedding response shape: {list(data.keys())}")
        # Remote APIs (e.g. Ollama) return unnormalized vectors; the index uses
        # inner product, so normalize to recover cosine == semantic_score ∈ [0,1].
        return [_l2_normalize(v) for v in vectors]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dimension probe"))
            logger.info("ApiEmbedder: probed embedding dimension = %d", self._dim)
        return self._dim

    @property
    def model_id(self) -> str:
        return f"url:{self._url}#{self._model}"

    def close(self) -> None:
        self._client.close()


def create_embedder() -> Embedder:
    """Factory: instantiate embedder based on TOOL_RAG_EMBEDDER env var.

    Values: `local` (default) or `url` (remote OpenAI-shaped API). `api` is a
    legacy alias for `url`.
    """
    kind = os.environ.get("TOOL_RAG_EMBEDDER", "local").lower()
    if kind in ("url", "api"):
        return ApiEmbedder()
    return LocalEmbedder()