"""Optional cross-encoder reranker for Tool-RAG retrieval.

A bi-encoder (the embedder) scores query and tool independently, so spurious
surface overlap can win (e.g. an image tool whose description mentions "http://"
outranking a docs tool for a "Streamable HTTP" query). A cross-encoder scores
each (query, tool) pair *jointly* in one forward pass — far better precision —
but only on a shortlist, so it runs as a second stage over the FAISS candidates.

Configurable via TOOL_RAG_RERANKER env var (off|local). Default off.

`local` runs the model in a separate worker process (tool_rag/rerank_worker.py)
rather than in the gateway: torch + transformers + the model cost ~1GB RSS,
and only exiting a process gives all of it back. The worker starts on first
use and exits after TOOL_RAG_RERANKER_IDLE_SECS without a request, so an idle
gateway sits at ~100MB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Small, multilingual (mMARCO, 14 languages) cross-encoder — pairs well with a
# multilingual embedder like Qwen3-Embedding. Override via TOOL_RAG_RERANKER_MODEL.
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Hard cap on pairs scored per query, so cross-encoder cost stays bounded at
# catalog scale regardless of the caller's top_k / over-fetch.
MAX_RERANK_CANDIDATES = 50

# Worker start = torch import + model load: ~7s warm, longer on a first-ever
# HF download. One scoring request takes <1s; the timeout only catches a hang.
_WORKER_START_TIMEOUT = 300.0
_WORKER_REQUEST_TIMEOUT = 30.0
_START_RETRY_SECS = 60.0
# How often the idle watchdog checks; caps how late past IDLE_SECS it stops.
_IDLE_CHECK_SECS = 15.0

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _sigmoid(x: float) -> float:
    """Map a cross-encoder logit to a relevance probability in (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class Reranker(ABC):
    """Abstract (query, document) pair scorer."""

    @abstractmethod
    async def ascore(self, query: str, documents: Sequence[str]) -> list[float] | None:
        """Return one relevance score in [0, 1] per document, aligned to input
        order — or None when no scores are available right now (the caller
        keeps its first-stage ranking)."""

    async def aclose(self) -> None:
        """Release resources (no-op by default)."""


class LocalReranker(Reranker):
    """sentence-transformers CrossEncoder in this process, lazy-loaded.

    Used inside the worker process; scoring runs in a thread so it never
    blocks an event loop."""

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
            try:
                # Cached (baked into the image / hf-cache volume): skip the
                # ~3s of HF Hub round-trips — this runs on every worker start.
                self._model = CrossEncoder(self._model_name, local_files_only=True)
            except Exception:
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

    async def ascore(self, query: str, documents: Sequence[str]) -> list[float] | None:
        return await asyncio.to_thread(self.score, query, documents)


class WorkerReranker(Reranker):
    """Scores via a `python -m tool_rag.rerank_worker` subprocess.

    Protocol: one JSON object per line. The worker writes {"ready": true} (or
    {"error": ...}) once the model is loaded, then answers each
    {"query", "docs"} line with {"scores": [...]} or {"error": ...}. Requests
    are serialized by a lock — one pipe, one in flight.

    cold_start:
      "skip" — a request that finds the worker not ready starts it in the
               background and returns None (embedder-only ranking) instead of
               waiting ~7s. Ranking then differs between cold and warm calls.
      "wait" — the request waits for the worker to come up.

    idle_secs: stop the worker after this long without a request; 0 = never.
    Any worker failure (crash, timeout, bad reply) kills it and returns None;
    the next request starts a fresh one.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, idle_secs: float = 300.0, cold_start: str = "skip") -> None:
        self._model_name = model_name
        self._idle_secs = idle_secs
        self._cold_start = cold_start
        self._proc: asyncio.subprocess.Process | None = None
        self._ready = False
        self._starting: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._last_used = 0.0
        self._retry_at = 0.0  # back-off after a failed start (e.g. model can't load)

    async def ascore(self, query: str, documents: Sequence[str]) -> list[float] | None:
        if not documents:
            return []
        self._last_used = time.monotonic()
        if not self._ready:
            if time.monotonic() < self._retry_at:
                return None
            starting = self._ensure_starting()
            if self._cold_start != "wait":
                logger.info("Reranker worker not ready; ranking with embedder only while it starts")
                return None
            await asyncio.shield(starting)
            if not self._ready:
                return None
        async with self._lock:
            if not self._ready or self._proc is None:
                return None  # stopped (idle / failure) while we waited for the lock
            try:
                return await asyncio.wait_for(self._request(query, documents), _WORKER_REQUEST_TIMEOUT)
            except Exception:
                logger.exception("Reranker worker request failed; restarting it on next use")
                await self._stop()
                return None
            finally:
                self._last_used = time.monotonic()

    async def _request(self, query: str, documents: Sequence[str]) -> list[float]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        line = json.dumps({"query": query, "docs": list(documents)}) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()
        reply = json.loads(await self._read_line())
        if "error" in reply:
            raise RuntimeError(f"reranker worker: {reply['error']}")
        scores = reply["scores"]
        if len(scores) != len(documents):
            raise RuntimeError(f"reranker worker returned {len(scores)} scores for {len(documents)} docs")
        return [float(s) for s in scores]

    async def _read_line(self) -> bytes:
        assert self._proc is not None and self._proc.stdout is not None
        line = await self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"reranker worker exited (rc={self._proc.returncode})")
        return line

    def _ensure_starting(self) -> asyncio.Task:
        if self._starting is None or self._starting.done():
            self._starting = asyncio.create_task(self._start())
        return self._starting

    async def _start(self) -> None:
        t0 = time.monotonic()
        logger.info("Starting reranker worker (%s) …", self._model_name)
        try:
            # stderr is inherited, so worker logs land in the gateway's log.
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "tool_rag.rerank_worker", self._model_name,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, cwd=str(_REPO_ROOT),
            )
            hello = json.loads(await asyncio.wait_for(self._read_line(), _WORKER_START_TIMEOUT))
            if not hello.get("ready"):
                raise RuntimeError(hello.get("error", f"unexpected hello {hello!r}"))
        except Exception:
            logger.exception("Reranker worker failed to start; ranking with embedder only (retry in %.0fs)", _START_RETRY_SECS)
            self._retry_at = time.monotonic() + _START_RETRY_SECS
            await self._stop()
            return
        self._ready = True
        self._last_used = time.monotonic()
        logger.info("Reranker worker ready (pid %s) in %.1fs", self._proc.pid, time.monotonic() - t0)
        if self._idle_secs > 0 and (self._idle_task is None or self._idle_task.done()):
            self._idle_task = asyncio.create_task(self._idle_watchdog())

    async def _idle_watchdog(self) -> None:
        while self._ready:
            await asyncio.sleep(min(self._idle_secs, _IDLE_CHECK_SECS))
            if time.monotonic() - self._last_used >= self._idle_secs and not self._lock.locked():
                async with self._lock:
                    if self._ready and time.monotonic() - self._last_used >= self._idle_secs:
                        logger.info("Reranker worker idle for %.0fs; stopping it to free memory", self._idle_secs)
                        await self._stop()

    async def _stop(self) -> None:
        self._ready = False
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()  # EOF → worker exits its read loop
            await asyncio.wait_for(proc.wait(), 5.0)
        except Exception:
            proc.kill()
            await proc.wait()

    async def aclose(self) -> None:
        for task in (self._idle_task, self._starting):
            if task is not None and not task.done():
                task.cancel()
        await self._stop()


def create_reranker() -> Reranker | None:
    """Factory: instantiate reranker based on TOOL_RAG_RERANKER env var.

    Returns None when disabled (the default), leaving the pipeline unchanged.
    """
    kind = os.environ.get("TOOL_RAG_RERANKER", "off").lower()
    if kind in ("local", "on", "1", "true"):
        model = os.environ.get("TOOL_RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        try:
            idle_secs = float(os.environ.get("TOOL_RAG_RERANKER_IDLE_SECS", "300"))
        except ValueError:
            idle_secs = 300.0
        cold_start = os.environ.get("TOOL_RAG_RERANKER_COLD_START", "skip").lower()
        if cold_start not in ("skip", "wait"):
            logger.warning("Unknown TOOL_RAG_RERANKER_COLD_START=%r, using 'skip'", cold_start)
            cold_start = "skip"
        return WorkerReranker(model, idle_secs=idle_secs, cold_start=cold_start)
    return None
