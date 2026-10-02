"""Reranker worker process, spawned by reranker.WorkerReranker.

Usage: python -m tool_rag.rerank_worker <model_name>

Loads the CrossEncoder once, then answers JSON-line requests on stdin:
  in:  {"query": "...", "docs": ["...", ...]}
  out: {"scores": [...]}  or  {"error": "..."}
The first line written is {"ready": true} (or {"error": ...} if the model
fails to load). Exits on stdin EOF — so it also dies with the gateway.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from tool_rag.reranker import LocalReranker


def main() -> int:
    # Keep stdout for the protocol only: library prints / progress bars that
    # write to fd 1 are redirected to stderr (the gateway's log).
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s rerank-worker: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # HF Hub request spam

    reranker = LocalReranker(sys.argv[1])
    try:
        reranker._load()
    except Exception as e:
        proto.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        return 1
    proto.write(json.dumps({"ready": True}) + "\n")

    for line in sys.stdin:
        try:
            req = json.loads(line)
            out = {"scores": reranker.score(req["query"], req["docs"])}
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        proto.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
