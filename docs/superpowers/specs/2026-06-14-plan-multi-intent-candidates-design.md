# Multi-intent candidate gathering for the `plan` meta-tool

**Date:** 2026-06-14
**Status:** Approved (design)
**Scope:** `gateway/server.py` (`handle_plan`, candidate gathering), `tool_rag/planner.py` (new `decompose`).

## Problem

The `plan` meta-tool builds a plan only from the candidate tools the retriever
surfaces for the **whole** query in a single retrieval (`_gather_candidates`,
default `top_k=10`). For a compound, multi-step task the cross-encoder reranker
scores each tool against the *entire* query, so a tool relevant to only a
minority sub-intent ranks low and never enters the candidate set — the planner
then cannot include it and reports it as `missing`.

Measured example (query: *"Search arXiv for 5 recent papers about AI in MedTech
business, navigate to abstract pages, take screenshots, extract
title/authors/summary"*):

- `playwright__browser_take_screenshot` ranked **#20 (0.099)** → absent from the
  top-10 candidates → plan emitted `"missing": ["Take screenshots of abstract pages"]`.
- The same tool ranks **#1 (1.0)** when retrieved against the focused intent
  *"take a screenshot of a web page"*.

So per-intent retrieval recovers the dropped tools; the fix is to decompose the
task and retrieve per intent before planning.

## Goal & non-goals

**Goal:** `plan` produces complete plans for compound tasks — every capability's
best tool reaches the planner — without inflating one giant `top_k`.

**Non-goals / unchanged:**
- `find_tools`, `run_tool`, `run_tools`, `describe_tool` — untouched.
- The retriever, ranker, reranker, indexer — untouched.
- The plan remains **advisory** (never executed).
- Policy scoping (allowed servers + `tool_prefixes`) is preserved on every
  retrieval, exactly as today.

## Architecture

### Components

1. **`Planner.decompose(query) -> list[str]`** (new; `tool_rag/planner.py`)
   - Added to the `Planner` ABC and implemented on `LlmPlanner`.
   - One LLM call to the same configured endpoint (`TOOL_RAG_PLANNER_URL/_MODEL/
     _API_KEY/_TEMPERATURE/timeout`), `response_format: {"type":"json_object"}`.
   - New `_DECOMPOSE_SYSTEM_PROMPT`: "split the task into short, independent
     search phrases, one per distinct capability/tool the task needs; return
     `{"intents": ["...", ...]}`."
   - Returns a list of non-empty strings. Tolerant parsing reuses the existing
     `_parse_json_object` helper. All LLM interaction stays in `planner.py`.

2. **`_retrieve_one(query, depth) -> list[dict]`** (new; `gateway/server.py`)
   - Factored out of today's `_gather_candidates`: policy-scoped
     `retriever.retrieve` + `tool_visible` filter + candidate-dict shaping, for
     **one** query. Returns up to `depth` shaped candidate dicts.
   - The existing single-query `_gather_candidates` is re-expressed in terms of
     `_retrieve_one` so the `find_tools` path is behaviourally unchanged
     (including the `tool_prefixes` over-fetch headroom).

3. **`_gather_candidates_multi(query) -> list[dict]`** (new; `gateway/server.py`)
   - Orchestrates decompose → per-intent retrieve → union. Used **only** by
     `handle_plan`.

### Data flow (inside `handle_plan`)

1. `intents = await planner.decompose(query)`, truncated to `MAX_INTENTS` (6).
2. `queries = [query] + intents` — the full query is always included as a base
   intent, so the new path can never surface worse candidates than today.
3. For each query, `_retrieve_one(q, depth=PER_INTENT_K)` run **concurrently**
   via `asyncio.gather(..., return_exceptions=True)`.
4. **Union:** flatten results; dedup by `call_name` keeping the **max** `score`
   seen across intents; sort by score descending; truncate to the union cap
   (`top_k`, default 30).
5. `planner.plan(query, candidates)` — unchanged.

Because each focused intent's top tool scores ≈1.0, every capability's best tool
floats to the top of the union and survives the cap even at modest `top_k`.

## Caps & configuration

Module-level constants, env-overridable (matching the repo's env-driven config):

| Setting | Default | Meaning |
|---|---|---|
| `TOOL_RAG_PLAN_MAX_INTENTS` | 6 | Max sub-intents kept from decomposition. |
| `TOOL_RAG_PLAN_PER_INTENT_K` | 5 | Retrieval depth per intent (and per base query). |
| `plan` arg `top_k` | **30** (was 10) | Final union cap after dedup. |

`top_k`'s meaning changes from "single-retrieval depth" to "deduped-union cap".
The default is raised 10 → 30 to suit the union.

## Error handling & fallback

The new path must never make `plan` worse than the current single-query path:

- `decompose` raises / times out / returns empty / returns non-strings →
  **fall back** to single-query `_gather_candidates(query, top_k=30)`. Logged at
  debug; not surfaced to the agent.
- Per-intent retrievals are independent (`return_exceptions=True`); a failing one
  is dropped, the rest proceed. If **all** fail → fall back to single-query,
  which itself falls back to the retriever's keyword scan when the index is empty.
- Empty union → the existing `{"steps": [], "notes": "No relevant tools found."}`
  payload.
- `planner.plan` errors are handled by the existing `try/except` in `handle_plan`.

## Performance risk: rerank latency

The reranker widens FAISS recall to ≥50 and cross-encodes up to 50 pairs **per
query** on CPU. With `[full query] + ≤6 intents` = up to 7 retrievals (~350
pairs) this is the dominant added cost.

Plan:
- Run the 7 retrievals **concurrently** (`asyncio.gather`).
- **Open question to settle during implementation:** the local `CrossEncoder`
  may not be threadsafe or may serialize on the GIL. The implementation step will
  **measure** warm end-to-end `plan` latency. If concurrency contends or is
  unsafe → fall back to sequential retrieval. If still too slow → gate per-intent
  retrieval to skip the reranker (embedding-only recall suffices here; the planner
  does final selection). Any such change is internal to the plan path.
- **Success criterion:** warm `plan` p50 under ~8s (decompose + retrievals +
  planner) on the current deployment.

## Testing & verification

No formal test suite exists in the repo; verification is script-based over the
live gateway (HTTP `/mcp` + `/tool-rag/*`), as used throughout development.

- **Logic checks:** `decompose` returns a capped list of non-empty phrases; union
  dedups by `call_name` keeping max score and respects the cap; a forced
  `decompose` error triggers the single-query fallback.
- **Integration (regression):** the MedTech compound query now yields a plan
  whose steps include `playwright__browser_take_screenshot`; a single-intent
  query (e.g. "search arXiv for papers") still plans correctly and is not
  over-decomposed into noise.
- **Latency:** measure warm `plan` p50 against the success criterion.

## Backward compatibility

- Only the `plan` path changes; all other meta-tools and the retriever are
  unchanged.
- `plan` callers that pass `top_k` still work; the value now caps the union.
- If no planner is configured, `plan` is not exposed (unchanged) and none of this
  code runs.
