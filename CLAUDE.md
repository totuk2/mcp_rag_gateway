# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Homelab MCP Gateway + Tool-RAG: a single Starlette/uvicorn service that proxies multiple upstream MCP servers (stdio, SSE, Streamable HTTP) behind one authenticated `/mcp` endpoint, plus a semantic tool-retrieval API (`/tool-rag/*`) so AI clients (e.g. LibreChat Deferred Tools) discover tools at runtime instead of loading the full catalog into context. The full design spec lives in `content.md` (in Polish).

## Commands

```bash
# Run the gateway (from this directory; venv in .venv)
python -m gateway                      # uvicorn on 0.0.0.0:8765

# Provision upstream servers from servers/*/manifest.yaml (see below)
python provision.py                    # or: make provision  (--host for host-run gateway)

# Docker
docker compose up --build              # gateway only; mounts ./config read-only
make up                                # provision + gateway + provisioned docker servers

# Install deps
pip install -r requirements.txt
```

## Adding upstream servers (provisioning)

`provision.py` turns a per-server `servers/<id>/manifest.yaml` into runnable config.
To add a server: drop sources (or `git clone` a repo) into `servers/<id>/`, add a
`manifest.yaml` (see `servers/MANIFEST.example.yaml`), run `python provision.py`, restart
the gateway. Tool-RAG resyncs and reindexes on startup, so the tools register themselves.

Manifest `kind`:
- `stdio` — runs `setup` once (fingerprinted via a `.provisioned` marker; `--force` re-runs), registers a stdio subprocess entry.
- `docker` — builds the image, emits a service into `docker-compose.servers.yml`, registers a `streamable_http`/`sse` URL (service name in compose, or `127.0.0.1:<port>` with `--host`).
- `remote` — registers an existing URL as-is.

When asked to "add a server from `<url>`/`<git repo>`", follow the **Agent playbook**
in README.md (the canonical step-by-step). In short: pick an `id`; classify the
source (running MCP URL → `remote`; repo with a `Dockerfile` → `docker`; repo that
runs as a process → `stdio`); for non-`remote`, `git clone <repo> servers/<id>` and
**read the repo's README** for the exact run command + transport; write
`servers/<id>/manifest.yaml`; `python provision.py`; grant the key in `keys.yaml`;
restart. Removing: delete `servers/<id>/`, drop its `keys.yaml` block, re-provision,
restart — the startup sync reconciles and prunes the server's tools.

Outputs are **generated, gitignored, never hand-edited**: `config/registry.generated.yaml`
and `docker-compose.servers.yml`. At startup `load_registries()` (registry.py) merges the
generated registry under the hand-written `config/registry.yaml`, and **hand-written entries
win on id conflict**. `servers/echo/` is the reference stdio example.

There is no test suite, linter config, or CI in this repo. The repo is not a git repository.

Smoke checks: `GET /health` (no auth), `POST /tool-rag/retrieve` with `{"query": "..."}`, MCP via `/mcp` with `Authorization: Bearer <key>`.

Configuration is entirely via env vars (see README.md table) + two YAML files in `config/`: `registry.yaml` (upstream servers) and `keys.yaml` (API keys → access policies). `servers/echo/` (manifest.yaml + server.py) is the reference stdio upstream.

## Architecture

Two packages compose into one ASGI app, assembled in `gateway/app.py:app_from_env()`:

**gateway/** — MCP proxy layer
- `app.py` — Starlette app factory; `APIKeyMiddleware` resolves Bearer token → `AccessPolicy` and stores it in a contextvar (`context.py`); lifespan hook runs the initial Tool-RAG sync + index build.
- `server.py` — the merged MCP `Server`. Every handler re-reads the request policy via `get_policy()`. **Admin lockdown:** `list_tools()` returns only the `find_tools` meta-tool for non-admin keys (admin keys also get the full catalog). `find_tools` is the in-band discovery entry point: calling it runs the Tool-RAG retriever **scoped to the key's allowed servers + `tool_prefixes`** (via `get_policy()`) and returns matching tools with `call_name` + `input_schema`. The raw `POST /tool-rag/retrieve` HTTP handler (`tool_rag/router.py`) is policy-scoped the same way (reads `current_policy`, narrows the body's `allowed_servers` to the key's grants, applies `tool_prefixes`); enforcement is skipped only under `TOOL_RAG_WITHOUT_AUTH=1` (no policy in context by design). The `Server` also sets the MCP `initialize` `instructions` field (only when Tool-RAG is on) telling agents to call `find_tools` first. `call_tool` still works for any allowed tool by name — including tools never listed (discovered via `find_tools`); strict clients that validate calls against the advertised list are a known limitation. The retriever is wired in via `build_gateway_server(registry, retriever=...)`.
- `backends.py` — `open_upstream_session()`: opens a *fresh* upstream session per request (stateless; no connection pooling — a stdio upstream spawns a new subprocess on every call).
- `merge.py` — namespacing: merged tool/prompt names are `{server_id}__{original}` (double underscore — `server_id` must never contain `__`); resources get opaque `gateway://{server_id}/{base64(uri)}` URIs.
- `registry.py` / `auth.py` / `policy.py` — YAML loaders; `AccessPolicy` gates per-server access with optional `tool_prefixes` / `uri_prefixes` / `prompt_prefixes` allowlists (empty = full access to that server).
- `tool_db.py` / `tool_record.py` — SQLite tool registry (source of truth for Tool-RAG); `ToolRecord` carries metadata like `tool_type` (read/write/admin/action/query), `permission_scope`, `status`.
- `sync_adapter.py` — connects to every upstream at startup, calls `list_tools()`, upserts into ToolDb; `_infer_tool_type()` heuristically classifies tools from name/description keywords. `full_sync()` also **reconciles**: tools whose `server_id` is no longer in the registry are deleted (removed servers don't orphan in the DB). Down-but-registered servers are kept (hidden at query time by liveness, not deleted).
- `health.py` — `ServerHealth` tracks which upstreams are currently reachable; a background loop (`health_loop`) probes them every `TOOL_RAG_HEALTHCHECK_INTERVAL`s (concurrent via `asyncio.gather`, stdio treated as always-up). The retriever drops tools of down servers. Fail-open: empty down-set ⇒ all up.
- `index_publisher.py` — bridges ToolDb changes into the FAISS index (used by the `incremental` startup mode).

**tool_rag/** — semantic retrieval layer
- `embedder.py` — `create_embedder()` factory: `LocalEmbedder` (sentence-transformers all-MiniLM-L6-v2, 384-dim, lazy-loaded) or `ApiEmbedder` (OpenAI/Ollama-shaped HTTP), selected by `TOOL_RAG_EMBEDDER=local|api`.
- `indexer.py` — FAISS `IndexIDMap(IndexFlatIP)` over one composite text per tool (name + description + type + server + input-schema property names + tags; no document chunking). Each tool has a stable int64 id, so `remove_tool`/re-index actually drop the old vector (plain `IndexFlatIP` can't remove — that previously leaked one orphaned vector per tool per reindex). Persisted to `tool_rag.index` / `tool_rag.meta` (meta carries a `version`; a stale/old-format meta forces a clean rebuild). Supports full and incremental reindex (incremental = tools where `last_indexed_at < last_seen_at`).
- `retriever.py` — pipeline: embed query → FAISS top-K×2 → metadata filters (allowed_servers / permission_scope / tool_type / status=active / **server liveness**) → rank → top-K. Falls back to keyword-only DB scan if the index is empty (`fallback_used: true`); the liveness filter applies on both paths.
- `ranker.py` — final score = 1.0×semantic + 0.25×keyword + 0.15×metadata + policy penalty; deterministic tie-break on `tool_id`.
- `router.py` — Starlette handlers for `/tool-rag/retrieve|reindex|health|metrics`.

Key flow on startup: registry load → `full_sync()` (for each upstream `list_tools()` → SQLite ToolDb, + reconcile removed servers) → reindex per `TOOL_RAG_STARTUP_REINDEX` (default `full` = clean rebuild; `incremental`; `off`) → initial liveness probe + optional background loops. Reindex on demand via `POST /tool-rag/reindex {"mode": "full"|"incremental"}` — but this rebuilds **from the DB only**, it does not re-query upstreams. To pick up tool changes on a *running* upstream without a restart, set `TOOL_RAG_RESYNC_INTERVAL > 0` (background re-pull); otherwise restart.

## Conventions and constraints

- `server_id` must match `^[A-Za-z0-9._-]+$` and must not contain `__` (it's the namespace separator).
- Design rules from the spec (`content.md`): the registry/ToolDb is the source of truth; the public API must never expose the full tool list to non-admin keys; retrieval must be deterministic for identical input + index; prefer adding adapters over rewriting existing gateway code.
- Auth applies to everything except `/health` (and `/tool-rag/*` only when `TOOL_RAG_WITHOUT_AUTH=1`).
- Per-request upstream sessions mean no shared state with upstream servers — keep handlers stateless and policy checks inside each handler.
