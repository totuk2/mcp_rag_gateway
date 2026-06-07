# MCP Gateway + Tool-RAG

**Single MCP endpoint** with **semantic tool retrieval**. The gateway proxies
multiple upstream MCP servers (stdio, SSE, Streamable HTTP) behind one
authenticated URL; Tool-RAG selects the right tool for each query instead of
loading every tool description into the model's context.

## Features

- **Unified MCP endpoint** — one URL, many upstream servers behind auth.
- **Semantic tool retrieval** — FAISS + sentence-transformers search.
- **Admin lockdown** — `list_tools()` returns nothing for non-admin keys; they discover tools via Tool-RAG.
- **Multi-transport** — stdio, SSE, Streamable HTTP upstreams.
- **Per-key policies** — server + prefix filters per API key.
- **Agent-friendly provisioning** — add a server from a URL or git repo with one command; an AI agent can follow the [playbook](#agent-playbook-add-a-server-from-a-url-or-repo) end-to-end.
- **Persistent registry** — SQLite tool store, synced on startup.
- **Tool-RAG API** — retrieve, reindex, health, metrics.
- **LibreChat-ready** — works with the Deferred Tools flow.

---

## Quick start

```bash
pip install -r requirements.txt

# Copy the examples if starting fresh:
cp config/registry.example.yaml config/registry.yaml
cp config/keys.example.yaml config/keys.yaml

python -m gateway                 # uvicorn on 0.0.0.0:8765
```

- Health:    `GET /health`            (no auth)
- MCP:       `/mcp`                    (`Authorization: Bearer <key>`)
- Tool-RAG:  `POST /tool-rag/retrieve` (`Authorization: Bearer <key>`)

### Docker

```bash
docker compose up --build         # gateway only; mounts ./config read-only
make up                           # provision + gateway + provisioned docker servers
```

---

## Adding upstream servers

**In one sentence:** put the server in `servers/<id>/`, describe it in a
`manifest.yaml`, run `python provision.py`, grant a key, restart.

### Agent playbook: add a server from a URL or repo

This repo is built so an AI agent can add a server from a single instruction like
*"add the server from https://mcp.example.com/mcp"* or *"add the server from
https://github.com/org/repo.git"*. Follow these steps deterministically.

**1 — Pick an `id`.** Lowercase, `^[A-Za-z0-9._-]+$`, no `__`. Derive it from the
repo/domain name (e.g. `repo`, `example-weather`).

**2 — Classify the source → `kind`:**

| The source is… | `kind` | Action |
|----------------|--------|--------|
| A URL that is already a **running** MCP endpoint (path ends in `/mcp` or `/sse`, or the user says "remote/hosted") | `remote` | nothing to fetch/build — just register the URL |
| A git repo / source tree containing a **`Dockerfile`** | `docker` | build + run as its own container |
| A git repo / source tree that runs as a local **process** (Python/Node/Go, no Dockerfile) | `stdio` | install deps once, run as a subprocess |

If a bare domain is given with no path (`https://example.com/`), assume `remote`
and try `/mcp` (Streamable HTTP) first, then `/sse`. If neither responds, ask the
user for the MCP URL.

**3 — Fetch the source** (skip for `remote`):

```bash
git clone <repo-url> servers/<id>        # or copy sources into servers/<id>/
```

Then **read the cloned repo's README** to find its exact run command and which
transport it speaks — MCP servers differ. Use that to fill `command` / `port` /
`transport` below.

**4 — Write `servers/<id>/manifest.yaml`** from the matching template:

```yaml
# remote — already-running MCP server
id: <id>
kind: remote
transport: streamable_http        # or: sse
url: "https://mcp.example.com/mcp"
headers:                          # optional
  Authorization: "Bearer <token>"
```

```yaml
# docker — repo with a Dockerfile
id: <id>
kind: docker
build: .                          # context (dir with the Dockerfile), default "."
port: 9000                        # port the server listens on inside the container
transport: streamable_http        # or: sse
path: /mcp                        # MCP path (default /mcp, or /sse for sse)
# command: ["serve", "--port", "9000"]   # optional, overrides the image CMD
```

```yaml
# stdio — repo that runs as a local process
id: <id>
kind: stdio
setup:                            # one-time install (re-runs only if this changes)
  - "pip install -r requirements.txt"   # or: "npm ci && npm run build", "go build -o bin/server ./..."
command: ["python", "server.py"]  # or: ["node", "dist/index.js"], ["./bin/server"]
env:                              # optional
  LOG_LEVEL: info
```

**5 — Provision, grant a key, restart:**

```bash
python provision.py               # add --host if the gateway runs on the host (not in compose)
# then grant access: add `<id>: {}` under a key's `servers:` in config/keys.yaml
python -m gateway                 # restart; Tool-RAG resyncs + reindexes automatically
```

**6 — Verify:** `POST /tool-rag/retrieve {"query": "<something the server does>"}`
returns its tools, and `GET /tool-rag/metrics` shows `tools_in_db` increased.

> **Removing a server:** delete `servers/<id>/` (or just its `manifest.yaml`),
> remove its block from `config/keys.yaml`, run `python provision.py`, and restart.
> The startup sync reconciles the registry and drops the server's tools
> automatically — no manual DB surgery. For `docker` kind, also
> `docker image rm mcp-server-<id>:latest`.

### Provisioning reference

`provision.py` turns each `servers/<id>/manifest.yaml` into runnable config and
writes two **generated** files (gitignored, never hand-edit):

- `config/registry.generated.yaml` — merged at startup *under* `config/registry.yaml`.
- `docker-compose.servers.yml` — `docker`-kind servers, joined to the gateway's network.

What each `kind` does:

| kind     | What provision does                                                         | Registered as                                 |
|----------|-----------------------------------------------------------------------------|-----------------------------------------------|
| `stdio`  | runs `setup` once (re-runs only when it changes, or with `--force`)         | stdio subprocess (`command` / `args` / `cwd`) |
| `docker` | `docker build` the image, emits a service into `docker-compose.servers.yml` | `streamable_http`/`sse` URL                   |
| `remote` | nothing to build                                                            | the given URL, as-is                          |

Flags: `--host` (gateway runs on the host → docker servers publish ports on
`127.0.0.1`), `--force` (re-run setup / rebuild images), `--only <id>`.

See `servers/MANIFEST.example.yaml` for the full field reference, and
`servers/echo/` for a working stdio example.

### Hand-editing the registry

Add an entry under `servers:` in `config/registry.yaml`:

```yaml
servers:
  echo:
    transport: stdio
    command: python
    args: ["servers/echo/server.py"]

  example_sse:
    transport: sse
    url: "http://127.0.0.1:9000/sse"
    headers: {}

  example_streamable:
    transport: streamable_http
    url: "http://127.0.0.1:9000/mcp"
    headers: {}
```

On id conflict, hand-written `registry.yaml` entries **win** over the generated
ones. Grant access to the server in `keys.yaml`, then restart.

### Naming

Merged tool/prompt names are `server_id__original` (two underscores).
`server_id` must match `^[A-Za-z0-9._-]+$` and must not contain `__`.

---

## API keys

`config/keys.yaml` maps Bearer tokens to access policies:

```yaml
keys:
  - id: dev-full
    secret: "dev-key-full-access"
    servers:
      echo: {}                        # full access to this server

  - id: dev-restricted
    secret: "dev-key-restricted"
    servers:
      echo:
        tool_prefixes: ["ping"]       # only tools starting with "ping"
```

- One of `secret` / `secret_hash` is required per entry. Hashed form:
  `secret_hash: "sha256:<hex>"` — prefer this outside a trusted lab network.
- Per-server rules: `tool_prefixes`, `uri_prefixes`, `prompt_prefixes` (empty = full access).
- `admin: true` lets a key call `list_tools()` and see the full catalog.

---

## Tool-RAG

On startup the gateway connects to every upstream, calls `list_tools()`, stores
metadata in SQLite, and rebuilds the FAISS vector index.

One record = one tool (no document chunking). The embedding text is composite:
**name + description + type + server + input-schema fields + tags**.

Final ranking score = `1.0 × semantic + 0.25 × keyword + 0.15 × metadata + policy_penalty`,
with a deterministic tie-break on `tool_id`. If the index is empty it falls back
to a keyword scan (`fallback_used: true`).

### Startup, refresh, and liveness

- **Clean rebuild on startup.** `TOOL_RAG_STARTUP_REINDEX=full` (default) rebuilds
  the index from scratch each boot, so added/changed/removed tools are reflected
  and the index stays leak-free. `incremental` only re-embeds changed tools;
  `off` skips reindex and uses the persisted index as-is.
- **Removed servers/tools are purged.** Each startup sync reconciles the registry:
  tools of a server no longer in the registry are deleted from the DB (and drop
  out of the rebuilt index). Just remove the server and restart.
- **Down servers are withheld.** A background loop probes upstreams every
  `TOOL_RAG_HEALTHCHECK_INTERVAL` seconds; tools of an unreachable server are
  excluded from `retrieve` until it recovers (staleness window = one interval).
  stdio upstreams are treated as always-up (they're spawned per call). Set the
  interval to `0` to disable.
- **Picking up live tool changes.** By default, a tool added/deprecated on an
  *already-running* upstream is picked up on the next restart. Set
  `TOOL_RAG_RESYNC_INTERVAL > 0` to re-pull `list_tools()` from upstreams in the
  background on that interval instead. Note: `POST /tool-rag/reindex` rebuilds from
  the local DB only — it does **not** re-query upstreams.

### API

All endpoints under `/tool-rag/`, Bearer-authed like `/mcp` (unless
`TOOL_RAG_WITHOUT_AUTH=1`).

#### `POST /tool-rag/retrieve`

```json
{
  "query": "check inventory for SKU 12345",
  "top_k": 5,
  "allowed_servers": ["warehouse-service"],
  "permission_scope": "read",
  "tool_type": "query"
}
```

Returns `query`, `results` (each with `tool_id`, `tool_name`, `server_name`,
`score`, `reason`, `description`, `input_schema`, `status`, `tool_type`), and
`fallback_used`.

#### `POST /tool-rag/reindex`

Body `{"mode": "full"}` (rebuild) or `{"mode": "incremental"}` (dirty tools only).

#### `GET /tool-rag/health`
Index size, DB size, `started_at`.

#### `GET /tool-rag/metrics`
`tools_in_index`, `tools_in_db`, `active_servers`, `stale_entries`.

---

## Configuration

### Environment variables

| Variable                       | Default                  | Purpose                          |
|--------------------------------|--------------------------|----------------------------------|
| `MCP_GATEWAY_CONFIG_DIR`       | `./config`               | YAML directory                   |
| `MCP_GATEWAY_REGISTRY`         | `registry.yaml`          | Hand-written registry path       |
| `MCP_GATEWAY_REGISTRY_GENERATED` | `registry.generated.yaml` | Provisioner-generated registry |
| `MCP_GATEWAY_KEYS`             | `keys.yaml`              | API keys path                    |
| `MCP_GATEWAY_MCP_PATH`         | `/mcp`                   | MCP HTTP path                    |
| `MCP_GATEWAY_HOST`             | `0.0.0.0`                | Bind address                     |
| `MCP_GATEWAY_PORT`             | `8765`                   | Port                             |
| `TOOL_RAG_ENABLED`             | `1`                      | Enable Tool-RAG                  |
| `TOOL_RAG_EMBEDDER`            | `local`                  | `local` or `api`                 |
| `TOOL_RAG_EMBED_URL`           | —                        | Remote embedding API URL         |
| `TOOL_RAG_DB`                  | `tool_registry.db`       | SQLite path                      |
| `TOOL_RAG_WITHOUT_AUTH`        | `0`                      | Skip auth for `/tool-rag/`       |
| `TOOL_RAG_STARTUP_REINDEX`     | `full`                   | `full` \| `incremental` \| `off` — index strategy at startup |
| `TOOL_RAG_HEALTHCHECK_INTERVAL`| `30`                     | Seconds between upstream liveness probes; `0` disables |
| `TOOL_RAG_HEALTHCHECK_TIMEOUT` | `5`                      | Per-probe connect timeout (seconds) |
| `TOOL_RAG_RESYNC_INTERVAL`     | `0`                      | Seconds between background re-pull of upstream tool lists; `0` = off |
| `UVICORN_LOG_LEVEL`            | `info`                   | Uvicorn log level                |

---

## Client integration

### LibreChat

Point an MCP server at `http://gateway:8765/mcp` with a **non-admin** token. The
Deferred Tools flow discovers tools through `/tool-rag/retrieve` at runtime
instead of loading the full catalog.

### Generic clients

Call `POST /tool-rag/retrieve` to get the top-K tools, pick one, then call it via
`/mcp` using the merged `server_id__tool` name.

---

## Project structure

```
provision.py            Manifest -> registry.generated.yaml + docker-compose.servers.yml
Makefile                provision / run / up convenience targets
gateway/
  app.py                Starlette app + routes + lifespan sync
  auth.py               API key -> AccessPolicy
  backends.py           open_upstream_session() (fresh session per request)
  server.py             merged MCP Server impl + policy enforcement
  sync_adapter.py       pulls tool metadata from upstreams + reconciles removed servers
  health.py             upstream liveness probing (ServerHealth + background loop)
  tool_db.py            SQLite tool store (WAL)
  tool_record.py        ToolRecord dataclass
  index_publisher.py    ToolDb -> FAISS index bridge (incremental startup mode)
  merge.py              namespace merging (server_id__tool, gateway:// URIs)
  policy.py             AccessPolicy, ServerRule
  registry.py           registry loaders + load_registries() merge
  context.py            request-scoped policy contextvar
tool_rag/
  embedder.py           text -> vector (local sentence-transformers or remote API)
  indexer.py            FAISS index (IndexIDMap(IndexFlatIP), removable vectors)
  ranker.py             scoring
  retriever.py          query -> top-K pipeline
  router.py             /tool-rag/* route handlers
servers/
  echo/                 reference stdio server (manifest.yaml + server.py)
  MANIFEST.example.yaml  manifest reference for all three kinds
config/
  registry.yaml, keys.yaml   (+ .example. variants)
```

---

## License

Internal homelab use.
