> ⚠️ **Research / experimental — beta.** This project is a homelab research
> prototype, not production-hardened software. APIs, schemas, and behavior may
> change without notice. Use at your own risk.

# MCP Gateway + Tool-RAG

**One authenticated MCP endpoint in front of many (potentially hundreds of)
upstream MCP servers**, with **semantic tool retrieval** so an agent loads only
the tools relevant to its current query instead of the full catalog.

The point is **context-window economy at catalog scale**: as you connect dozens
or hundreds of servers, loading every tool's description into the model's context
becomes impractical. The gateway proxies all upstreams (stdio, SSE, Streamable
HTTP) behind a single URL, and Tool-RAG does semantic search over the combined
tool catalog so each query surfaces just the relevant tools.

## Features

- **Unified MCP endpoint** — one URL, many upstream servers behind auth.
- **Semantic tool retrieval** — FAISS + sentence-transformers search.
- **In-band discovery** — non-admin keys see two meta-tools via `list_tools()`: `find_tools` (semantic retrieval → matching tools + schemas) and `run_tool` (execute any discovered tool). Works with any MCP client, no out-of-band config.
- **Strict-client support** — after `find_tools`, the gateway registers the discovered tools for that session and emits `tools/list_changed`, so clients that validate calls against the advertised list (e.g. LibreChat) can call them.
- **Multi-transport** — stdio, SSE, Streamable HTTP upstreams.
- **Per-key policies** — server + prefix filters per API key.
- **Agent-friendly provisioning** — add a server from a URL or git repo with one command; an AI agent can follow the [playbook](#agent-playbook-add-a-server-from-a-url-or-repo) end-to-end.
- **Persistent registry** — SQLite tool store, synced on startup.
- **Tool-RAG API** — retrieve, reindex, health, metrics.
- **LibreChat-ready** — works with the Deferred Tools flow.

### Design notes & caveats

- **The non-admin lockdown is what keeps context small.** `list_tools()` returns
  only the `find_tools` + `run_tool` meta-tools for non-admin keys
  (`gateway/server.py`), so a client never loads the full catalog. The agent
  discovers tools by *calling* `find_tools` (in-band, MCP-native), then executes
  them via `run_tool` (or by name, for clients that allow unlisted calls).
- **Both discovery paths are policy-scoped.** `find_tools` and `POST
  /tool-rag/retrieve` both read the caller's `AccessPolicy` and restrict results
  to the key's granted servers + `tool_prefixes`. A body-supplied `allowed_servers`
  can only *narrow* within the grant, never broaden it. Enforcement is skipped
  only when `TOOL_RAG_WITHOUT_AUTH=1` (no policy in context by design).
- **Why a meta-tool, not just the HTTP route.** An MCP agent can only invoke MCP
  *tools*; it cannot issue a raw `POST /tool-rag/retrieve` (only the host framework
  can). `find_tools` is therefore the only discovery path a generic agent can
  reach on its own. The gateway also sets the MCP `initialize` `instructions` field
  telling the agent to call `find_tools` first.
- **Calling discovered tools.** `find_tools` returns each tool's `call_name`. Two
  ways to execute it: the `run_tool` meta-tool (always works — it's in
  `list_tools()`), or a direct call by `call_name`. The direct call works on this
  gateway (`call_tool` executes any *allowed* tool regardless of listing) **for
  clients that let the model emit an unlisted name**; for strict clients that
  validate against the advertised list, `find_tools` registers the discovered
  tools for the session and emits `tools/list_changed` so they re-fetch and accept
  the call (`run_tool` sidesteps the issue entirely).
- **Schemas are returned eagerly.** Both `find_tools` and `/tool-rag/retrieve`
  attach each matched tool's full `input_schema` in the same response — there is
  no separate "describe tool" call (`tool_rag/router.py:76-93`). This favours
  **call reliability** (the exact schema is in context when the model builds
  arguments) at some cost to context savings.
- **Context savings today = "catalog → top-K," not "names-only shortlist + lazy
  schema fetch."** Because each returned tool carries its full schema, a retrieve
  of many tools can still be heavy. The savings scale with **selectivity**:
  retrieve many, call few. A lighter two-phase discovery is on the
  [roadmap](#roadmap--future-considerations).

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

Configuration is read from a `.env` file (loaded by the gateway service's
`env_file`). Copy the template and edit:

```bash
cp .env.example .env              # then tweak; .env is gitignored (may hold secrets)
```

`.env` is optional — the gateway boots on built-in defaults if it's absent.

```bash
docker compose up --build         # gateway only; mounts ./config read-only, loads .env
make up                           # provision (host Python) + gateway + provisioned docker servers
make up-docker                    # same, but provisioning runs in a container too — no host Python needed
```

`make up-docker` is the fully-containerized path. It runs provisioning in a
throwaway container (the gateway image already has PyYAML; the repo is mounted so
the generated files land on the host), then builds and runs the gateway and every
docker-kind server together:

```bash
docker compose run --rm provision                                  # generate registry + servers compose
docker compose -f docker-compose.yml -f docker-compose.servers.yml up --build
```

Docker-kind servers are **built by `compose up --build`**, not by the provisioner —
so the provisioner needs no Docker socket. Pass flags through `run`, e.g.
`docker compose run --rm provision --force`.

---

## Adding upstream servers

**In one sentence:** put the server in `servers/<id>/`, describe it in a
`manifest.yaml`, run `python provision.py`, grant a key, restart.

**Or do it the easiest way:** let `claude` handle it using SKILL `add-mcp` followed by URL of the 
MCP server or pointing to a server code in the `servers/*` folder.

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
env:                              # optional, set inside the container
  LOG_LEVEL: info                 #   literal
  API_KEY: "${IMAGES_KEY}"        #   interpolated from the root .env at `compose up`
# env_file: [.env]                # optional env file(s) relative to servers/<id>/
```

> **Server secrets:** put a docker server's secret env in the root `.env` and
> reference it from the manifest as `${VAR}` — `docker compose up` interpolates
> it, so the secret never lands in the committed manifest. (Interpolation
> applies to `docker`-kind servers only; `stdio` `env:` values are literal.)

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
| `docker` | emits a service (with `build:` context) into `docker-compose.servers.yml`; the image is built by `compose up --build` | `streamable_http`/`sse` URL                   |
| `remote` | nothing to build                                                            | the given URL, as-is                          |

Flags: `--host` (gateway runs on the host → docker servers publish ports on
`127.0.0.1`), `--force` (re-run stdio `setup` steps; docker images are rebuilt by
`compose up --build`, not here), `--only <id>`.

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

**Optional cross-encoder reranking.** With `TOOL_RAG_RERANKER=local`, a second
stage scores each `(query, tool)` pair jointly with a small multilingual
cross-encoder and replaces the bi-encoder's `semantic` term — far better
precision when surface tokens mislead the bi-encoder (e.g. an image tool that
mentions `http://` outranking a docs tool for a "Streamable HTTP" query). It
runs only on the FAISS shortlist (bounded to 50 candidates), so cost stays fixed
at catalog scale. Default off (no extra model download). Pairs naturally with a
stronger multilingual `url` embedder. When `TOOL_RAG_RERANKER=local`, `docker
compose build` bakes the model into the image (warm at boot, no runtime HF fetch);
the `hf-cache` volume otherwise downloads it lazily on first use and persists it.

### Startup, refresh, and liveness

- **Clean rebuild on startup.** `TOOL_RAG_STARTUP_REINDEX=full` (default) rebuilds
  the index from scratch each boot, so added/changed/removed tools are reflected
  and the index stays leak-free. `incremental` only re-embeds changed tools;
  `off` skips reindex and uses the persisted index as-is.
- **Changing the embedding model is safe.** The index meta records the
  embedder's `dim` and `model_id`; on startup `indexer._load()` rebuilds the
  index from scratch if either differs from the current embedder — so swapping
  the model (or its dimension) is just "change the env var, restart," with no
  stale-vector trap even under `incremental`/`off` reindex. (Update
  `TOOL_RAG_EMBED_DIM` to match the new model, or unset it to auto-probe.)
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

Results are scoped to the calling key's policy: the request's `allowed_servers`
is intersected with the key's granted servers (it can only narrow, never
broaden), and per-server `tool_prefixes` are applied. Scoping is skipped only
under `TOOL_RAG_WITHOUT_AUTH=1`.

#### `POST /tool-rag/reindex`

Body `{"mode": "full"}` (rebuild) or `{"mode": "incremental"}` (dirty tools only).

#### `GET /tool-rag/health`
Index size, DB size, `started_at`.

#### `GET /tool-rag/metrics`
`tools_in_index`, `tools_in_db`, `active_servers`, `stale_entries`.

---

## Configuration

Under Docker, set these in `.env` (copy `.env.example`); it's loaded into the
gateway container at `docker compose up`. For a host run (`python -m gateway`),
export them in your shell instead. All variables are optional — defaults below.

### Environment variables

| Variable                       | Default                  | Purpose                          |
|--------------------------------|--------------------------|----------------------------------|
| `MCP_GATEWAY_CONFIG_DIR`       | `./config`               | YAML directory                   |
| `MCP_GATEWAY_REGISTRY`         | `<config>/registry.yaml` | Hand-written registry path — **absolute when set** (not joined with `CONFIG_DIR`); leave unset to use the default |
| `MCP_GATEWAY_REGISTRY_GENERATED` | `<config>/registry.generated.yaml` | Provisioner-generated registry — absolute when set |
| `MCP_GATEWAY_KEYS`             | `<config>/keys.yaml`     | API keys path — absolute when set |
| `MCP_GATEWAY_MCP_PATH`         | `/mcp`                   | MCP HTTP path                    |
| `MCP_GATEWAY_HOST`             | `0.0.0.0`                | Bind address                     |
| `MCP_GATEWAY_PORT`             | `8765`                   | Port                             |
| `TOOL_RAG_ENABLED`             | `1`                      | Enable Tool-RAG                  |
| `TOOL_RAG_EMBEDDER`            | `local`                  | `local` (sentence-transformers) or `url` (remote OpenAI-shaped API; `api` is a legacy alias) |
| `TOOL_RAG_EMBED_URL`           | —                        | Full remote embeddings endpoint, e.g. `http://ollama:11434/v1/embeddings` (not the base URL) |
| `TOOL_RAG_EMBED_MODEL`         | `text-embedding-3-small` | Model name sent to the embeddings API (set to your Ollama tag, e.g. `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0`) |
| `TOOL_RAG_EMBED_API_KEY`       | —                        | Bearer token for the embeddings API (optional; omit for keyless Ollama) |
| `TOOL_RAG_EMBED_DIM`           | —                        | Embedding dimension for the `url` embedder (e.g. `1024` for Qwen3-0.6B). If unset it is probed once at startup (requires the endpoint reachable at boot) |
| `TOOL_RAG_RERANKER`            | `off`                    | `off` or `local` — cross-encoder reranking of FAISS candidates |
| `TOOL_RAG_RERANKER_MODEL`      | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder model (small, multilingual, 14 languages) |
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

Point an MCP server at `http://gateway:8765/mcp` with a **non-admin** token.
Discovery is in-band: `list_tools()` exposes only `find_tools` + `run_tool`, and
after `find_tools` the gateway emits `tools/list_changed` so LibreChat picks up
and calls the discovered tools — no full catalog loaded. (The Deferred Tools flow
can also call `POST /tool-rag/retrieve` directly; it's policy-scoped the same way.)

### Generic MCP clients (in-band)

Connect to `/mcp` with a **non-admin** token. `list_tools()` returns the
`find_tools` + `run_tool` meta-tools (and the `initialize` instructions explain
them). The agent:

1. Calls `find_tools` with `{"query": "<what you want to do>"}` (optional `top_k`).
2. Reads the returned `results` — each has a `call_name`, `description`, and full
   `input_schema`.
3. Executes the chosen tool with `run_tool` (`{"call_name": "<server__tool>",
   "arguments": {...}}`) — or calls the `call_name` directly if the client allows
   unlisted names.

No out-of-band knowledge of `/tool-rag/*` is required — discovery is fully in the
MCP protocol. (Frameworks may still call `POST /tool-rag/retrieve` directly; it
is policy-scoped to the caller's key the same way `find_tools` is.)

---

## Project structure

```
provision.py            Manifest -> registry.generated.yaml + docker-compose.servers.yml
Makefile                provision / run / up convenience targets
gateway/
  app.py                Starlette app + routes + lifespan sync
  auth.py               API key -> AccessPolicy
  backends.py           open_upstream_session() (fresh session per request)
  server.py             merged MCP Server impl + policy enforcement + find_tools meta-tool
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
  reranker.py           optional cross-encoder rerank stage
  retriever.py          query -> top-K pipeline
  router.py             /tool-rag/* route handlers
servers/
  echo/                 reference stdio server (manifest.yaml + server.py)
  MANIFEST.example.yaml  manifest reference for all three kinds
config/
  registry.yaml, keys.yaml   (+ .example. variants)
```

---

## Roadmap / future considerations

**Two-phase (lazy) tool discovery.** Today both `find_tools` and `retrieve`
return full schemas eagerly (see [Design notes & caveats](#design-notes--caveats)).
A lazy flow would compress context further at catalog scale:

- **Optional schemas on `retrieve`** — e.g. `include_schema=false` to return a
  cheap names + description shortlist for discovery.
- **A per-tool `describe` / `get-schema` endpoint** — fetch the exact
  `input_schema` on demand, only for the tool the agent actually chose.

This is the original design intent (discover cheaply → fetch schema → call), and
it maximises context savings. The trade-off:

| | Context savings | Call reliability |
|---|---|---|
| **Eager (current)** | weaker — pays for unused schemas | strong — schema always in context |
| **Lazy (two-phase)** | strong at scale / high selectivity | reliable **only** if the loop enforces fetch-before-call |

**Requirement before switching the default:** lazy mode is only as reliable as
the agent loop's enforcement that a tool's schema is fetched *before* it may be
called (the way Deferred-Tools / ToolSearch gating works). Until that guardrail
exists in the client integration, **eager stays the default**; the optional
`include_schema=false` + `describe` endpoint can ship first as an opt-in.

---

## License

Internal homelab use.
