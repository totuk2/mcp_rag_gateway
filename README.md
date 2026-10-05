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
- **Schemas are eager by default, lazy on request.** By default `find_tools` and
  `/tool-rag/retrieve` attach each matched tool's full `input_schema` in the same
  response — favouring **call reliability** (the exact schema is in context when
  the model builds arguments). Pass `include_schema=false` for a lighter
  names+description shortlist and fetch a chosen tool's schema on demand via
  `describe_tool` / `GET /tool-rag/tool/<id>` (see [Meta-tools](#meta-tools-in-band-orchestration)).
- **Context savings scale with selectivity.** In eager mode each returned tool
  carries its full schema, so a broad retrieve can still be heavy — retrieve many,
  call few. Lazy mode (`include_schema=false` → `describe_tool`) compresses the
  discovery step further at catalog scale, at the cost of an extra round-trip.

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

The tool DB, FAISS index and derived tool catalog live in the `tool-rag-data`
volume (`/app/data`), so rebuilds keep them and an unchanged catalog costs no LLM
calls. All of it is derived from the upstreams and rebuilt automatically; to start
from scratch: `docker compose down && docker volume rm <project>_tool-rag-data`.

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
# volumes: [data:/data]           # optional named volumes (`<id>-data`), survive `up --build`
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
compose build` bakes the model into the image (no runtime HF fetch);
the `hf-cache` volume otherwise downloads it lazily on first use and persists it.

The model runs in a separate worker process (`tool_rag/rerank_worker.py`), not in
the gateway: torch + the model cost ~900MB RSS, and only a process exit returns
all of it. The worker starts on the first query that needs it (~6s) and exits
after `TOOL_RAG_RERANKER_IDLE_SECS` without a request (default 300; `0` = keep it
running), so an idle gateway sits at ~100MB. With `TOOL_RAG_RERANKER_COLD_START=skip`
(default) a query that finds the worker cold is ranked by the embedder alone while
the worker starts in the background — so the first query after an idle period can
rank differently from later ones; `wait` makes it block until reranked instead.
A crashed or hung worker degrades to embedder-only ranking and is restarted on
next use.

### Meta-tools (in-band orchestration)

Non-admin keys see a small set of meta-tools instead of the full catalog. All are
policy-scoped and self-hosted (no third party in the loop):

- **`find_tools`** `{query, top_k?, include_schema?}` — semantic discovery. Returns
  matching tools with `call_name` (+ `input_schema` unless `include_schema=false`).
- **`run_tool`** `{call_name, arguments}` — execute one discovered tool.
- **`run_tools`** `{calls: [{call_name, arguments, id?}], max_concurrency?}` — execute
  several **in parallel** in one call. Per-call error isolation (one failure doesn't
  abort the batch); concurrency bounded by `TOOL_RAG_MAX_PARALLEL`.
- **`describe_tool`** `{call_name}` — fetch one tool's full `input_schema` on demand
  (the second phase of lazy discovery). Also registers the tool for strict clients
  via `tools/list_changed`.
- **`browse_tools`** `{category? | query?}` — general questions about what's available,
  without loading tools (see [Tool catalog](#tool-catalog-browse_tools)): no arguments →
  overview of domains and categories with counts; `category` → one category or domain;
  `query` → "do I have tools for X?" verdict `strong`/`weak`/`none`.
- **`get_skill`** `{name}` — fetch a playbook (see [Skills](#skills-playbooks-and-delegate));
  listed when the key can see at least one skill.
- **`get_site_recipes`** `{query}` / **`save_site_recipe`** `{site, task, url_template?, steps?, …}`
  — shared memory of how to query websites by browser automation (see
  [Site recipes](#site-recipes-browser-memory)); `save_site_recipe` is listed only for keys
  granted a browser server.
- **`delegate`** `{task, skill?, max_steps?}` — hand a whole multi-step task to a
  gateway-side sub-agent; *listed when `TOOL_RAG_AGENT=on` and a planner is configured.*
- **`plan`** `{query, top_k?}` — *only listed when `TOOL_RAG_PLANNER=llm`.* Discovers
  candidates and asks a configurable own/OpenAI-shaped LLM (e.g. your Ollama) for a
  structured multi-step plan: `{steps: [{id, call_name, arguments_hint, depends_on,
  group, tool_type}], notes, missing}`. **Advisory only** — it never executes; the
  client fills concrete arguments and runs each `group` via `run_tools`.

**Two-phase (lazy) schema.** Default is eager (`find_tools` returns full schemas) for
call reliability. For maximum context savings, call `find_tools(..., include_schema=false)`
for a names+descriptions shortlist, then `describe_tool(call_name)` (or
`GET /tool-rag/tool/<id>`) for the schema of the tool you actually chose.

**Two-model topology (planner).** The client model (LibreChat's) drives the loop,
fills arguments, and sequences calls; the planner model (gateway-side) only produces
the plan. The planner is a bounded JSON task — a 7B–14B instruct model with JSON mode
suffices, and should be ≥ the client model's planning ability. Enable `plan` when the
client model is the weak link; a strong client plans fine from `find_tools` alone.

### Tool catalog (`browse_tools`)

The agent can't see the catalog, so `browse_tools` lets it ask general questions
first ("what kinds of tools do I have?", "do I have anything for X?"). A compact
per-key overview (~150 tokens: domains with descriptions, categories with counts)
is also embedded in the `browse_tools` description, and the domain names in
`find_tools`'s, so the agent knows what to expect before it asks anything.

- **Domains** are the upstream servers, each with a one-sentence description.
- **Categories** are a **dynamic taxonomy**: the planner LLM derives them from the
  current tool set and tags every tool with 1–3 of them (`tool_rag/enrichment.py`).
  They are recomputed whenever a server or tool is added, removed, or changes its
  description (fingerprint check after every sync — startup, `TOOL_RAG_RESYNC_INTERVAL`,
  or `POST /tool-rag/catalog/refresh`). An unchanged catalog costs no LLM calls.
  Existing category names stay stable (previous names are fed back, and a category
  still carried by a tool is never dropped); only tools whose content changed are
  re-tagged — everything only when a new category appears. Tags are also added to
  each tool's embedding text.
- **Hand-written metadata wins.** Optional manifest `description:` and `categories:`
  (see `servers/MANIFEST.example.yaml`) override the LLM; manifest categories are
  always present and always applied to that server's tools. Without a planner the
  catalog falls back to manifest categories + domains, with descriptions taken from
  the upstream's own `initialize` instructions.
- **Coverage verdicts** (`query`) are judged by the planner LLM over the catalog and
  the top search matches, returning the covering domains/categories and a
  `suggested_query` for `find_tools`. Generic tools (browser automation, code
  execution) that could only do something "by hand" yield at most `weak`. Without a
  planner the verdict falls back to score thresholds, which are less reliable for
  oblique or non-English questions.
- **Scope:** only tools the key can see (granted servers, `tool_prefixes`, active,
  server up). Non-admin keys get summaries only — never a per-category tool list —
  unless `TOOL_RAG_CATALOG_LIST_TOOLS=1`; admin keys get the list.
- **Failures are retried:** an LLM error (e.g. a rate limit) applies a partial
  result and retries in the background after 1, 2 and 4 minutes.

### Skills (playbooks) and `delegate`

Tools say *what* is possible; a **skill** says *how* to tackle a class of task: which
tools in which order, what to try when the first search finds nothing, when to stop.
Skills live in `skills/<name>/SKILL.md` (committed, baked into the image):

```markdown
---
name: hard-flight-routing
description: when to use it (also used to match queries to the skill)
requires_servers: [flight-research]   # visible only to keys granted all of these
---
<procedure in markdown>
```

They surface in-band: `find_tools` adds `related_skill` (+ "call get_skill") when the
query matches a skill's description (embedding similarity ≥ `SKILLS_MATCH_THRESHOLD`,
default 0.5), `browse_tools` lists them and adds `related_skill` to coverage verdicts,
and `get_skill` returns the body. Skills are re-read on restart.

**`delegate`** runs a task end to end on the gateway: a tool-calling loop on the planner
model (`tool_rag/agent.py`) with `find_tools` / `describe_tool` / `run_tool` / `get_skill`,
seeded with the given or best-matching skill. Every upstream call runs inside the
caller's request, so the calling key's policy applies exactly as for direct calls; the
sub-agent can't call `delegate` again. Bounded by `max_steps` (default 15, cap 30) and
`TOOL_RAG_AGENT_TIMEOUT` (default 240 s); sends MCP progress notifications when the
client passes a `progressToken`. Returns `{answer, steps, stopped_reason, skill}`.
Clients must allow long tool calls — in LibreChat set the MCP server's `timeout`
(e.g. `300000`).

### Stateful upstreams (sticky sessions)

The gateway opens a fresh upstream session per tool call, which loses per-session state —
e.g. Playwright's open page (`browser_navigate` then `browser_snapshot` → `about:blank`).
Mark such a server `stateful: true` in its manifest (or `registry.yaml`): the gateway then
keeps **one upstream session per downstream MCP session** and routes that session's calls
to it, one at a time. Sticky sessions close after `STICKY_SESSION_IDLE_SECS` (600) idle,
when the client session ends, beyond `STICKY_MAX_SESSIONS` (20, LRU), on upstream failure
(one retry on a fresh session) and on shutdown. `playwright` is stateful; it keeps
`--shared-browser-context`, so concurrent client sessions share one browser — fine for a
single-user homelab. Clients that open a new MCP session per call (e.g. one-shot curl) get
no stickiness.

### Long calls: background jobs

MCP clients give up on a tool call after a fixed time (LibreChat: `timeout`, default
30 s), while `research_route` takes up to ~75 s and `delegate` minutes. Every tool call
therefore answers within `TOOL_CALL_SYNC_SECS` (default 25): with the result if done,
otherwise with `{"status": "running", "job_id": …}` while the call keeps running in the
background (`gateway/jobs.py`). The `get_job_result` meta-tool waits up to ~20 s and
returns the original result, or `running` again. Jobs belong to the API key that started
them, are kept 30 min after completion (dropped when fetched) and are lost on restart.

### Response projection and result handles

`run_tool` and `run_tools` items accept `fields` — JSON paths such as `"query"`,
`"itineraries[].price"`, `"items[0].title"` — and return only those parts of a JSON
result. Whenever the gateway shortens a result (a projection, a compact flight digest, or
any text over `RESULT_MAX_CHARS`, default 40000), it keeps the full text for
`RESULT_TTL_S` (30 min) under a `result_id`; `get_result {result_id, fields?, grep?,
offset?, limit?}` reads it back projected, grepped (regex over lines; JSON is
pretty-printed first) or in character windows — instead of re-running the call. Results
belong to the key that produced them and are in memory only (`gateway/results.py`).

### Hot reload (no restarts)

A gateway restart drops every client's MCP session (LibreChat then reports "Connection
closed" until it reconnects). Instead, every `CONFIG_RELOAD_INTERVAL` seconds (10) the
gateway re-reads changed `keys.yaml`, `registry*.yaml` and `skills/*/SKILL.md` and applies
them in place: new keys work immediately; a registry change re-syncs upstreams (adding,
updating and pruning tools), refreshes the catalog, reindexes and drops sticky sessions of
changed servers. A file that fails to parse is logged and the previous config stays. Force
it with `POST /tool-rag/reload` (admin). So: add a server = provision + `docker compose up -d
--no-deps <server>`; the gateway picks it up by itself.

### Tool reviews (quarantine for external servers)

Upstream tool descriptions go straight into models' context, so a server that changes them
can inject instructions ("tool poisoning" / "rug pull"). For servers with `review_changes`
— by default any URL outside the LAN (not a compose name, `*.lan`/`*.local` or private IP;
set it explicitly in a manifest or `registry.yaml` to override) — the first sync is trusted,
and afterwards:
- a **new tool** is quarantined: hidden from search and catalog, calls blocked;
- a **changed** name/description/input schema keeps the **approved** version in use (what
  models see) until approved.

`GET /tool-rag/reviews` lists pending changes with the approved and proposed text;
`POST /tool-rag/reviews/approve` or `/reject` with `{"tool_ids": [...]}` or `{"all": true}`
(admin only). Rejected changes stay rejected until the upstream changes again; an upstream
reverting to the approved text clears its review. `/tool-rag/metrics` shows `pending_reviews`.

### Metrics

`GET /tool-rag/metrics/tools` — per tool: calls, errors, avg/p50/p95/max latency, background
jobs, and for upstream calls the result size before (`raw_chars`) and after shaping
(`sent_chars`); admin keys also get calls per key. `GET /tool-rag/metrics/prometheus` — the
same as Prometheus counters (Bearer-authed like the rest of `/tool-rag`).

### Compact flight-search results

Fare engines return huge payloads (a round-trip search: Kiwi ~54 KB, Google Flights
~120 KB, Duffel ~63 KB — plus the same again as `structuredContent`). For servers flagged
`compact_results: true` (`kiwi`, `google-flights`, `flights`, `skiplagged`) the gateway
rewrites search results into a digest (`gateway/compactors.py`): the first
`COMPACT_MAX_OPTIONS` (10) options in the engine's order, one line each — price, route,
times, stops, carriers + flight numbers, return leg, separate-ticket/self-transfer flags,
bags, booking link — plus the total count. That's 2–4 KB instead of 30–120 KB. Pass
`"full": true` to `run_tool` (or a `run_tools` item) for the raw result. Unparseable
results pass through unchanged. Independently, `run_tools` no longer returns a tool's
`structuredContent` when it has text content (it was a mirror, doubling the size).

### Site recipes (browser memory)

When an agent manages to get data from a website by browser automation (e.g. an airline's
flight search via Playwright), it saves **how**: `save_site_recipe` with the site, a task
("search one-way flights"), a `url_template` with placeholders (`{origin}`,
`{date:YYYY-MM-DD}`) and/or short `steps` naming the labels/buttons used, plus `notes`.
Next time, the recipe reaches the agent's context without it having to ask: after any
successful tool call whose arguments contain a URL (e.g. `browser_navigate`), the gateway
appends that site's recipes to the result (once per MCP session per site);
`get_site_recipes` looks them up by domain, URL or name. Agents report outcomes
(`recipe_id` + `worked`); a re-save of the same site+task replaces the steps.

Agents are nudged and, in `delegate`, required to record what they learned: the first
successful visit to a site without a recipe appends a reminder to save one, and a
`delegate` run that browsed such a site gets one forced `save_site_recipe` turn before it
finishes (steps are exact tool calls with key arguments, so they replay as-is). Recipes for
domains named in a `delegate` task go into its prompt up front.

Stored in the gateway DB (`site_recipes`, persistent `tool-rag-data` volume) and **shared
across keys**, with guards because recipe text partly originates from web pages and is
injected into other agents' context: only keys granted a server from
`RECIPES_WRITER_SERVERS` (default `playwright`) or admin keys may write; size limits;
a `url_template` must stay on the recipe's own domain; author key, success/failure counts
and last-worked date are kept (failing recipes sink and are flagged `STALE`); injected
text is fenced and labelled as untrusted data. The `hard-flight-routing` skill and the
`delegate` sub-agent use and grow this memory.

### Flight research stack

`servers/flight-research` (our code) plus four fare engines, wired for hard routes
("Gdańsk → Aleppo"):

| Server | Kind | What | Key |
|---|---|---|---|
| `kiwi` | remote | Kiwi.com: self-transfer combos, different-airport connections, ±3 days | — |
| `skiplagged` | remote | Skiplagged: flights, fare calendars, hotels, cars | — |
| `google-flights` | docker | Google Flights via [fli](https://github.com/punitarani/fli) | — |
| `flights` | docker | Duffel (NDC) | `DUFFEL_API_KEY_LIVE` |
| `flight-research` | docker | `nearby_airports` (OurAirports), `airport_routes` / `airport_schedule` (AeroDataBox), `research_route`, `report_border_status` / `border_status` | `AERODATABOX_KEY` (optional) |

`research_route(origin, destination, date, …)` expands nearby airports, finds hubs that
fly into the destination area (AeroDataBox route stats; without a key, from the
engines' connection airports), queries all engines for the whole trip and for
origin→hub / hub→destination legs in parallel (~20-75 s budget), combines self-transfer
legs (≥3 h), and ranks by price + time + risk (self-transfer, stops, ground distance,
border crossing). It returns a one-line-per-option `summary`, full `options`, `hubs`,
`coverage` (what ran / failed) and `manual_checks` (last-leg airlines no engine priced —
check their websites). The `hard-flight-routing` skill drives it plus the fallbacks
(Playwright on airline sites, stopping at CAPTCHAs).

Ground legs use land neighbours from GeoNames (`countryInfo.txt`, plus `land_link`
rows in `flight_research/overrides.csv` for tunnels/bridges). Legs to or from an island
state aren't dropped; they're flagged "no road link" for the user to check the ferry.
Border closures are a **memory**, not a table: agents record what a web search found
with `report_border_status` (status, source URL, note; latest report wins), stored in
`/data/cache.db` (the `flight-research-data` volume). `research_route` ranks options
across a reported closed or restricted border down and lists them in `border_warnings`,
and names crossed borders with no report in `unverified_borders`.

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

Pass `"include_schema": false` to get a lighter shortlist without `input_schema`;
fetch a chosen tool's schema with `GET /tool-rag/tool/<tool_id>`.

#### `GET /tool-rag/tool/{tool_id}`

On-demand schema fetch (lazy two-phase). Returns `{tool_id, tool_name, server_name,
description, tool_type, input_schema, status}`. Policy-scoped: the tool's server must
be granted and visible to the key.

#### `POST /tool-rag/reindex`

Body `{"mode": "full"}` (rebuild) or `{"mode": "incremental"}` (dirty tools only).

#### `GET /tool-rag/catalog`
Same views as `browse_tools`: no parameters → overview; `?category=<name>`;
`?query=<text>` → coverage verdict. Policy-scoped like `retrieve`.

#### `POST /tool-rag/catalog/refresh`
Admin key only. Forces a full catalog recompute (taxonomy, tags, descriptions —
LLM calls) and an incremental reindex if tags changed.

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
| `TOOL_RAG_EMBED_QUERY_INSTRUCTION` | —                    | Asymmetric-model query prefix: queries are wrapped `Instruct: <this>\nQuery: <q>` (documents embedded raw). Set for instruction-tuned embedders like Qwen3-Embedding; leave empty for symmetric models (MiniLM). Query-side only — no reindex |
| `TOOL_RAG_RERANKER`            | `off`                    | `off` or `local` — cross-encoder reranking of FAISS candidates |
| `TOOL_RAG_RERANKER_MODEL`      | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder model (small, multilingual, 14 languages) |
| `TOOL_RAG_RERANKER_IDLE_SECS`  | `300`                    | Stop the reranker worker process after this many idle seconds (frees ~900MB); `0` = never |
| `TOOL_RAG_RERANKER_COLD_START` | `skip`                   | `skip` = rank with the embedder alone while the worker starts; `wait` = block until reranked |
| `TOOL_RAG_MAX_PARALLEL`        | `8`                      | Cap on concurrent `run_tools` upstream calls |
| `TOOL_RAG_PLANNER`             | `off`                    | `off` or `llm` — enable the LLM-backed `plan` meta-tool |
| `TOOL_RAG_PLANNER_URL`         | —                        | Chat-completions endpoint for the planner (e.g. `http://ollama:11434/v1/chat/completions`) |
| `TOOL_RAG_PLANNER_MODEL`       | —                        | Planner model (e.g. `qwen2.5:14b-instruct`) |
| `TOOL_RAG_PLANNER_API_KEY`     | —                        | Optional bearer for the planner endpoint |
| `TOOL_RAG_PLANNER_TEMPERATURE` | `0.1`                    | Planner sampling temperature |
| `TOOL_RAG_DB`                  | `tool_registry.db`       | SQLite path; the FAISS index is stored next to it. Pinned to `/app/data/tool_registry.db` (the `tool-rag-data` volume) under compose |
| `TOOL_RAG_AGENT`               | `off`                    | `on` = list the `delegate` sub-agent meta-tool (needs `TOOL_RAG_PLANNER=llm` with a tool-calling model) |
| `TOOL_RAG_AGENT_TIMEOUT`       | `240`                    | Seconds budget for one `delegate` run |
| `TOOL_CALL_SYNC_SECS`          | `25`                     | Tool calls slower than this return a `job_id`; fetch with `get_job_result` |
| `RESULT_MAX_CHARS`             | `40000`                  | Longer tool results are truncated with a `result_id` for `get_result` |
| `RESULT_TTL_S`                 | `1800`                   | How long stored full results stay readable via `get_result` |
| `CONFIG_RELOAD_INTERVAL`       | `10`                     | Seconds between hot-reload checks of keys/registry/skills; `0` = off |
| `COMPACT_MAX_OPTIONS`          | `10`                     | Options kept per compacted flight-search result (`compact_results` servers) |
| `STICKY_SESSION_IDLE_SECS`     | `600`                    | Close a sticky upstream session (stateful servers) after this idle time |
| `STICKY_MAX_SESSIONS`          | `20`                     | Max open sticky upstream sessions (least recently used closed first) |
| `RECIPES_ENABLED`              | `1`                      | Site recipes (browser memory) on/off |
| `RECIPES_WRITER_SERVERS`       | `playwright`             | Comma-separated servers; keys granted any of them (or admin) may save recipes |
| `SKILLS_DIR`                   | `<repo>/skills`          | Skills (playbooks) directory |
| `SKILLS_MATCH_THRESHOLD`       | `0.5`                    | Cosine similarity for matching a query to a skill (`related_skill`) |
| `TOOL_RAG_CATALOG_LIST_TOOLS`  | `0`                      | `1` = `browse_tools` `category` also lists tools (≤25) for non-admin keys; default summaries only |
| `TOOL_RAG_CATALOG_REFRESH_TIMEOUT` | `90`                 | Seconds budget for a catalog refresh's LLM calls; on timeout the previous catalog stays |
| `TOOL_RAG_CATALOG_STRONG` / `_WEAK` | `0.45` / `0.3`      | Fallback coverage thresholds (no planner) on reranked scores |
| `TOOL_RAG_CATALOG_STRONG_NORERANK` / `_WEAK_NORERANK` | `0.85` / `0.7` | Same, when the reranker is off |
| `TOOL_RAG_WITHOUT_AUTH`        | `0`                      | Skip auth for `/tool-rag/`       |
| `TOOL_RAG_STARTUP_REINDEX`     | `full`                   | `full` \| `incremental` \| `off` — index strategy at startup |
| `TOOL_RAG_HEALTHCHECK_INTERVAL`| `30`                     | Seconds between upstream liveness probes; `0` disables |
| `TOOL_RAG_HEALTHCHECK_TIMEOUT` | `5`                      | Per-probe connect timeout (seconds) |
| `TOOL_RAG_RESYNC_INTERVAL`     | `0`                      | Seconds between background re-pull of upstream tool lists (and catalog refresh); `0` = off. E.g. `600` to follow tool changes on running upstreams |
| `TOOL_RAG_SHORTLIST_DESC_CHARS`| `300`                    | Cap (chars) on the teaser description in the `find_tools` shortlist; `0` = full text. Full description always available via `describe_tool` and used for embedding (no reindex) |
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
  planner.py            optional LLM-backed plan meta-tool (off by default)
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

**Optional Rube hybrid fallback (privacy trade-off).** Composio's hosted
[Rube](https://rube.app) is itself an MCP server, so you can register it as a
`remote` upstream (`kind: remote, url: https://rube.app/mcp, headers: {Authorization:
"${RUBE_TOKEN}"}`) and its tools surface through `find_tools` like any other server —
a "private-first, fall back to Rube for SaaS apps you haven't self-integrated" hybrid.
**Caveat:** that traffic goes to Composio and Composio brokers the OAuth, which cuts
against this gateway's self-hosting/traffic-ownership goal — so it's opt-in, not a
default. A future enhancement could auto-suggest Rube's search tool only when local
retrieval scores fall below a threshold (`TOOL_RAG_RUBE_FALLBACK`, off by default).

**Two-phase (lazy) tool discovery — shipped as opt-in.** Lazy discovery now exists:
`find_tools(..., include_schema=false)` (or `POST /tool-rag/retrieve` with the same
flag) returns a cheap names+description shortlist, and `describe_tool` /
`GET /tool-rag/tool/<id>` fetch the exact `input_schema` on demand. The trade-off:

| | Context savings | Call reliability |
|---|---|---|
| **Eager (default)** | weaker — pays for unused schemas | strong — schema always in context |
| **Lazy (opt-in)** | strong at scale / high selectivity | reliable **only** if the loop enforces fetch-before-call |

**Eager stays the default** because lazy mode is only as reliable as the agent loop's
enforcement that a schema is fetched *before* the tool is called (the way
Deferred-Tools / ToolSearch gating works). Flip to lazy per call when your client
enforces that.

---

## License

Internal homelab use.
