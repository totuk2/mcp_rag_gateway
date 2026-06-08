---
name: add-mcp
description: >-
  Add an upstream MCP server to this gateway from a single pointer — a running
  MCP URL, a git repo, or a local source tree. Use when the user says things
  like "add the MCP server from <url/repo>", "wire up <repo> as an MCP server",
  "/add-mcp <pointer>", or otherwise wants a new upstream registered, provisioned,
  key-granted, and verified end to end.
---

# Add an MCP server

Register a new upstream behind the gateway, end to end: classify → fetch →
manifest → secrets → provision → grant → restart → verify. The canonical prose
is the **Agent playbook** in `README.md`; this skill is the executable version
with the gotchas baked in. Read `CLAUDE.md` and the README playbook if anything
here is ambiguous.

The user's argument is the **pointer**: a running MCP URL, a git repo URL, or a
local path to server code. If no pointer was given, ask for one before starting.

## Step 1 — Pick an `id`

Lowercase, must match `^[A-Za-z0-9._-]+$`, and **must not contain `__`** (it's the
tool-namespace separator). Derive it from the repo/domain (e.g. `context7`,
`example-weather`). Confirm it's not already in `servers/`, `config/registry.yaml`,
or `config/registry.generated.yaml`.

## Step 2 — Classify the source → `kind`

| The pointer is… | `kind` |
|---|---|
| A URL that is already a **running** MCP endpoint (ends in `/mcp` or `/sse`, or user says "remote/hosted") | `remote` |
| A git repo / source tree with a **`Dockerfile`** | `docker` |
| A git repo / source tree that runs as a local **process** (no Dockerfile) | `stdio` |

If given a bare domain with no path, assume `remote` and try `/mcp` (Streamable
HTTP) first, then `/sse`. If neither responds, ask the user for the exact MCP URL.

## Step 3 — Fetch the source (skip for `remote`)

```bash
git clone <repo-url> servers/<id>        # or copy the source tree into servers/<id>/
```

Then **read the cloned repo's README** to find its exact run command and which
transport it speaks (stdio vs streamable_http vs sse) and which port it binds.
MCP servers differ — do not guess. Use what you find to fill the manifest below.

## Step 4 — Handle secrets BEFORE writing the manifest

`servers/<id>/manifest.yaml` **is committed to git** — never put a raw API key in
it. Where the secret goes depends on `kind`:

- **`remote` needing an auth header** → do **not** use a manifest. Hand-write the
  entry in `config/registry.yaml` (gitignored), putting the key in `headers:`.
  Remote provisioning only registers a URL, so a manifest adds nothing here.
  ```yaml
  # config/registry.yaml  (gitignored)
  servers:
    <id>:
      transport: streamable_http        # or sse
      url: "https://mcp.example.com/mcp"
      headers:
        Authorization: "Bearer ${ ... }"   # or the server's documented header name
  ```
- **`docker` needing env secrets** → put the secret in the root `.env` (gitignored)
  and reference it from the manifest as `${VAR}` — `docker compose up` interpolates
  it, so it never lands in the committed manifest:
  ```yaml
  # servers/<id>/manifest.yaml  (committed)
  env:
    API_KEY: "${MY_SERVER_KEY}"   # value lives in root .env
  ```
- **`stdio` env** → values are literal (no compose, no interpolation). Avoid real
  secrets in the committed manifest; if unavoidable, document it with the user.

Add the actual secret to `.env` (docker) or `config/registry.yaml` (remote) and
remind the user it's there.

## Step 5 — Write `servers/<id>/manifest.yaml` (skip for secret-bearing remote)

Use `servers/MANIFEST.example.yaml` as the field reference. Templates:

```yaml
# docker — repo with a Dockerfile
id: <id>
kind: docker
build: .                          # context (dir with the Dockerfile)
port: 9000                        # port the server listens on inside the container
transport: streamable_http        # or: sse
path: /mcp                        # default /mcp (or /sse for sse)
# command: ["serve", "--port", "9000"]   # optional, overrides image CMD
# env: { LOG_LEVEL: info, API_KEY: "${MY_SERVER_KEY}" }
```

```yaml
# stdio — repo that runs as a local process
id: <id>
kind: stdio
setup: ["pip install -r requirements.txt"]   # one-time; re-runs only if this changes
command: ["python", "server.py"]             # exact run command from the repo README
# env: { LOG_LEVEL: info }
```

```yaml
# remote WITHOUT a secret — fine as a manifest
id: <id>
kind: remote
transport: streamable_http        # or: sse
url: "https://mcp.example.com/mcp"
```

## Step 6 — Provision

```bash
python provision.py               # add --host if the gateway runs on the host, not in compose
```

For `docker` kind the image is built later by `docker compose up --build`, not by
the provisioner. Generated files (`config/registry.generated.yaml`,
`docker-compose.servers.yml`) are gitignored — never hand-edit them.

## Step 7 — Grant a key

Add `<id>: {}` under the `servers:` of each key in `config/keys.yaml` that should
reach it. **Grant every key that needs it — this is the most common miss.** In
particular, if the gateway runs with `GATEWAY_ANON_KEY=<id>` (the unauthenticated
LibreChat path), grant the **anon** key too, or the new server is invisible to
that client:

```yaml
- id: anon
  servers:
    <existing>: {}
    <id>: {}        # ← add this
```

Optionally scope with `tool_prefixes` / `uri_prefixes` / `prompt_prefixes`.

## Step 8 — Restart and verify

Restart the gateway so it re-reads `keys.yaml`/registry and Tool-RAG resyncs +
reindexes (`python -m gateway`, or the relevant `docker compose up`). Then:

```bash
# something the server does:
curl -s -H "Authorization: Bearer <key>" \
  -X POST localhost:8765/tool-rag/retrieve -d '{"query":"<task>"}'
curl -s -H "Authorization: Bearer <key>" localhost:8765/tool-rag/metrics   # tools_in_db should rise
```

The retrieve should return the new server's `<id>__*` tools. If a `find_tools`
call returns only other servers' tools, the calling key isn't granted `<id>` —
go back to Step 7 (check the anon key). If a restart trips a LibreChat "circuit
breaker is open," that's the client's cooldown after the gateway dropped — wait
it out or reconnect the MCP connection.

## Gotchas (learned the hard way)

- Manifests are committed; `config/registry.yaml`, `config/keys.yaml`, `.env`,
  and the generated files are gitignored. Keep secrets only in the gitignored set.
- `${VAR}` interpolation works for **docker** servers only (via compose), not stdio.
- `POST /tool-rag/reindex` rebuilds from the local DB only — it does **not**
  re-query upstreams. To pick up a server's tools you must restart (or set
  `TOOL_RAG_RESYNC_INTERVAL > 0`).
- Removing a server: delete `servers/<id>/` (or its manifest) or its
  `registry.yaml` block, drop its `keys.yaml` grants, re-provision, restart — the
  startup sync reconciles and prunes its tools. For docker also
  `docker image rm mcp-server-<id>:latest`.
