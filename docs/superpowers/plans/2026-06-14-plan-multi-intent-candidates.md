# Plan Multi-Intent Candidate Gathering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `plan` meta-tool decompose a compound task into sub-intents, retrieve tools per intent (plus the full query), and union the candidates — so it stops dropping minority-intent tools (measured: `playwright__browser_take_screenshot` ranked #20 and was omitted).

**Architecture:** Add `Planner.decompose()` (one cheap LLM call returning search phrases). In `gateway/server.py`, add a pure `_union_candidates()` merger and a `_gather_candidates_multi()` orchestrator that fans the existing per-query `_gather_candidates()` across `[full query] + sub-intents` concurrently, unions by `call_name` (max score), and caps. `handle_plan` calls the multi-gatherer; everything else is unchanged. Spec: `docs/superpowers/specs/2026-06-14-plan-multi-intent-candidates-design.md`.

**Tech Stack:** Python 3 (async), httpx (planner LLM calls), the existing FAISS/sentence-transformers retriever, MCP server. No pytest in this repo — logic checks run as standalone `assert` scripts inside the gateway Docker image; integration is verified live over HTTP.

---

## Conventions for this plan

- **Commit messages:** short, max 7 words, prefixed `[IMP]`/`[FIX]`/`[REF]`/`[DEL]`, **no** co-author trailer. Commits are SSH-signed with a passphrase the agent can't supply — commit with `git -c commit.gpgsign=false ...`.
- **Running logic checks (no pytest, laptop `.venv` is broken):** run inside the gateway image, which has `mcp`/`httpx` installed. Pattern used below:
  ```bash
  docker compose run --rm --no-deps --entrypoint python provision /app/<script>.py
  ```
  The `provision` service (in `docker-compose.yml`) builds from the gateway image and mounts the repo at `/app`, so freshly-edited files and check scripts are visible without rebuilding.
- **Integration checks** run from this host against the live gateway at `http://tower.lan:8765` after the user redeploys on tower (gateway restart only — these are pure-Python changes, no image rebuild needed for upstreams).

---

## File structure

- **Modify `tool_rag/planner.py`** — add `_DECOMPOSE_SYSTEM_PROMPT`; refactor the chat HTTP POST out of `plan()` into a shared `_chat(messages)`; add abstract `decompose()` to `Planner` and implement it on `LlmPlanner`. (Light deps: stdlib + lazy httpx — importable without mcp/faiss.)
- **Modify `gateway/server.py`** — add three `DEFAULT_PLAN_*` constants; add pure `_union_candidates()`; add `_gather_candidates_multi()` inside `build_gateway_server` (closure over `retriever`/`planner`/`_gather_candidates`); point `handle_plan` at it and raise the `top_k` default to 30; update the `plan` tool-definition docstring for `top_k`.
- **Modify `.env.example`** — document the two new env knobs.

---

## Task 1: Add `decompose()` to the planner

**Files:**
- Modify: `tool_rag/planner.py`
- Check script: `_check_decompose.py` (repo root, temporary — deleted in Step 7)

- [ ] **Step 1: Write the failing check**

Create `_check_decompose.py` at the repo root:

```python
"""Logic check for LlmPlanner.decompose parsing (no network: _chat is stubbed)."""
import asyncio
from tool_rag.planner import LlmPlanner


def run():
    p = LlmPlanner(url="http://x", model="m")

    async def fake_chat(messages):
        # Simulate the model returning JSON with junk to be filtered.
        return '{"intents": ["search arxiv for papers", "  ", 5, "take a screenshot"]}'

    p._chat = fake_chat  # type: ignore[assignment]
    out = asyncio.get_event_loop().run_until_complete(p.decompose("whatever"))
    assert out == ["search arxiv for papers", "take a screenshot"], out

    async def bad_chat(messages):
        return "not json at all"

    p._chat = bad_chat  # type: ignore[assignment]
    out2 = asyncio.get_event_loop().run_until_complete(p.decompose("whatever"))
    assert out2 == [], out2  # tolerant: bad JSON -> empty list, no raise

    print("OK: decompose parsing")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
docker compose run --rm --no-deps --entrypoint python provision /app/_check_decompose.py
```
Expected: FAIL — `AttributeError`/`TypeError` because `decompose` and `_chat` don't exist yet (`LlmPlanner` has no `decompose`; `Planner` ABC also lacks it).

- [ ] **Step 3: Add the decomposition prompt**

In `tool_rag/planner.py`, immediately after the `_SYSTEM_PROMPT` block (after line 38), add:

```python
_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a user TASK into short, independent search phrases — one per distinct "
    "capability or tool the task needs (e.g. 'search arxiv for papers', 'take a "
    "screenshot of a web page'). Each phrase is a standalone query used to look up the "
    "right tool, so name the action and object, not the whole task. Respond with ONLY "
    'a JSON object: {"intents": ["phrase", ...]}. Use 1-6 phrases; fewer is fine for '
    "simple tasks. Do not invent capabilities the task does not mention."
)
```

- [ ] **Step 4: Refactor the chat POST into a shared `_chat`**

In `LlmPlanner`, replace the body of `plan()` from the `import httpx` line through the `data = resp.json()` block with a call to a new helper. Concretely, change `plan()` (currently lines 86–118) so its HTTP section becomes:

```python
    async def plan(self, query: str, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tools = [_compact_tool(c) for c in candidates]
        valid_names = {t["call_name"] for t in tools}
        user = (
            f"TASK:\n{query}\n\nAVAILABLE TOOLS (JSON):\n{json.dumps(tools)}\n\n"
            "Return the plan JSON now."
        )
        content = await self._chat([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
        raw = _parse_json_object(content)
        return _normalize_plan(raw, valid_names, candidates)

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        """POST an OpenAI-shaped chat-completions request and return message content."""
        if not self._url:
            raise RuntimeError("LlmPlanner: TOOL_RAG_PLANNER_URL not configured")
        if not self._model:
            raise RuntimeError("LlmPlanner: TOOL_RAG_PLANNER_MODEL not configured")
        import httpx

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
```

(The `not self._url` / `not self._model` guards move into `_chat`, so they still run on every `plan()` call.)

- [ ] **Step 5: Add `decompose` to the ABC and `LlmPlanner`**

In the `Planner` ABC (after the `plan` abstractmethod, around line 61) add:

```python
    @abstractmethod
    async def decompose(self, query: str) -> list[str]:
        """Split a task into short per-capability search phrases (best-effort)."""
```

In `LlmPlanner`, add the implementation (e.g. right after `_chat`):

```python
    async def decompose(self, query: str) -> list[str]:
        """Return short per-capability search phrases for the task, or [] on any
        parse failure (caller falls back to single-query retrieval)."""
        content = await self._chat([
            {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"TASK:\n{query}\n\nReturn the intents JSON now."},
        ])
        try:
            raw = _parse_json_object(content)
        except RuntimeError:
            return []
        intents = raw.get("intents")
        if not isinstance(intents, list):
            return []
        return [s.strip() for s in intents if isinstance(s, str) and s.strip()]
```

- [ ] **Step 6: Run the check to verify it passes**

Run:
```bash
docker compose run --rm --no-deps --entrypoint python provision /app/_check_decompose.py
```
Expected: `OK: decompose parsing`

- [ ] **Step 7: Delete the temporary check and commit**

```bash
rm _check_decompose.py
git add tool_rag/planner.py
git -c commit.gpgsign=false commit -m "[IMP] add planner decompose for plan"
```

---

## Task 2: Add the pure `_union_candidates` merger

**Files:**
- Modify: `gateway/server.py`
- Check script: `_check_union.py` (repo root, temporary — deleted in Step 5)

- [ ] **Step 1: Write the failing check**

Create `_check_union.py` at the repo root:

```python
"""Logic check for _union_candidates: dedup by call_name (max score), sort, cap."""
from gateway.server import _union_candidates


def run():
    lists = [
        [{"call_name": "a", "score": 0.4}, {"call_name": "b", "score": 0.9}],
        [{"call_name": "a", "score": 0.8}, {"call_name": "c", "score": 0.2}],
        [{"call_name": None, "score": 1.0}],  # dropped: no call_name
    ]
    out = _union_candidates(lists, cap=10)
    names = [c["call_name"] for c in out]
    # dedup: a kept once; max score for a is 0.8; sort by score desc -> b(0.9),a(0.8),c(0.2)
    assert names == ["b", "a", "c"], names
    a = next(c for c in out if c["call_name"] == "a")
    assert a["score"] == 0.8, a
    # cap truncates by best score
    assert [c["call_name"] for c in _union_candidates(lists, cap=2)] == ["b", "a"]
    # deterministic tie-break on call_name when scores equal
    tie = _union_candidates([[{"call_name": "z", "score": 0.5}, {"call_name": "y", "score": 0.5}]], cap=10)
    assert [c["call_name"] for c in tie] == ["y", "z"], tie
    print("OK: union candidates")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
docker compose run --rm --no-deps --entrypoint python provision /app/_check_union.py
```
Expected: FAIL — `ImportError: cannot import name '_union_candidates'`.

- [ ] **Step 3: Add the constants and the pure merger**

In `gateway/server.py`, after the `DEFAULT_SHORTLIST_DESC_CHARS = 300` block (after line 69) add:

```python
# plan multi-intent candidate gathering (decompose -> retrieve per intent -> union).
# See docs/superpowers/specs/2026-06-14-plan-multi-intent-candidates-design.md.
DEFAULT_PLAN_MAX_INTENTS = 6        # max sub-intents kept from decomposition
DEFAULT_PLAN_PER_INTENT_K = 5       # retrieval depth per intent (and per base query)
DEFAULT_PLAN_TOP_K = 30             # default union cap (the plan `top_k` arg)


def _union_candidates(
    lists: list[list[dict[str, Any]]], cap: int
) -> list[dict[str, Any]]:
    """Merge per-intent candidate lists into one ranked list.

    Dedup by `call_name` keeping the entry with the max `score`; sort by score
    descending with a deterministic tie-break on `call_name`; truncate to `cap`
    (cap <= 0 means no truncation). Pure and deterministic — no I/O."""
    best: dict[str, dict[str, Any]] = {}
    for lst in lists:
        for c in lst:
            name = c.get("call_name")
            if not name:
                continue
            prev = best.get(name)
            if prev is None or (c.get("score") or 0) > (prev.get("score") or 0):
                best[name] = c
    merged = sorted(
        best.values(),
        key=lambda c: (-(c.get("score") or 0), c.get("call_name") or ""),
    )
    return merged[:cap] if cap > 0 else merged
```

- [ ] **Step 4: Run the check to verify it passes**

Run:
```bash
docker compose run --rm --no-deps --entrypoint python provision /app/_check_union.py
```
Expected: `OK: union candidates`

- [ ] **Step 5: Delete the temporary check and commit**

```bash
rm _check_union.py
git add gateway/server.py
git -c commit.gpgsign=false commit -m "[IMP] add union candidates merger"
```

---

## Task 3: Add `_gather_candidates_multi` and wire it into `handle_plan`

**Files:**
- Modify: `gateway/server.py` (inside `build_gateway_server`, near `_gather_candidates` at line 325 and `handle_plan` at line 610)

- [ ] **Step 1: Add `_gather_candidates_multi` after `_gather_candidates`**

In `gateway/server.py`, immediately after `_gather_candidates` returns (after line 357), add this nested function (same indentation level as `_gather_candidates`, so it closes over `planner` and `_gather_candidates`):

```python
    async def _gather_candidates_multi(query: str, top_k: int) -> list[dict[str, Any]]:
        """plan-only: decompose the task, retrieve per intent (+ the full query),
        and union the candidates. Falls back to single-query retrieval whenever
        decomposition yields nothing or every per-intent retrieval fails, so this
        never returns worse candidates than `_gather_candidates` alone."""
        assert planner is not None
        try:
            max_intents = int(os.environ.get("TOOL_RAG_PLAN_MAX_INTENTS", DEFAULT_PLAN_MAX_INTENTS))
        except (TypeError, ValueError):
            max_intents = DEFAULT_PLAN_MAX_INTENTS
        try:
            per_intent_k = int(os.environ.get("TOOL_RAG_PLAN_PER_INTENT_K", DEFAULT_PLAN_PER_INTENT_K))
        except (TypeError, ValueError):
            per_intent_k = DEFAULT_PLAN_PER_INTENT_K

        try:
            intents = await planner.decompose(query)
        except Exception:
            logger.debug("plan decompose failed; single-query candidates", exc_info=True)
            intents = []
        intents = [s for s in intents if isinstance(s, str) and s.strip()][:max_intents]
        if not intents:
            single, _ = await _gather_candidates(query, top_k)
            return single

        queries = [query] + intents  # full query always included as a base intent
        settled = await asyncio.gather(
            *[_gather_candidates(q, per_intent_k) for q in queries],
            return_exceptions=True,
        )
        lists: list[list[dict[str, Any]]] = []
        for s in settled:
            if isinstance(s, BaseException):
                logger.debug("per-intent retrieval failed", exc_info=s)
                continue
            cands, _fallback = s
            lists.append(cands)
        merged = _union_candidates(lists, top_k)
        if not merged:
            single, _ = await _gather_candidates(query, top_k)
            return single
        return merged
```

- [ ] **Step 2: Point `handle_plan` at the multi-gatherer and raise the default `top_k`**

In `handle_plan` (lines 619–625), change the `top_k` default and the candidate call. Replace:

```python
        try:
            top_k = int(args.get("top_k", 10))
        except (TypeError, ValueError):
            top_k = 10
        if not get_policy().servers:
            return _err_tool("No servers are granted to this API key.")
        candidates, _ = await _gather_candidates(query, top_k)
```

with:

```python
        try:
            top_k = int(args.get("top_k", DEFAULT_PLAN_TOP_K))
        except (TypeError, ValueError):
            top_k = DEFAULT_PLAN_TOP_K
        if not get_policy().servers:
            return _err_tool("No servers are granted to this API key.")
        candidates = await _gather_candidates_multi(query, top_k)
```

- [ ] **Step 3: Update the `plan` tool-definition docstring for `top_k`**

In `_plan_definition` (lines 304–308), update the `top_k` description so the advertised schema matches the new behaviour. Replace:

```python
                "top_k": {
                    "type": "integer",
                    "description": "Max candidate tools to consider (default 10).",
                    "default": 10,
                },
```

with:

```python
                "top_k": {
                    "type": "integer",
                    "description": "Max candidate tools (after per-intent union) to consider (default 30).",
                    "default": 30,
                },
```

- [ ] **Step 4: Verify the module imports cleanly**

Run:
```bash
docker compose run --rm --no-deps --entrypoint python provision -c "import gateway.server; print('import OK')"
```
Expected: `import OK` (no syntax/NameError; confirms the new closures and constants resolve).

- [ ] **Step 5: Commit**

```bash
git add gateway/server.py
git -c commit.gpgsign=false commit -m "[IMP] plan gathers multi-intent candidates"
```

---

## Task 4: Document the new env knobs

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add the knobs to `.env.example`**

Open `.env.example`, find the planner section (the lines beginning `TOOL_RAG_PLANNER`). Immediately after the last `TOOL_RAG_PLANNER_*` line, add:

```bash
# plan multi-intent candidate gathering (only affects the `plan` meta-tool):
# plan decomposes the task into sub-intents, retrieves PER_INTENT_K tools per
# intent (plus the full query), and unions/dedups them up to the plan `top_k`
# (default 30). Defaults are sensible; override only to tune coverage vs. latency.
# TOOL_RAG_PLAN_MAX_INTENTS=6
# TOOL_RAG_PLAN_PER_INTENT_K=5
```

(If `.env.example` does not contain a `TOOL_RAG_PLANNER` block, add the snippet under the `# ── Tool-RAG ──` section header instead.)

- [ ] **Step 2: Commit**

```bash
git add .env.example
git -c commit.gpgsign=false commit -m "[IMP] document plan multi-intent env knobs"
```

---

## Task 5: Live integration verification (deploy + regression + latency)

**Files:** none (verification only). This task confirms the spec's success criteria and settles the concurrency/threadsafety open question.

- [ ] **Step 1: Redeploy the gateway on tower**

These are pure-Python changes; only the gateway container needs to restart (no upstream rebuild). Ask the user to run on tower (SSH passphrase is interactive in their terminal):

```bash
cd /mnt/user/hp-victus/OtherProjects/mcp-servers/mcp_rag_gateway
docker compose -f docker-compose.yml -f docker-compose.servers.yml up -d --build mcp-gateway
```

Wait for the user to confirm it's back up (`curl -s tower.lan:8765/health` → `{"status":"ok"}`).

- [ ] **Step 2: Warm the reranker, then run the regression query**

From this host:

```bash
python - <<'PY'
import httpx, json, time
H={"Authorization":"Bearer dev-key-full-access"}
B="http://tower.lan:8765/mcp"
hdr={**H,"Content-Type":"application/json","Accept":"application/json, text/event-stream"}
c=httpx.Client(timeout=120.0)
def parse(r):
    if "text/event-stream" in r.headers.get("content-type",""):
        out=None
        for ln in r.text.splitlines():
            if ln.startswith("data:"): out=json.loads(ln[5:].strip())
        return out
    return r.json()
r=c.post(B,headers=hdr,json={"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"p","version":"0"}}})
sid=r.headers.get("mcp-session-id"); hdr["mcp-session-id"]=sid
c.post(B,headers=hdr,json={"jsonrpc":"2.0","method":"notifications/initialized","params":{}})
def call(name,args,i):
    return parse(c.post(B,headers=hdr,json={"jsonrpc":"2.0","id":i,"method":"tools/call","params":{"name":name,"arguments":args}}))
# warm the reranker
call("find_tools",{"query":"warm up"},2)
# regression: compound query must now include a screenshot tool
t=time.time()
res=call("plan",{"query":"Search arXiv for 5 recent papers about AI in MedTech business, navigate to abstract pages, take screenshots, extract title/authors/summary"},3)
dt=time.time()-t
payload=json.loads(res["result"]["content"][0]["text"])
names=[s["call_name"] for s in payload.get("steps",[])]
print(f"plan latency: {dt:.1f}s")
print("steps:", names)
print("missing:", payload.get("missing"))
assert any("browser_take_screenshot" in n or "browser_navigate" in n for n in names), \
    "REGRESSION: no playwright tool in plan steps"
print("OK: playwright tool present in plan")
PY
```
Expected: a `playwright__browser_navigate`/`browser_take_screenshot` step is present (the bug is fixed), and `plan latency` is printed. **Success criterion: warm latency p50 ≲ 8s.**

- [ ] **Step 2a: Settle the concurrency open question**

Read the printed latency:
- **≲ 8s:** concurrent `asyncio.gather` is fine — no change needed.
- **> 8s or errors that look like reranker/CrossEncoder contention** (e.g. tracebacks from `sentence_transformers`/torch in the gateway logs): the local CrossEncoder is contending under concurrency. Apply the documented fallback — in `_gather_candidates_multi`, replace the `asyncio.gather(...)` block with a sequential loop:
  ```python
        lists: list[list[dict[str, Any]]] = []
        for q in queries:
            try:
                cands, _fallback = await _gather_candidates(q, per_intent_k)
                lists.append(cands)
            except Exception:
                logger.debug("per-intent retrieval failed", exc_info=True)
  ```
  Re-run Step 1–2. If sequential is still > 8s, open a follow-up to add a "skip rerank for plan candidate gathering" flag (out of scope for this plan — log it, don't build it here). Commit any change with `git -c commit.gpgsign=false commit -m "[FIX] serialize plan per-intent retrieval"`.

- [ ] **Step 3: Confirm a simple task still plans (no over-decomposition)**

```bash
python - <<'PY'
import httpx, json
H={"Authorization":"Bearer dev-key-full-access"}
B="http://tower.lan:8765/mcp"
hdr={**H,"Content-Type":"application/json","Accept":"application/json, text/event-stream"}
c=httpx.Client(timeout=120.0)
def parse(r):
    if "text/event-stream" in r.headers.get("content-type",""):
        out=None
        for ln in r.text.splitlines():
            if ln.startswith("data:"): out=json.loads(ln[5:].strip())
        return out
    return r.json()
r=c.post(B,headers=hdr,json={"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"p","version":"0"}}})
hdr["mcp-session-id"]=r.headers.get("mcp-session-id")
c.post(B,headers=hdr,json={"jsonrpc":"2.0","method":"notifications/initialized","params":{}})
res=parse(c.post(B,headers=hdr,json={"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plan","arguments":{"query":"search arXiv for recent papers on diffusion models"}}}))
payload=json.loads(res["result"]["content"][0]["text"])
names=[s["call_name"] for s in payload.get("steps",[])]
print("steps:", names)
assert any("arxiv__" in n for n in names), "single-intent plan lost arxiv tools"
print("OK: single-intent plan still works")
PY
```
Expected: `arxiv__*` steps present; `OK: single-intent plan still works`.

- [ ] **Step 4: Final commit (if Step 2a changed code) and wrap-up**

If no code changed in Step 2a, nothing to commit. Otherwise the `[FIX]` commit from Step 2a covers it. Report results (regression fixed, latency, concurrency decision) to the user.

---

## Self-review notes (completed by plan author)

- **Spec coverage:** decompose (Task 1) ✓; per-intent retrieve + full-query base + union/dedup/cap (Tasks 2–3) ✓; caps & env config (Tasks 3–4) ✓; fallback on decompose failure / all-retrieval failure / empty union (Task 3, Step 1) ✓; rerank-latency risk + concurrency decision (Task 5, Step 2a) ✓; testing approach (logic checks + live HTTP) ✓; backward-compat (only `plan` path touched; `top_k` still honoured) ✓.
- **Type consistency:** `_gather_candidates(query, top_k) -> (list, bool)` consumed as `cands, _fallback` everywhere; `_union_candidates(lists, cap) -> list`; `decompose(query) -> list[str]`; `_chat(messages) -> str` used by both `plan` and `decompose`.
- **No placeholders:** every code/command step shows concrete content.
