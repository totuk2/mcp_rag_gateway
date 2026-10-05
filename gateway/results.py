"""Result handles and response projection (keep big tool results out of context).

Two ideas from Uber's MCP gateway design:
  - response projection: the caller names the fields it needs (`fields`, paths
    like "itineraries[].price", "query", "items[0].title") and the gateway returns
    only those;
  - results as handles: when the gateway shortens a result (compact digest, or a
    text over RESULT_MAX_CHARS), the full result is kept for RESULT_TTL_S under a
    `result_id`, and `get_result` reads it back sliced, projected or grepped —
    instead of re-running the call (slow, and e.g. fares change).

Results belong to the API key that produced them; in memory only (lost on restart).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

RESULT_TTL_S = float(os.environ.get("RESULT_TTL_S", "1800"))
RESULT_MAX_CHARS = int(os.environ.get("RESULT_MAX_CHARS", "40000"))
RESULT_STORE_MAX_CHARS = int(os.environ.get("RESULT_STORE_MAX_CHARS", "50000000"))
DEFAULT_SLICE = 20000
MAX_GREP_LINES = 200

_TOKEN = re.compile(r"^([^\[\]]*)(?:\[(\d*)\])?$")
_MISSING = object()


class ProjectionError(ValueError):
    pass


def _parse_path(path: str) -> list[tuple[str, str | None]]:
    toks = []
    for part in path.strip().split("."):
        m = _TOKEN.match(part)
        if not m or (not m.group(1) and m.group(2) is None):
            raise ProjectionError(f"bad field path {path!r} (use a.b, items[].x, items[0].x)")
        toks.append((m.group(1), m.group(2)))  # idx: None = no brackets, "" = all, "3" = index
    return toks


def _build(obj: Any, toks: list[tuple[str, str | None]]) -> Any:
    name, idx = toks[0]
    rest = toks[1:]
    cur = obj
    if name:
        if not isinstance(cur, dict) or name not in cur:
            return _MISSING
        cur = cur[name]
    if idx is not None:
        if not isinstance(cur, list):
            return _MISSING
        if idx == "":
            items = [x if not rest else _build(x, rest) for x in cur]
            cur = [x for x in items if x is not _MISSING]
            rest = []
        else:
            i = int(idx)
            if i >= len(cur):
                return _MISSING
            cur = cur[i]
    if rest:
        cur = _build(cur, rest)
        if cur is _MISSING:
            return _MISSING
    return {name: cur} if name else cur


def _merge(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(out[k], v) if k in out else v
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_merge(x, y) for x, y in zip(a, b)]
    return b


def project(data: Any, fields: list[str]) -> Any:
    """Keep only the given paths of a JSON value, preserving its shape."""
    out: Any = _MISSING
    for f in fields:
        part = _build(data, _parse_path(f))
        if part is not _MISSING:
            out = part if out is _MISSING else _merge(out, part)
    return {} if out is _MISSING else out


def view(text: str, fields: list[str] | None = None, grep: str | None = None,
         offset: int = 0, limit: int = DEFAULT_SLICE) -> tuple[str, int]:
    """A window onto a stored result: projected (`fields`), filtered (`grep`,
    regex over lines; JSON is pretty-printed first so each value is a line), then
    sliced (`offset`/`limit` chars). Returns (text, total_chars_before_slicing)."""
    body = text
    if fields:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProjectionError("this result is not JSON; use grep or offset/limit instead") from e
        body = json.dumps(project(data, fields), ensure_ascii=False)
    if grep:
        try:
            rx = re.compile(grep, re.IGNORECASE)
        except re.error as e:
            raise ProjectionError(f"bad grep regex: {e}") from e
        try:
            lines = json.dumps(json.loads(body), ensure_ascii=False, indent=1).splitlines()
        except json.JSONDecodeError:
            lines = body.splitlines()
        hits = [f"{i + 1}: {l}" for i, l in enumerate(lines) if rx.search(l)]
        body = "\n".join(hits[:MAX_GREP_LINES]) + (
            f"\n… {len(hits) - MAX_GREP_LINES} more matching lines" if len(hits) > MAX_GREP_LINES else "")
        if not hits:
            body = "(no matching lines)"
    total = len(body)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), RESULT_MAX_CHARS))
    return body[offset:offset + limit], total


@dataclass
class _Stored:
    key_id: str
    tool: str
    text: str
    created: float


class ResultStore:
    def __init__(self) -> None:
        self._items: OrderedDict[str, _Stored] = OrderedDict()
        self._chars = 0

    def put(self, key_id: str, tool: str, text: str) -> str:
        self._gc()
        rid = "r_" + secrets.token_hex(6)
        self._items[rid] = _Stored(key_id, tool, text, time.monotonic())
        self._chars += len(text)
        while self._chars > RESULT_STORE_MAX_CHARS and len(self._items) > 1:
            _, old = self._items.popitem(last=False)
            self._chars -= len(old.text)
        return rid

    def get(self, key_id: str, rid: str) -> _Stored | None:
        self._gc()
        s = self._items.get(rid)
        return s if s is not None and s.key_id == key_id else None

    def _gc(self) -> None:
        now = time.monotonic()
        for rid in [r for r, s in self._items.items() if now - s.created > RESULT_TTL_S]:
            self._chars -= len(self._items.pop(rid).text)
