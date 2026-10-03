"""Tiny SQLite TTL cache (API quotas: AeroDataBox free tier, ECB rates)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

_PATH = os.environ.get("FLIGHT_RESEARCH_CACHE", "/data/cache.db")
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
        _conn = sqlite3.connect(_PATH, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, expires REAL)")
    return _conn


def get(key: str) -> Any | None:
    with _lock:
        row = _db().execute("SELECT value, expires FROM cache WHERE key = ?", (key,)).fetchone()
    if row is None or row[1] < time.time():
        return None
    return json.loads(row[0])


def put(key: str, value: Any, ttl_s: float) -> None:
    with _lock:
        _db().execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value), time.time() + ttl_s))
        _db().commit()
