"""SQLite-backed persistent store for tool metadata.

Thread-safe via single connection + WAL mode.
Auto-creates schema on first use.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gateway.tool_record import ToolRecord, ToolStatus, ToolType


class ToolDb:
    """Persistent store for ToolRecords backed by SQLite."""

    def __init__(self, db_path: str | Path = "tool_registry.db"):
        self._lock = threading.Lock()
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                tool_id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                server_id TEXT NOT NULL,
                server_name TEXT NOT NULL DEFAULT '',
                transport TEXT NOT NULL DEFAULT '',
                input_schema TEXT NOT NULL DEFAULT '{}',
                output_schema TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                permission_scope TEXT NOT NULL DEFAULT '',
                risk_level TEXT NOT NULL DEFAULT 'low',
                tool_type TEXT NOT NULL DEFAULT 'action',
                status TEXT NOT NULL DEFAULT 'active',
                version TEXT NOT NULL DEFAULT '0.0.1',
                last_seen_at TEXT NOT NULL DEFAULT '',
                last_indexed_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_server_id ON tools(server_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_status ON tools(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_tool_type ON tools(tool_type)"
        )
        # Catalog (tool_rag/enrichment.py): per-server profile, per-tool tag cache,
        # and the versioned dynamic taxonomy. Derived data — safe to drop.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS server_profiles (
                server_id TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                fingerprint TEXT NOT NULL DEFAULT '',
                upstream_title TEXT NOT NULL DEFAULT '',
                upstream_instructions TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_tags (
                tool_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                taxonomy_version INTEGER NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]'
            )
        """
        )
        # Site recipes (gateway/recipes.py): how to query a website that agents
        # learned by browser automation. Shared across keys; NOT derived data.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS site_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site TEXT NOT NULL,
                task TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                url_template TEXT NOT NULL DEFAULT '',
                steps TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                author_key TEXT NOT NULL DEFAULT '',
                successes INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                verified_at TEXT NOT NULL DEFAULT '',
                UNIQUE(site, task)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog_taxonomy (
                version INTEGER PRIMARY KEY AUTOINCREMENT,
                catalog_fingerprint TEXT NOT NULL,
                categories TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )
        conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")

    @staticmethod
    def _serialize(val: Any) -> str:
        if val is None:
            return "null"
        return json.dumps(val, ensure_ascii=False, default=str)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ToolRecord:
        return ToolRecord(
            tool_id=row["tool_id"],
            tool_name=row["tool_name"],
            description=row["description"],
            server_id=row["server_id"],
            server_name=row["server_name"],
            transport=row["transport"],
            input_schema=json.loads(row["input_schema"]),
            output_schema=json.loads(row["output_schema"]) if row["output_schema"] else None,
            tags=tuple(json.loads(row["tags"])),
            permission_scope=row["permission_scope"],
            risk_level=row["risk_level"],
            tool_type=row["tool_type"],  # type: ignore[arg-type]
            status=row["status"],  # type: ignore[arg-type]
            version=row["version"],
            last_seen_at=row["last_seen_at"],
            last_indexed_at=row["last_indexed_at"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_tool(self, record: ToolRecord) -> None:
        """Insert or update a tool record."""
        now = self._now()
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO tools (
                    tool_id, tool_name, description, server_id, server_name, transport,
                    input_schema, output_schema, tags, permission_scope, risk_level,
                    tool_type, status, version, last_seen_at, last_indexed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                    tool_name=excluded.tool_name,
                    description=excluded.description,
                    server_id=excluded.server_id,
                    server_name=excluded.server_name,
                    transport=excluded.transport,
                    input_schema=excluded.input_schema,
                    output_schema=excluded.output_schema,
                    tags=excluded.tags,
                    permission_scope=excluded.permission_scope,
                    risk_level=excluded.risk_level,
                    tool_type=excluded.tool_type,
                    status=excluded.status,
                    version=excluded.version,
                    last_seen_at=excluded.last_seen_at,
                    last_indexed_at=excluded.last_indexed_at,
                    updated_at=excluded.updated_at
                """,
                (
                    record.tool_id,
                    record.tool_name,
                    record.description,
                    record.server_id,
                    record.server_name,
                    record.transport,
                    self._serialize(record.input_schema),
                    self._serialize(record.output_schema),
                    self._serialize(list(record.tags)),
                    record.permission_scope,
                    record.risk_level,
                    record.tool_type,
                    record.status,
                    record.version,
                    record.last_seen_at,
                    record.last_indexed_at,
                    now,
                ),
            )
            conn.commit()

    def get_tool(self, tool_id: str) -> ToolRecord | None:
        """Fetch one tool by ID."""
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM tools WHERE tool_id = ?", (tool_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def delete_tool(self, tool_id: str) -> bool:
        """Remove a tool record. Returns True if deleted."""
        with self._lock:
            cur = self._connect().execute("DELETE FROM tools WHERE tool_id = ?", (tool_id,))
            conn = self._connect()
            conn.commit()
        return cur.rowcount > 0

    def list_tools(
        self,
        server_id: str | None = None,
        tool_type: ToolType | None = None,
        status: ToolStatus | None = None,
    ) -> list[ToolRecord]:
        """List tools with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if server_id is not None:
            clauses.append("server_id = ?")
            params.append(server_id)
        if tool_type is not None:
            clauses.append("tool_type = ?")
            params.append(tool_type)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock:
            rows = self._connect().execute(
                f"SELECT * FROM tools{where} ORDER BY server_id, tool_name", params
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all_tools(self) -> list[ToolRecord]:
        """Return every tool in the DB."""
        with self._lock:
            rows = self._connect().execute(
                "SELECT * FROM tools ORDER BY server_id, tool_name"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def mark_stale(self, server_id: str, before: str | None = None) -> int:
        """Mark tools as stale for a given server.

        If *before* is provided, only tools whose last_seen_at is older
        (or empty) are marked.  Returns count of affected rows.
        """
        if before:
            with self._lock:
                cur = self._connect().execute(
                    "UPDATE tools SET status='deprecated', updated_at=? "
                    "WHERE server_id=? AND (last_seen_at < ? OR last_seen_at = '')",
                    (self._now(), server_id, before),
                )
                self._connect().commit()
            return cur.rowcount
        with self._lock:
            cur = self._connect().execute(
                "UPDATE tools SET status='deprecated', updated_at=? WHERE server_id=?",
                (self._now(), server_id),
            )
            self._connect().commit()
        return cur.rowcount

    def get_stale_count(self) -> int:
        """Number of tools with status = 'deprecated'."""
        with self._lock:
            row = self._connect().execute(
                "SELECT COUNT(*) AS cnt FROM tools WHERE status='deprecated'"
            ).fetchone()
        return row["cnt"] if row else 0

    def count_tools(self) -> int:
        """Total number of tool records."""
        with self._lock:
            row = self._connect().execute("SELECT COUNT(*) AS cnt FROM tools").fetchone()
        return row["cnt"] if row else 0

    def count_servers(self) -> int:
        """Number of distinct servers in the DB."""
        with self._lock:
            row = self._connect().execute(
                "SELECT COUNT(DISTINCT server_id) AS cnt FROM tools"
            ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Catalog: server profiles, tool tag cache, taxonomy
    # ------------------------------------------------------------------

    def set_server_upstream_info(self, server_id: str, title: str, instructions: str) -> None:
        """Record what the upstream said about itself at `initialize` (sync time)."""
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO server_profiles (server_id, upstream_title, upstream_instructions, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(server_id) DO UPDATE SET
                    upstream_title=excluded.upstream_title,
                    upstream_instructions=excluded.upstream_instructions
                """,
                (server_id, title, instructions, self._now()),
            )
            conn.commit()

    def save_server_profile(self, server_id: str, description: str, source: str, fingerprint: str) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO server_profiles (server_id, description, source, fingerprint, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(server_id) DO UPDATE SET
                    description=excluded.description,
                    source=excluded.source,
                    fingerprint=excluded.fingerprint,
                    updated_at=excluded.updated_at
                """,
                (server_id, description, source, fingerprint, self._now()),
            )
            conn.commit()

    def list_server_profiles(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._connect().execute("SELECT * FROM server_profiles").fetchall()
        return {r["server_id"]: dict(r) for r in rows}

    def delete_server_profile(self, server_id: str) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM server_profiles WHERE server_id = ?", (server_id,))
            conn.commit()

    def get_all_tool_tags(self) -> dict[str, dict[str, Any]]:
        """tool_id -> {fingerprint, taxonomy_version, tags: tuple}."""
        with self._lock:
            rows = self._connect().execute("SELECT * FROM tool_tags").fetchall()
        return {
            r["tool_id"]: {
                "fingerprint": r["fingerprint"],
                "taxonomy_version": r["taxonomy_version"],
                "tags": tuple(json.loads(r["tags"])),
            }
            for r in rows
        }

    def save_tool_tags(self, entries: dict[str, tuple[str, int, tuple[str, ...]]]) -> None:
        """Bulk upsert tool_id -> (fingerprint, taxonomy_version, tags)."""
        with self._lock:
            conn = self._connect()
            conn.executemany(
                """
                INSERT INTO tool_tags (tool_id, fingerprint, taxonomy_version, tags)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint,
                    taxonomy_version=excluded.taxonomy_version,
                    tags=excluded.tags
                """,
                [(tid, fp, ver, self._serialize(list(tags))) for tid, (fp, ver, tags) in entries.items()],
            )
            conn.commit()

    def delete_tool_tags(self, tool_ids: list[str]) -> None:
        with self._lock:
            conn = self._connect()
            conn.executemany("DELETE FROM tool_tags WHERE tool_id = ?", [(t,) for t in tool_ids])
            conn.commit()

    def get_current_taxonomy(self) -> dict[str, Any] | None:
        """Latest taxonomy: {version, catalog_fingerprint, categories: [{name, description}]}."""
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM catalog_taxonomy ORDER BY version DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "version": row["version"],
            "catalog_fingerprint": row["catalog_fingerprint"],
            "categories": json.loads(row["categories"]),
        }

    def save_taxonomy(self, catalog_fingerprint: str, categories: list[dict[str, str]]) -> int:
        """Store a new taxonomy version; returns its version number."""
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "INSERT INTO catalog_taxonomy (catalog_fingerprint, categories, created_at) VALUES (?, ?, ?)",
                (catalog_fingerprint, self._serialize(categories), self._now()),
            )
            conn.commit()
        return int(cur.lastrowid)

    def update_taxonomy(self, version: int, catalog_fingerprint: str, categories: list[dict[str, str]]) -> None:
        """Rewrite a taxonomy version in place (same category names, new fingerprint
        and/or descriptions) — tags assigned under it stay valid."""
        with self._lock:
            conn = self._connect()
            conn.execute(
                "UPDATE catalog_taxonomy SET catalog_fingerprint = ?, categories = ? WHERE version = ?",
                (catalog_fingerprint, self._serialize(categories), version),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Site recipes
    # ------------------------------------------------------------------

    @staticmethod
    def _recipe_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["steps"] = json.loads(d["steps"])
        return d

    def upsert_recipe(self, site: str, task: str, label: str, url_template: str, steps: list[str],
                      notes: str, author_key: str) -> dict[str, Any]:
        """Insert or replace the recipe for (site, task); saving means it just worked,
        so it counts as a success and refreshes verified_at."""
        now = self._now()
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO site_recipes (site, task, label, url_template, steps, notes, author_key,
                                          successes, failures, created_at, updated_at, verified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?)
                ON CONFLICT(site, task) DO UPDATE SET
                    label=COALESCE(NULLIF(excluded.label, ''), site_recipes.label),
                    url_template=excluded.url_template, steps=excluded.steps,
                    notes=excluded.notes, author_key=excluded.author_key,
                    successes=site_recipes.successes + 1, failures=0,
                    updated_at=excluded.updated_at, verified_at=excluded.verified_at
                """,
                (site, task, label, url_template, self._serialize(steps), notes, author_key, now, now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM site_recipes WHERE site = ? AND task = ?", (site, task)).fetchone()
        return self._recipe_row(row)

    def report_recipe(self, recipe_id: int, worked: bool) -> dict[str, Any] | None:
        now = self._now()
        with self._lock:
            conn = self._connect()
            if worked:
                conn.execute("UPDATE site_recipes SET successes = successes + 1, verified_at = ? WHERE id = ?",
                             (now, recipe_id))
            else:
                conn.execute("UPDATE site_recipes SET failures = failures + 1 WHERE id = ?", (recipe_id,))
            conn.commit()
            row = conn.execute("SELECT * FROM site_recipes WHERE id = ?", (recipe_id,)).fetchone()
        return self._recipe_row(row) if row else None

    def find_recipes(self, site: str | None = None, text: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """By exact site (domain), or by substring over site/label/task. Most reliable first."""
        clauses, params = [], []
        if site:
            clauses.append("site = ?")
            params.append(site)
        if text:
            clauses.append("(site LIKE ? OR label LIKE ? OR task LIKE ?)")
            params += [f"%{text}%"] * 3
        where = " WHERE " + " OR ".join(clauses) if clauses else ""
        with self._lock:
            rows = self._connect().execute(
                f"SELECT * FROM site_recipes{where} "
                "ORDER BY (successes - 2 * failures) DESC, verified_at DESC, id LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._recipe_row(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None