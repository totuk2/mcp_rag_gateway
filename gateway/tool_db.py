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

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None