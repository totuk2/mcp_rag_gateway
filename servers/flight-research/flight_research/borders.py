"""Land borders: which countries touch (data) and what's known about crossing them (memory).

GEONAMES_COUNTRY_INFO (https://download.geonames.org/export/dump/countryInfo.txt,
CC BY 4.0) lists each country's land neighbours. `land_link` override rows add the
fixed links it lacks (tunnels, bridges, causeways). A country with no neighbours is
an island state: no road to anywhere abroad, so a ground leg there is flagged, not
dropped — the user has to check the ferry or sea connection.

Whether a border can actually be crossed changes with politics, so it isn't a table
in code: it's a memory in the cache DB. Agents add to it via report_border_status
when a web search shows a crossing closed, restricted or reopened (latest report per
border wins). `closed_border` override rows seed it once and never overwrite reports.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from flight_research import airports, cache

COUNTRY_INFO = os.environ.get("GEONAMES_COUNTRY_INFO", "/app/data/countryInfo.txt")
# Ground legs are short (dest_radius_km), so paths through more countries are no option.
MAX_CROSSINGS = int(os.environ.get("BORDER_MAX_CROSSINGS", "2"))
# Older reports still count, but are shown as possibly outdated.
STALE_DAYS = float(os.environ.get("BORDER_STATUS_STALE_DAYS", "90"))
STATUSES = ("open", "restricted", "closed")

_neighbours: dict[str, set[str]] = {}
_loaded = False


def pair(a: str, b: str) -> str:
    return "-".join(sorted((a.upper(), b.upper())))


def load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if os.path.exists(COUNTRY_INFO):
        with open(COUNTRY_INFO, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if line.startswith("#") or len(cols) < 18 or len(cols[0]) != 2:
                    continue
                _neighbours.setdefault(cols[0], set()).update(n for n in cols[17].split(",") if n)
    for key, value in airports.override_rows("land_link"):
        a, b = key.upper(), value.upper()
        _neighbours.setdefault(a, set()).add(b)
        _neighbours.setdefault(b, set()).add(a)
    now = time.time()
    with cache._lock:
        db = cache._db()
        db.execute("CREATE TABLE IF NOT EXISTS border_status (pair TEXT PRIMARY KEY, status TEXT NOT NULL, "
                   "note TEXT, source_url TEXT, reported_at REAL NOT NULL, origin TEXT NOT NULL)")
        db.executemany("INSERT OR IGNORE INTO border_status VALUES (?, ?, ?, NULL, ?, 'seed')",
                       [(pair(k, v), "closed",
                         "seeded from the former built-in list; re-verify", now)
                        for k, v in airports.override_rows("closed_border")])
        db.commit()


def neighbours(code: str) -> set[str]:
    load()
    return _neighbours.get(code.upper(), set())


def is_known_country(code: str) -> bool:
    load()
    # Without GeoNames data any ISO-shaped code is accepted.
    return code.upper() in _neighbours if _neighbours else len(code) == 2 and code.isalpha()


def _row(r: tuple) -> dict:
    p, status, note, url, at, origin = r
    return {"border": p, "status": status, "note": note or "", "source_url": url,
            "reported": datetime.fromtimestamp(at, timezone.utc).date().isoformat(),
            "stale": time.time() - at > STALE_DAYS * 86400, "origin": origin}


def known() -> dict[str, dict]:
    """Every remembered border, by pair ("AM-TR")."""
    load()
    with cache._lock:
        rows = cache._db().execute("SELECT * FROM border_status").fetchall()
    return {r[0]: _row(r) for r in rows}


def lookup(a: str | None = None, b: str | None = None) -> list[dict]:
    rows = known().values()
    if a and b:
        return [r for r in rows if r["border"] == pair(a, b)]
    if a:
        return [r for r in rows if a.upper() in r["border"].split("-")]
    return sorted(rows, key=lambda r: r["border"])


def report(a: str, b: str, status: str, source_url: str, note: str = "") -> dict:
    """Remember the latest known state of the land border a-b (replaces older reports)."""
    load()
    a, b, status = a.upper(), b.upper(), status.lower()
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    if a == b or not (is_known_country(a) and is_known_country(b)):
        raise ValueError(f"need two different ISO country codes (got {a!r}, {b!r})")
    if not source_url.startswith(("http://", "https://")):
        raise ValueError("source_url must be the http(s) URL the information came from")
    with cache._lock:
        db = cache._db()
        db.execute("INSERT OR REPLACE INTO border_status VALUES (?, ?, ?, ?, ?, 'report')",
                   (pair(a, b), status, note.strip()[:500], source_url[:500], time.time()))
        db.commit()
    out = lookup(a, b)[0]
    if b not in neighbours(a):
        out["warning"] = f"{a} and {b} share no land border (GeoNames): this has no effect on ground routing"
    return out


def route(a: str, b: str, memory: dict[str, dict] | None = None) -> dict:
    """Best road path from country a to country b, as border crossings.

    Paths up to MAX_CROSSINGS borders; among them fewest closed, then restricted, then
    crossings — so a closed border is used only when there is no way around it.
    `land_link` False = no such path (island state or too far): ground leg unverified.
    """
    load()
    a, b = a.upper(), b.upper()
    if a == b:
        return {"land_link": True, "path": [a], "crossings": []}
    memory = known() if memory is None else memory
    status = lambda x, y: (memory.get(pair(x, y)) or {}).get("status")
    best: tuple | None = None
    if not _neighbours:  # no GeoNames data: assume the direct border, islands unknown
        best = (0, 0, 2, [a, b])
    stack = [[a]] if _neighbours else []
    while stack:
        path = stack.pop()
        for n in sorted(_neighbours.get(path[-1], ())):
            if n in path:
                continue
            p = path + [n]
            if n == b:
                st = [status(x, y) for x, y in zip(p, p[1:])]
                key = (st.count("closed"), st.count("restricted"), len(p), p)
                best = min(best, key) if best else key
            elif len(p) <= MAX_CROSSINGS:
                stack.append(p)
    if best is None:
        islands = [c for c in (a, b) if not _neighbours.get(c)]
        return {"land_link": False, "path": [a, b], "crossings": [], "islands": islands}
    path = best[-1]
    crossings = []
    for x, y in zip(path, path[1:]):
        m = memory.get(pair(x, y))
        crossings.append({"border": pair(x, y), "status": m["status"] if m else None,
                          **({k: m[k] for k in ("reported", "source_url", "note", "stale")} if m else {})})
    return {"land_link": True, "path": path, "crossings": crossings}


def warnings(r: dict) -> list[str]:
    """Human-readable cautions for a route() result (empty = nothing known against it)."""
    a, b = r["path"][0], r["path"][-1]
    if not r["land_link"]:
        if r.get("islands"):
            return [f"{'/'.join(r['islands'])} is an island state: no road link {a}→{b}; "
                    "double-check the ferry/sea connection before relying on this airport"]
        return [f"no road route {a}→{b} within {MAX_CROSSINGS} border crossings; verify the ground transfer"]
    out = []
    for c in r["crossings"]:
        if c["status"] in ("closed", "restricted"):
            src = f", {c['source_url']}" if c.get("source_url") else ""
            note = f": {c['note']}" if c.get("note") else ""
            old = " — may be outdated, re-verify" if c.get("stale") else ""
            out.append(f"border {c['border']} reported {c['status'].upper()} ({c['reported']}{src}){note}{old}")
    return out
