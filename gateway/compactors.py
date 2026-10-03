"""Compact oversized flight-search results before they reach the client model.

Fare engines return huge payloads (a single Kiwi / Google Flights / Duffel
round-trip search is 50-120 KB of JSON, plus the same again as
structuredContent), which floods the calling agent's context. For servers
flagged `compact_results: true`, the gateway rewrites a search result into a
short digest: the first COMPACT_MAX_OPTIONS options in the engine's own order,
one line each (price, route, times, stops, carriers/flight numbers, flags,
booking link), the total count, and how to get the raw result
(`run_tool` / `run_tools` with `full: true`).

Formats handled (by tool name + payload shape): Kiwi `search-flight`, Google
Flights (fli) `search_flights`, Duffel `search_flights` / `search_multi_city`,
Skiplagged `sk_flights_search` (markdown table: rows capped). Anything that
doesn't parse is returned unchanged — compaction must never lose a result.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

MAX_OPTIONS = int(os.environ.get("COMPACT_MAX_OPTIONS", "10"))
_NOTE = ("Compacted by the gateway: first {shown} of {total} options, in the engine's order. "
         "For the raw result call run_tool (or a run_tools item) with \"full\": true.")


def _t(ts: Any) -> str:
    """'2026-12-05T19:25:00' -> '12-05 19:25' (local times as given by the engine)."""
    s = str(ts or "")
    return f"{s[5:10]} {s[11:16]}" if len(s) >= 16 else s


def _h(seconds: Any) -> str:
    try:
        return f"{float(seconds) / 3600:.1f}h"
    except (TypeError, ValueError):
        return "?"


def _digest(engine: str, query: str, total: int, lines: list[str], extra: dict | None = None) -> str:
    out = {"compacted": True, "engine": engine, "query": query, "total": total, "shown": len(lines),
           "options": lines, **(extra or {}), "note": _NOTE.format(shown=len(lines), total=total)}
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------- Kiwi

def _kiwi_dir(d: dict) -> str:
    segs = d.get("segments") or []
    flights = ", ".join(f"{s.get('carrierName') or s.get('carrier')} {s.get('flightNumber') or ''}".strip() for s in segs)
    route = "-".join(d.get("route") or [d.get("from", "?"), d.get("to", "?")])
    return (f"{route} {_t(d.get('departureTime'))}→{_t(d.get('arrivalTime'))} "
            f"({_h(d.get('durationSeconds'))}, {d.get('stops', len(segs) - 1)} stop) [{flights}]")


def _kiwi(text: str, _args: dict) -> str | None:
    j = json.loads(text)
    its = j.get("itineraries")
    if not isinstance(its, list):
        return None
    cur = j.get("currency", "")
    lines = []
    for it in its[:MAX_OPTIONS]:
        parts = [f"{it.get('price')} {cur}", _kiwi_dir(it.get("outbound") or {})]
        if it.get("inbound"):
            parts.append("return " + _kiwi_dir(it["inbound"]))
        carriers = {s.get("carrier") for d in (it.get("outbound"), it.get("inbound")) if d for s in d.get("segments") or []}
        if len(carriers) > 1:
            parts.append("separate tickets (Kiwi combo)")
        bag = it.get("baggage") or {}
        if bag:
            parts.append(f"bags: personal {bag.get('personalItem', 0)}, cabin {bag.get('cabinBag', 0)}, checked {bag.get('checkedBag', 0)}")
        parts.append(it.get("bookingUrl") or "")
        lines.append(" | ".join(p for p in parts if p))
    return _digest("kiwi", j.get("query", ""), int(j.get("resultsCount") or len(its)), lines)


# ---------------------------------------------------------------- Google Flights (fli)

def _short(name: Any) -> str:
    s = str(name or "?").replace(" International", "").replace(" Airport", "")
    return s if len(s) <= 22 else s[:21] + "…"


def _google_dir(legs: list[dict]) -> str:
    if not legs:
        return ""
    stops = [_short(legs[0].get("departure_airport_name") or legs[0].get("departure_airport"))]
    stops += [_short(l.get("arrival_airport_name") or l.get("arrival_airport")) for l in legs]
    flights = ", ".join(f"{l.get('airline_code') or ''}{l.get('flight_number') or ''}" for l in legs)
    mins = sum(int(l.get("duration") or 0) for l in legs)
    return (f"{' → '.join(stops)} {_t(legs[0].get('departure_time'))}→{_t(legs[-1].get('arrival_time'))} "
            f"({len(legs) - 1} stop, {mins / 60:.1f}h flying) [{flights}]")


def _google(text: str, args: dict) -> str | None:
    j = json.loads(text)
    fl = j.get("flights")
    if not isinstance(fl, list):
        return None
    ret = str(args.get("return_date") or "")
    lines = []
    for f in fl[:MAX_OPTIONS]:
        legs = f.get("legs") or []
        # Round trips list outbound and return legs together: split at the return date.
        cut = next((i for i, l in enumerate(legs) if ret and str(l.get("departure_time", ""))[:10] >= ret), len(legs))
        parts = [f"{f.get('price')} {f.get('currency', '')}", _google_dir(legs[:cut])]
        if legs[cut:]:
            parts.append("return " + _google_dir(legs[cut:]))
        if f.get("self_transfer"):
            parts.append("self-transfer")
        lines.append(" | ".join(p for p in parts if p))
    q = f"{args.get('origin')}→{args.get('destination')} {args.get('departure_date')}" + (f" / {ret}" if ret else "")
    return _digest("google-flights", q, int(j.get("count") or len(fl)), lines,
                   {"booking": "https://www.google.com/travel/flights (search the same route/date)"})


# ---------------------------------------------------------------- Duffel

def _duffel(text: str, args: dict) -> str | None:
    j = json.loads(text)
    offers = j.get("offers")
    if not isinstance(offers, list):
        return None
    lines = []
    for o in offers[:MAX_OPTIONS]:
        p = o.get("price") or {}
        parts = [f"{p.get('amount')} {p.get('currency', '')}"]
        for i, s in enumerate(o.get("slices") or []):
            route = "-".join([s.get("origin", "?")] + [c.get("airport", "?") for c in s.get("connections") or []]
                             + [s.get("destination", "?")])
            parts.append(f"{'return ' if i else ''}{route} {_t(s.get('departure'))}→{_t(s.get('arrival'))} "
                         f"({s.get('stops', 0)} stop, {s.get('duration', '')}) [{s.get('carrier', '')}]")
        parts.append(f"offer_id {o.get('offer_id')}")
        lines.append(" | ".join(parts))
    params = args.get("params") or args
    q = f"{params.get('origin', '')}→{params.get('destination', '')} {params.get('departure_date', '')}"
    return _digest("duffel", q, len(offers), lines)


# ---------------------------------------------------------------- Skiplagged (markdown)

def _skiplagged(text: str, _args: dict) -> str | None:
    if not text.lstrip().startswith("#"):
        return None
    lines = text.splitlines()
    rows = [i for i, l in enumerate(lines) if l.startswith("| $")]
    if len(rows) <= MAX_OPTIONS:
        return None  # already small
    keep = set(range(rows[0])) | set(rows[:MAX_OPTIONS])
    out = [l for i, l in enumerate(lines) if i in keep]
    out.append(f"\n_{_NOTE.format(shown=MAX_OPTIONS, total=len(rows))}_")
    return "\n".join(out)


_BY_TOOL: dict[str, list[Callable[[str, dict], str | None]]] = {
    "search-flight": [_kiwi],
    "search_flights": [_google, _duffel],
    "search_multi_city": [_duffel],
    "sk_flights_search": [_skiplagged],
}


def compact(tool: str, arguments: dict | None, text: str) -> str | None:
    """Digest for a known flight-search result, or None to keep the original."""
    for fn in _BY_TOOL.get(tool, ()):
        try:
            out = fn(text, arguments or {})
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if out is not None and len(out) < len(text):
            return out
    return None
