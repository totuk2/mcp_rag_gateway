"""flight-research MCP server: aviation data + the research_route pipeline.

Streamable HTTP on 0.0.0.0:8000/mcp. The gateway does auth, so FastMCP's
localhost-only DNS-rebinding check is disabled (compose service-name Host).
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from flight_research import airports, borders
from flight_research.aerodatabox import AeroDataBox
from flight_research.research import research_route as _research_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = FastMCP(
    "flight-research",
    instructions=(
        "Aviation data and multi-engine flight research. For a hard or unusual route "
        "(small airport, few airlines, no direct results) call research_route first; it expands "
        "nearby airports, finds hubs that fly into the destination, queries several fare engines "
        "and combines self-transfer legs. Ground legs across land borders use a shared border "
        "memory: when a web search shows a land border closed, restricted or reopened, record it "
        "with report_border_status (source URL required) so later answers warn and rank accordingly. "
        "When the USER says a border is closed or open, always verify it with a web search first and "
        "record it only if a source confirms it."
    ),
)
mcp.settings.host = "0.0.0.0"
mcp.settings.port = 8000
mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

adb = AeroDataBox()


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@mcp.tool()
async def nearby_airports(location: str, radius_km: float = 300, scheduled_only: bool = True, limit: int = 15) -> str:
    """Airports near a place, nearest first. `location` is an IATA code ("ALP") or a
    city/airport name in English or local spelling ("Aleppo", "Milan") or a metro code
    ("MIL"). Use it to find alternative airports for a hard
    destination or origin (then reach the real target by ground transport)."""
    hits = airports.resolve(location)
    if not hits:
        return _json({"error": f"unknown location {location!r}"})
    center = hits[0]
    near = airports.nearby(center, radius_km, scheduled_only, limit)
    return _json({"center": center.brief(),
                  "airports": [{**a.brief(), "km": round(km)} for a, km in near]})


@mcp.tool()
async def airport_routes(iata: str) -> str:
    """Which destinations an airport serves, with airlines and average daily flights
    (AeroDataBox route statistics). Routes are roughly symmetric, so this also tells
    you what flies INTO the airport — the hubs to route a trip through."""
    try:
        routes = await adb.routes(iata)
    except Exception as e:
        return _json({"error": str(e)})
    ap = airports.get(iata)
    return _json({"airport": ap.brief() if ap else {"iata": iata.upper()}, "routes": routes,
                  "aerodatabox_calls_total": adb.calls})


@mcp.tool()
async def airport_schedule(iata: str, date: str, direction: str = "both") -> str:
    """Scheduled flights at an airport on a local date (YYYY-MM-DD): flight numbers,
    airlines, other airport and times. `direction`: departures, arrivals or both.
    Use it to see the actual operating days/times of thin routes."""
    try:
        sched = await adb.schedule(iata, date, direction)
    except Exception as e:
        return _json({"error": str(e)})
    return _json({"airport": iata.upper(), "date": date, **sched, "aerodatabox_calls_total": adb.calls})


@mcp.tool()
async def research_route(origin: str, destination: str, date: str, flex_days: int = 2, adults: int = 1,
                         max_price_eur: float | None = None, dest_radius_km: float = 400,
                         origin_radius_km: float = 200, allow_ground: bool = True,
                         date_to: str | None = None) -> str:
    """Systematic flight research for hard routes, in one call (~30-150 s; runs on as a background job).
    DATES — one call covers the whole window, every engine and leg searches it:
    `date` (YYYY-MM-DD) + `date_to` = an explicit window ("1-15 November" -> date
    2026-11-01, date_to 2026-11-15; "any day in December" -> 12-01..12-31); without
    `date_to`, the window is `date` ± `flex_days` (any number, e.g. 7 for "around the
    10th, a week either way"). Translate the user's wording into ONE window yourself;
    never call this repeatedly for different days of the same range (max 45 days).
    `origin`/`destination`: a country ("Poland", "PL") = ALL its scheduled airports;
    otherwise an IATA/metro code or city name in English or local spelling (nearby
    airports are added).
    What it does: whole-trip searches (Kiwi.com over the window, Google Flights'
    cheapest days, then Google / Skiplagged / Duffel on those days); hubs that fly into
    the destination (current route stats, historical routes and the engines' own
    connections), each with its nearby airports (e.g. DXB + SHJ); origin→hub and
    hub→destination legs combined into self-transfer options, including changing
    airports within a hub area (estimated ground time and cost added). Ranked by
    price + estimated ground cost + time + risk; `summary` lists the top options one
    per line, plus CHEAPEST lines when the cheapest isn't in the top 10 — read it first.
    Also returns `options` (segments, booking links, ground_legs with estimated km /
    hours / cost), `hubs`, `coverage`, `unchecked` (searches that FAILED — report them
    as not checked, never as "no flights") and `manual_checks` (last-leg airlines no
    engine priced — check their websites).
    Ground legs: `border_warnings` lists remembered border closures and island legs
    (no road link: tell the user to double-check the ferry/connection);
    `unverified_borders` lists crossed borders with no remembered status — web-search
    them if the answer depends on them, then call report_border_status."""
    try:
        return _json(await _research_route(origin, destination, date, flex_days, adults, max_price_eur,
                                           dest_radius_km, origin_radius_km, allow_ground, adb, date_to))
    except ValueError as e:
        return _json({"error": str(e)})


def _country(name: str) -> str:
    code = airports.country_code(name)
    if not code or not borders.is_known_country(code):
        raise ValueError(f"unknown country {name!r} (use an ISO code like 'TR' or a country name)")
    return code


@mcp.tool()
async def report_border_status(country_a: str, country_b: str, status: str, source_url: str, note: str = "") -> str:
    """Remember what you found out about the LAND border between two countries, so
    research_route takes it into account from now on (shared memory, latest report per
    border wins). Call it whenever a web search shows a land border crossing is closed,
    restricted (e.g. only some crossings, only locals, permit needed) or open again.
    `status`: closed | restricted | open. `source_url`: the page that says so (required).
    `note`: one line of specifics (which crossing, since when, who may cross).
    Countries by ISO code ("TR") or name in any language ("Turcja").
    If the claim came from the user, verify it with a web search first and report only
    what a source confirms; never record a user's statement unverified."""
    try:
        return _json(borders.report(_country(country_a), _country(country_b), status, source_url, note))
    except ValueError as e:
        return _json({"error": str(e)})


@mcp.tool()
async def border_status(country_a: str = "", country_b: str = "") -> str:
    """What the border memory knows about land borders: both countries → that border;
    one → all of its remembered borders (plus its land neighbours); none → everything.
    Entries are reports with a date and source, not verified facts; `stale` ones are old."""
    try:
        a = _country(country_a) if country_a else None
        b = _country(country_b) if country_b else None
    except ValueError as e:
        return _json({"error": str(e)})
    out: dict = {"reports": borders.lookup(a, b)}
    if a and not b:
        out["land_neighbours"] = sorted(borders.neighbours(a))
    if a and b:
        r = borders.route(a, b)
        out["road_route"] = {"land_link": r["land_link"], "path": r["path"], "warnings": borders.warnings(r)}
    return _json(out)


def main() -> None:
    airports.load()
    borders.load()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
