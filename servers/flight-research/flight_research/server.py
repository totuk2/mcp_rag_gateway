"""flight-research MCP server: aviation data + the research_route pipeline.

Streamable HTTP on 0.0.0.0:8000/mcp. The gateway does auth, so FastMCP's
localhost-only DNS-rebinding check is disabled (compose service-name Host).
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from flight_research import airports
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
        "and combines self-transfer legs."
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
    """Systematic flight research for hard routes, in one call (~30-75 s).
    Expands nearby airports around origin and destination, finds hubs that actually
    fly into the destination area, queries Kiwi.com, Google Flights, Skiplagged and
    Duffel for the whole trip and for origin→hub / hub→destination legs, combines
    self-transfer options, and ranks them by price, time and risk.
    Returns `options` (ranked, with segments and booking links), `hubs`, `coverage`
    (what was searched / failed) and `manual_checks` (airlines on the last leg that no
    engine priced — check their websites). For a flexible window (e.g. "any day in
    December") pass `date` = first day and `date_to` = last day: the whole-trip search
    covers every day in the range in ONE call — prefer this over many calls with
    different dates. `origin`/`destination` may be a country ("Poland", "PL") = any of
    its main airports; otherwise an IATA/metro code or
    city name in English or local spelling;
    `date`: YYYY-MM-DD."""
    try:
        return _json(await _research_route(origin, destination, date, flex_days, adults, max_price_eur,
                                           dest_radius_km, origin_radius_km, allow_ground, adb, date_to))
    except ValueError as e:
        return _json({"error": str(e)})


def main() -> None:
    airports.load()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
