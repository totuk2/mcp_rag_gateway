"""Fare engines, each called over MCP and normalized to one itinerary shape:

    {"source", "price", "currency", "booking_url",
     "segments": [{"from", "to", "dep", "arr", "carrier", "flight"}],
     "self_transfer": bool | None, "notes": [str]}

Engines: Kiwi.com and Skiplagged (hosted MCP), Google Flights (fli, the
`google-flights` compose service) and Duffel (the `flights` compose service).
URLs are env-configurable; an engine whose URL is empty is skipped.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from datetime import date as Date

import logging

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from flight_research import airports

URLS = {
    "kiwi": os.environ.get("KIWI_MCP_URL", "https://mcp.kiwi.com"),
    "google": os.environ.get("GOOGLE_FLIGHTS_MCP_URL", "http://google-flights:8000/mcp"),
    "skiplagged": os.environ.get("SKIPLAGGED_MCP_URL", "https://mcp.skiplagged.com/mcp"),
    "duffel": os.environ.get("DUFFEL_MCP_URL", "http://flights:8000/mcp"),
}
ENGINE_TIMEOUT = float(os.environ.get("ENGINE_TIMEOUT", "45"))
# Identical engine queries within this window share one upstream call: parallel
# research_route runs (e.g. Poland -> Amman / Beirut / Damascus) overlap heavily.
CACHE_TTL_S = float(os.environ.get("ENGINE_CACHE_TTL", "900"))
_cache: dict[str, tuple[float, str]] = {}
_inflight: dict[str, asyncio.Future] = {}
# Hosted engines answer 503 when hit by many sessions at once (Kiwi): cap concurrent
# calls per engine URL and retry transient failures with back-off.
ENGINE_CONCURRENCY = int(os.environ.get("ENGINE_CONCURRENCY", "3"))
RETRIES = int(os.environ.get("ENGINE_RETRIES", "3"))
RETRY_BACKOFF_S = float(os.environ.get("ENGINE_RETRY_BACKOFF_S", "2"))
_sems: dict[tuple[int, str], asyncio.Semaphore] = {}


log = logging.getLogger(__name__)


class EngineError(RuntimeError):
    pass


def root_cause(e: BaseException) -> BaseException:
    """First leaf of anyio TaskGroup exception groups (the real error)."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e


def _transient(e: BaseException) -> bool:
    e = root_cause(e)
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 500, 502, 503, 504)
    import anyio
    return isinstance(e, (httpx.TransportError, anyio.BrokenResourceError, anyio.ClosedResourceError))


async def call_mcp(url: str, tool: str, args: dict) -> str:
    """Cached + de-duplicated engine call (see CACHE_TTL_S)."""
    key = json.dumps([url, tool, args], sort_keys=True)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
        return hit[1]
    if key in _inflight:
        return await asyncio.shield(_inflight[key])
    fut = asyncio.get_running_loop().create_future()
    _inflight[key] = fut
    try:
        sem = _sems.setdefault((id(asyncio.get_running_loop()), url), asyncio.Semaphore(ENGINE_CONCURRENCY))
        # The slot is held through the back-off: an overloaded engine gets less traffic,
        # and a retried call keeps its place instead of queueing behind everything.
        async with sem:
            for attempt in range(RETRIES + 1):
                try:
                    text = await _call_mcp(url, tool, args)
                    break
                except Exception as e:
                    if attempt >= RETRIES or not _transient(e):
                        raise
                    log.info("%s %s: %s; retry %d", url, tool, type(root_cause(e)).__name__, attempt + 1)
                    # Jitter: parallel calls that failed together must not retry together.
                    await asyncio.sleep(RETRY_BACKOFF_S * (2 ** attempt) * (0.5 + random.random()))
    except BaseException as e:
        if not fut.done():
            fut.set_exception(e)
            fut.exception()  # mark retrieved; waiters re-raise it
        raise
    else:
        _cache[key] = (time.monotonic(), text)
        if len(_cache) > 2000:
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:500]:
                del _cache[k]
        fut.set_result(text)
        return text
    finally:
        _inflight.pop(key, None)


async def _call_mcp(url: str, tool: str, args: dict) -> str:
    """One fresh MCP session per call (engines are stateless); returns the text content."""
    async with create_mcp_http_client({}, httpx.Timeout(20.0, read=ENGINE_TIMEOUT)) as hc:
        async with streamable_http_client(url, http_client=hc) as (read, write, _):
            async with ClientSession(read, write) as s:
                await s.initialize()
                res = await s.call_tool(tool, args)
    text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
    if res.isError:
        raise EngineError(text[:300] or "tool error")
    return text


def _seg(frm, to, dep, arr, carrier, flight) -> dict:
    return {"from": frm, "to": to, "dep": dep, "arr": arr, "carrier": carrier, "flight": flight}


# ---------------------------------------------------------------- Kiwi.com

async def kiwi(origins: list[str], destinations: list[str], day: str, flex_days: int = 0,
               adults: int = 1, diff_airport: bool = False, day_to: str | None = None) -> list[dict]:
    d = Date.fromisoformat(day)
    args = {
        "flyFrom": ",".join(origins), "flyTo": ",".join(destinations),
        "departureDate": f"{d:%d/%m/%Y}",
        "adults": adults, "currency": "EUR", "locale": "en", "sort": "price",
        "allow_self_transfer": True, "allow_diff_airport_connection": diff_airport,
    }
    if day_to:  # a whole date range in one query (Kiwi searches every day in it)
        args["departureDateTo"] = f"{Date.fromisoformat(day_to):%d/%m/%Y}"
    else:
        args["departureDateFlexDays"] = max(0, min(flex_days, 3))
    text = await call_mcp(URLS["kiwi"], "search-flight", args)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise EngineError(f"kiwi: unexpected response: {text[:200]}") from e
    out = []
    for it in data.get("itineraries") or []:
        ob = it.get("outbound") or {}
        segs = [_seg(s.get("from"), s.get("to"), s.get("departureTime"), s.get("arrivalTime"),
                     s.get("carrierName") or s.get("carrier"), s.get("flightNumber"))
                for s in ob.get("segments") or []]
        carriers = {s["carrier"] for s in segs}
        out.append({"source": "kiwi", "price": it.get("price"), "currency": data.get("currency", "EUR"),
                    "booking_url": it.get("bookingUrl"), "segments": segs,
                    # Kiwi sells multi-carrier combos as virtual interlining (separate tickets).
                    "self_transfer": len(carriers) > 1,
                    "notes": ["Kiwi.com combination of separate tickets"] if len(carriers) > 1 else []})
    return out


# ---------------------------------------------------------------- Google Flights (fli)

async def google(origins: list[str], destinations: list[str], day: str, adults: int = 1) -> list[dict]:
    text = await call_mcp(URLS["google"], "search_flights", {
        "origin": ",".join(origins), "destination": ",".join(destinations), "departure_date": day,
        "passengers": adults,
    })
    data = json.loads(text)
    if not data.get("success", True):
        raise EngineError(f"google: {str(data)[:200]}")
    out = []
    for f in data.get("flights") or []:
        segs = [_seg(airports.code_for_name(l.get("departure_airport_name") or l.get("departure_airport")),
                     airports.code_for_name(l.get("arrival_airport_name") or l.get("arrival_airport")),
                     l.get("departure_time"), l.get("arrival_time"), l.get("airline"),
                     f"{l.get('airline_code') or ''}{l.get('flight_number') or ''}" or None)
                for l in f.get("legs") or []]
        if not segs:
            continue
        url = (f"https://www.google.com/travel/flights?q=Flights%20from%20{segs[0]['from']}"
               f"%20to%20{segs[-1]['to']}%20on%20{day}")
        out.append({"source": "google", "price": f.get("price"), "currency": f.get("currency"),
                    "booking_url": url, "segments": segs, "self_transfer": None, "notes": []})
    return out


async def google_dates(origins: list[str], destinations: list[str], day: str, day_to: str,
                       adults: int = 1) -> list[tuple[str, float, str]]:
    """Cheapest one-way fare per departure day in [day, day_to]: [(YYYY-MM-DD, price, currency)],
    cheapest first. Used to pick which days to search in detail."""
    text = await call_mcp(URLS["google"], "search_dates", {
        "origin": ",".join(origins), "destination": ",".join(destinations), "start_date": day,
        "end_date": day_to, "is_round_trip": False, "passengers": adults,
    })
    data = json.loads(text)
    if not data.get("success", True):
        raise EngineError(f"google dates: {str(data)[:200]}")
    out = []
    for d in data.get("dates") or []:
        dd = d.get("date")
        dd = (dd[0] if isinstance(dd, list) and dd else dd) or ""
        if d.get("price") is not None and dd:
            out.append((dd[:10], float(d["price"]), d.get("currency") or "EUR"))
    return sorted(out, key=lambda x: (x[1], x[0]))


# ---------------------------------------------------------------- Skiplagged

_SK_SEG = re.compile(r"([A-Z]{3}) → ([A-Z]{3}) \((\S+ [\d:]+)\S* → (\S+ [\d:]+)\S*\)")


async def skiplagged(origin: str, destination: str, day: str, adults: int = 1) -> list[dict]:
    text = await call_mcp(URLS["skiplagged"], "sk_flights_search", {
        "origin": origin, "destination": destination, "departureDate": day, "adults": adults,
        "limit": 10, "includeVirtualInterlining": True, "includeHiddenCity": False,
    })
    if not text.lstrip().startswith("#"):
        raise EngineError(f"skiplagged: {text[:200]}")
    out = []
    for line in text.splitlines():
        if not line.startswith("| $"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 7:
            continue
        price = float(cols[0].lstrip("$").replace(",", ""))
        trip = re.search(r"#trip=([\w-]+)", cols[6])
        flights = trip.group(1).split("-") if trip else []
        segs = []
        for i, (frm, to, dep, arr) in enumerate(_SK_SEG.findall(cols[5])):
            segs.append(_seg(frm, to, dep.replace(" ", "T"), arr.replace(" ", "T"), None,
                             flights[i] if i < len(flights) else None))
        link = re.search(r"\]\((https?://[^)]+)\)", cols[6])
        virtual = "virtual" in cols[3].lower()
        out.append({"source": "skiplagged", "price": price, "currency": "USD",
                    "booking_url": link.group(1) if link else None, "segments": segs,
                    "self_transfer": virtual,
                    "notes": [f"airlines: {cols[4]}"] + (["virtual interline (separate tickets)"] if virtual else [])})
    return out


# ---------------------------------------------------------------- Duffel (flights server)

async def duffel(origin: str, destination: str, day: str, adults: int = 1) -> list[dict]:
    text = await call_mcp(URLS["duffel"], "search_flights", {"params": {
        "type": "one_way", "origin": origin, "destination": destination, "departure_date": day,
        "adults": adults,
    }})
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise EngineError(f"duffel: {text[:200]}") from e
    out = []
    for o in data.get("offers") or []:
        sl = (o.get("slices") or [{}])[0]
        if not sl:
            continue
        # Duffel's summary has connection airports/times but not per-segment flight numbers.
        stops = [sl.get("origin")] + [c.get("airport") for c in sl.get("connections") or []] + [sl.get("destination")]
        deps = [sl.get("departure")] + [c.get("departure") for c in sl.get("connections") or []]
        arrs = [c.get("arrival") for c in sl.get("connections") or []] + [sl.get("arrival")]
        segs = [_seg(stops[i], stops[i + 1], deps[i], arrs[i], sl.get("carrier"), None) for i in range(len(stops) - 1)]
        price = o.get("price") or {}
        out.append({"source": "duffel", "price": float(price.get("amount") or 0) or None,
                    "currency": price.get("currency"), "booking_url": None, "segments": segs,
                    "self_transfer": False, "notes": [f"Duffel offer {o.get('offer_id')}"]})
    return out


ENGINES = {"kiwi": kiwi, "google": google, "skiplagged": skiplagged, "duffel": duffel}


def enabled(name: str) -> bool:
    return bool(URLS.get(name))
