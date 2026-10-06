"""research_route: systematic search for hard-to-reach destinations.

Pipeline (one call, bounded time budget, one date window for every search):
  1. Expand airports: a country means all its scheduled airports (OurAirports);
     a city/airport adds nearby ones (a destination alternative implies a
     ground transfer).
  2. Whole-trip searches: Kiwi over the window (origins split into small groups —
     one Kiwi query returns only ~15 itineraries, so a long origin list hides
     cheaper ones), Google's cheapest-day calendar, then Google / Skiplagged /
     Duffel on the cheapest days.
  3. Hubs into the destination, from every source available: AeroDataBox route
     stats (current), OpenFlights routes (historical) and the connections the
     engines just returned. Each hub becomes an *area*: the hub plus scheduled
     airports within METRO_RADIUS_KM (DXB + SHJ, IST + SAW…), found by distance.
  4. Per area: origin→area and area→destination legs over the window, combined into
     self-transfer itineraries; switching airports inside an area costs the ground
     transfer's estimated time (added to the minimum connection) and cost.
  5. Rank by a cost score (price + estimated ground cost + time + risk), deterministically.
  6. Report coverage (what was queried), `unchecked` (searches that failed — not
     evidence of "no flights") and manual checks (last-leg airlines no engine priced).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from datetime import date as Date, datetime, timedelta

from flight_research import airports, borders, engines, fx, ground
from flight_research.aerodatabox import AeroDataBox

BUDGET_S = float(os.environ.get("RESEARCH_BUDGET_S", "150"))
CONCURRENCY = int(os.environ.get("RESEARCH_CONCURRENCY", "10"))  # per-engine caps: engines.py
MAX_WINDOW_DAYS = int(os.environ.get("RESEARCH_MAX_WINDOW_DAYS", "45"))
KIWI_CHUNK = int(os.environ.get("RESEARCH_KIWI_CHUNK", "4"))  # origins per whole-trip Kiwi query
# Origins per origin→hub Kiwi query: only the cheapest first legs matter, so bigger groups.
KIWI_LEG_CHUNK = int(os.environ.get("RESEARCH_KIWI_LEG_CHUNK", "8"))
GOOGLE_MAX_AIRPORTS = 5   # Google gets the busiest origins (one query, multi-airport)
DETAIL_DAYS = 2           # cheapest days searched by single-day engines
MAX_HUBS = int(os.environ.get("RESEARCH_MAX_HUBS", "5"))
METRO_RADIUS_KM = float(os.environ.get("RESEARCH_METRO_RADIUS_KM", "80"))
METRO_MAX = 3             # extra airports per hub area
MIN_CONNECT_H = 3      # self-transfer on separate tickets
MAX_CONNECT_H = 36
LEG_CANDIDATES = 12    # cheapest first legs per area kept for combination
PAIRS_PER_LEG = 2      # best second legs kept per first leg
# Score = price in EUR + estimated ground cost + penalties (EUR-equivalent).
PER_HOUR = 8.0
SELF_TRANSFER = 25.0
AIRPORT_CHANGE = 15.0  # self-transfer that also changes airport
PER_STOP = 10.0
BORDER_CROSSING = 80.0  # per land border on the ground leg (visas, queues, time)
# Ground legs across a border the memory reports closed/restricted (borders.py):
# kept, but ranked down. An island leg (no road link) is only flagged: it costs what
# one ordinary border crossing does, nothing extra.
CLOSED_BORDER = 300.0
RESTRICTED_BORDER = 100.0
NO_LAND_LINK = BORDER_CROSSING
# Ranked list diversity: at most this many options per (arrival airport, last hub).
PER_PATTERN = 2
# Destination alternatives (ground transfer) kept for searching, cleanest first.
MAX_DEST_ALTS = 6


def _parse(t: str | None) -> datetime | None:
    if not t:
        return None
    try:
        return datetime.fromisoformat(t[:19])
    except ValueError:
        return None


def _hours(segs: list[dict]) -> float | None:
    a, b = _parse(segs[0]["dep"]), _parse(segs[-1]["arr"])
    # Local times at different airports: approximate (time zones not normalized).
    return round((b - a).total_seconds() / 3600, 1) if a and b and b > a else None


def _key(segs: list[dict]) -> tuple:
    return tuple((s["from"], s["to"], s["flight"] or (s["dep"] or "")[:16]) for s in segs)


def _route(segs: list[dict]) -> str:
    """KRK-SHJ~DXB-KBL: `~` marks an airport change on the ground between flights."""
    out = segs[0]["from"]
    for prev, s in zip([None] + segs[:-1], segs):
        if prev and prev["to"] != s["from"]:
            out += f"~{s['from']}"
        out += f"-{s['to']}"
    return out


def _chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i:i + n] for i in range(0, len(items), max(1, n))]


class _Runner:
    """Runs engine calls under a semaphore and a shared deadline, logging coverage."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.coverage: list[dict] = []

    async def run(self, engine: str, label: str, coro_fn, *args, **kw) -> list:
        entry = {"engine": engine, "query": label, "ok": False, "results": 0}
        self.coverage.append(entry)
        if not engines.enabled(engine):
            entry["error"] = "engine not configured"
            return []
        t = time.monotonic()
        try:
            async with self.sem:
                left = self.deadline - time.monotonic()
                if left <= 1:
                    raise asyncio.TimeoutError
                res = await asyncio.wait_for(coro_fn(*args, **kw), timeout=left)
            entry.update(ok=True, results=len(res))
            return res
        except asyncio.TimeoutError:
            entry["error"] = "time budget exhausted"
        except Exception as e:  # engine failures are data, not fatal
            e = engines.root_cause(e)
            entry["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        finally:
            entry["ms"] = int((time.monotonic() - t) * 1000)
        return []


async def _gather_until(deadline: float, *coros) -> list:
    """Like gather, but returns at the deadline: unfinished calls count as empty and are
    cancelled without waiting (closing an MCP session to a hung engine can take long)."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    if not tasks:
        return []
    done, pending = await asyncio.wait(tasks, timeout=max(1.0, deadline - time.monotonic() + 1))
    for t in pending:
        t.cancel()
    return [t.result() if t in done and not t.cancelled() and t.exception() is None else [] for t in tasks]


def _blocked(g: dict) -> bool:
    return any(c["status"] in ("closed", "restricted") for c in g["crossings"])


def _ground_penalty(g: dict) -> float:
    st = [c["status"] for c in g["crossings"]]
    return (BORDER_CROSSING * len(st) + CLOSED_BORDER * st.count("closed")
            + RESTRICTED_BORDER * st.count("restricted") + NO_LAND_LINK * (not g["land_link"]))


def _window(day: str, date_to: str | None, flex_days: int) -> tuple[Date, Date, list[str]]:
    """[start, end] departure window: `date`..`date_to`, else `date` ± flex_days; never
    before today, at most MAX_WINDOW_DAYS long."""
    notes = []
    start = Date.fromisoformat(day)
    if date_to:
        end = Date.fromisoformat(date_to)
        if end < start:
            raise ValueError("date_to is before date")
    else:
        flex = max(0, int(flex_days or 0))
        start, end = start - timedelta(days=flex), start + timedelta(days=flex)
    today = Date.today()
    if end < today:
        raise ValueError("the whole date window is in the past")
    if start < today:
        start = today
    if (end - start).days + 1 > MAX_WINDOW_DAYS:
        end = start + timedelta(days=MAX_WINDOW_DAYS - 1)
        notes.append(f"date window capped to {MAX_WINDOW_DAYS} days ({start}..{end}); "
                     "call again for the rest of the range")
    return start, end, notes


async def research_route(origin: str, destination: str, day: str, flex_days: int = 2, adults: int = 1,
                         max_price_eur: float | None = None, dest_radius_km: float = 400,
                         origin_radius_km: float = 200, allow_ground: bool = True,
                         adb: AeroDataBox | None = None, date_to: str | None = None) -> dict:
    t0 = time.monotonic()
    adb = adb or AeroDataBox()
    calls_before = adb.calls
    start, end, notes = _window(day, date_to, flex_days)
    s_day, e_day = start.isoformat(), end.isoformat()
    span = e_day if end > start else None  # Kiwi: one query covers every day in the window
    # A country ("Poland", "PL") means: any of its scheduled airports, all equally fine.
    o_country, d_country = airports.country_airports(origin), airports.country_airports(destination)
    o_hits = o_country or airports.resolve(origin)
    d_hits = d_country or airports.resolve(destination)
    if not o_hits or not d_hits:
        return {"error": f"unknown {'origin' if not o_hits else 'destination'}: "
                         f"{origin if not o_hits else destination!r} (use an IATA code, city or country)"}
    O, D = o_hits[0], d_hits[0]
    if o_country:
        o_alts = [(a, 0.0) for a in o_country[1:]]
    else:
        o_alts = [(a, km) for a, km in airports.nearby(O, origin_radius_km) if a.type != "small_airport"][:3]
    memory = borders.known()
    ground_route = {}  # country -> borders.route() to D.country

    def border_route(country: str) -> dict:
        if country not in ground_route:
            ground_route[country] = borders.route(country, D.country, memory)
        return ground_route[country]

    if d_country:
        d_alts = [(a, 0.0) for a in d_country[1:]]
    elif allow_ground:
        # Nothing is dropped for its border: airports behind a reported closure go
        # after the clean ones; island legs keep their place (flagged on the options).
        near = airports.nearby(D, dest_radius_km)
        d_alts = sorted(near, key=lambda x: (_blocked(border_route(x[0].country)), x[1]))[:MAX_DEST_ALTS]
    else:
        d_alts = []
    O_set = [O.iata] + [a.iata for a, _ in o_alts]
    D_set = [D.iata] + [a.iata for a, _ in d_alts]
    O_google = O_set[:GOOGLE_MAX_AIRPORTS]
    ground_km = {D.iata: 0.0, **{a.iata: round(km) for a, km in d_alts}}
    runner = _Runner(t0 + BUDGET_S)
    rates_task = asyncio.ensure_future(fx.rates())

    # 1. Whole trip over the window (Kiwi, per origin group) + Google's day calendar
    #    + current route stats into the destination set (AeroDataBox), concurrently.
    def kiwi_whole(dests: list[str]) -> list:
        return [runner.run("kiwi", f"{','.join(c)}→{','.join(dests)} {s_day}..{e_day}", engines.kiwi,
                           c, dests, s_day, 0, adults, True, span) for c in _chunks(O_set, KIWI_CHUNK)]

    async def adb_routes(code: str):
        try:
            return code, await adb.routes(code), None
        except Exception as e:
            return code, [], f"{type(e).__name__}: {str(e)[:160]}"

    whole_main = kiwi_whole([D.iata])
    whole_alts = kiwi_whole(D_set[1:]) if len(D_set) > 1 else []
    cal = (runner.run("google", f"{','.join(O_google)}→{D.iata} cheapest days {s_day}..{e_day}",
                      engines.google_dates, O_google, [D.iata], s_day, e_day, adults)
           if span else asyncio.sleep(0, result=[]))
    adb_jobs = [adb_routes(c) for c in D_set] if adb.enabled else []
    res1 = await _gather_until(runner.deadline, *whole_main, *whole_alts, cal, *adb_jobs)
    adb_res = [r for r in res1[len(whole_main) + len(whole_alts) + 1:] if r]
    n_w = len(whole_main) + len(whole_alts)
    whole_res = [it for r in res1[:n_w] for it in r]
    calendar = res1[n_w]

    # Days worth a detailed single-day search: Google's cheapest, else Kiwi's cheapest.
    best_days = [d for d, _, _ in calendar[:DETAIL_DAYS]]
    if not best_days:
        kiwi_days = sorted((it["price"] or 1e9, (it["segments"][0]["dep"] or "")[:10]) for it in whole_res
                           if it["segments"])
        best_days = list(dict.fromkeys(d for _, d in kiwi_days if s_day <= d <= e_day))[:DETAIL_DAYS] or [s_day]

    # 2. Hubs into the destination set, from every source, merged into areas.
    cand: dict[str, dict] = {}

    def add_hub(h: str, serves: str, airlines=(), weight: float = 0.0, source: str = "") -> None:
        if not h or h in D_set or h in O_set or airports.get(h) is None:
            return
        e = cand.setdefault(h, {"iata": h, "score": 0.0, "daily_flights": 0.0, "airlines": set(),
                                "serves": set(), "sources": set()})
        if not (source.startswith("openflights") and source in e["sources"]):  # historical: once per hub
            e["score"] += weight
        e["airlines"].update(a for a in airlines if a)
        e["serves"].add(serves)
        e["sources"].add(source)

    for code, rts, err in adb_res:
        runner.coverage.append({"engine": "aerodatabox", "query": f"routes {code}", "ok": err is None,
                                "results": len(rts), **({"error": err} if err else {})})
        for r in rts:
            add_hub(r["iata"], code, r["airlines"], r["daily_flights"], "aerodatabox")
            if r["iata"] in cand:
                cand[r["iata"]]["daily_flights"] += r["daily_flights"]
    if not adb.enabled:
        notes.append("AERODATABOX_KEY not set: hubs from historical route data + engine connections.")
    for code in D_set:
        for src, als in airports.routes_into(code).items():
            add_hub(src, code, (), 1.0, "openflights (historical)")  # old data: a tie-breaker
    for it in whole_res:
        segs = it["segments"]
        if len(segs) > 1 and segs[-1]["to"] in D_set:
            add_hub(segs[-1]["from"], segs[-1]["to"], (), 3.0, "engine results")
    areas: list[dict] = []
    for h in sorted(cand.values(), key=lambda h: (-h["score"], h["iata"])):
        home = next((a for a in areas if h["iata"] in a["airports"]), None)
        if home:  # already covered by a bigger hub's area: merge the evidence
            home["airlines"] |= h["airlines"]
            home["serves"] |= h["serves"]
            home["sources"] |= h["sources"]
            continue
        if len(areas) >= MAX_HUBS:
            continue
        ap = airports.get(h["iata"])
        metro = [a.iata for a, _ in airports.nearby(ap, METRO_RADIUS_KM)
                 if a.type != "small_airport" and a.iata not in O_set and a.iata not in D_set][:METRO_MAX]
        areas.append({**h, "airports": [h["iata"]] + metro, "city": ap.city, "country": ap.country,
                      "airlines": set(h["airlines"]), "serves": set(h["serves"]), "sources": set(h["sources"])})

    # 3. Detailed single-day searches + per-area legs over the window, concurrently.
    detail = []
    for d in best_days:
        detail.append(runner.run("google", f"{','.join(O_google)}→{D.iata} {d}", engines.google, O_google, [D.iata], d, adults))
    detail += [
        runner.run("skiplagged", f"{O.iata}→{D.iata} {best_days[0]}", engines.skiplagged, O.iata, D.iata, best_days[0], adults),
        runner.run("duffel", f"{O.iata}→{D.iata} {best_days[0]}", engines.duffel, O.iata, D.iata, best_days[0], adults),
    ]
    if len(D_set) > 1:
        detail.append(runner.run("google", f"{','.join(O_google)}→{','.join(D_set[1:])} {best_days[0]}",
                                 engines.google, O_google, D_set[1:], best_days[0], adults))
    leg_end = (end + timedelta(days=2)).isoformat()
    nxt = (Date.fromisoformat(best_days[0]) + timedelta(days=1)).isoformat()
    # Queue order is priority order (engine semaphores are FIFO): when the time budget
    # runs out, what's left unsearched should be the least useful part. First the
    # hub→destination legs (one per area, without them no combination exists), then
    # origin→hub legs round-robin over the areas (busiest origin group first), Google last.
    legs1, legs2 = [], []
    for i, a in enumerate(areas):
        aps = a["airports"]
        legs2.append((i, runner.run("kiwi", f"{','.join(aps)}→{','.join(D_set)} {s_day}..{leg_end}", engines.kiwi,
                                    aps, D_set, s_day, 0, adults, False, leg_end)))
    for c in _chunks(O_set, KIWI_LEG_CHUNK):
        for i, a in enumerate(areas):
            aps = a["airports"]
            legs1.append((i, runner.run("kiwi", f"{','.join(c)}→{','.join(aps)} {s_day}..{e_day}", engines.kiwi,
                                        c, aps, s_day, 0, adults, False, span)))
    for i, a in enumerate(areas):
        aps = a["airports"]
        legs2.append((i, runner.run("google", f"{aps[0]}→{','.join(D_set[:3])} {nxt}", engines.google,
                                    [aps[0]], D_set[:3], nxt, adults)))
        legs1.append((i, runner.run("google", f"{','.join(O_google)}→{aps[0]} {best_days[0]}", engines.google,
                                    O_google, [aps[0]], best_days[0], adults)))
    # Coroutines start in argument order: legs2's Kiwi part first (see above).
    res2 = await _gather_until(runner.deadline, *detail, *(c for _, c in legs2), *(c for _, c in legs1))
    whole_res += [it for r in res2[:len(detail)] for it in r]
    leg2_res, leg1_res = res2[len(detail):len(detail) + len(legs2)], res2[len(detail) + len(legs2):]
    rates = await rates_task
    usd = rates.get("USD")

    def eur(it: dict) -> float | None:
        return fx.to_eur(it.get("price"), it.get("currency"), rates)

    # 4. Combine area legs into self-transfer itineraries.
    by1, by2 = defaultdict(list), defaultdict(list)
    for (i, _), res in zip(legs1, leg1_res):
        by1[i] += [it for it in res if it["segments"] and it["segments"][-1]["to"] in areas[i]["airports"]]
    for (i, _), res in zip(legs2, leg2_res):
        by2[i] += [it for it in res if it["segments"] and it["segments"][0]["from"] in areas[i]["airports"]
                   and it["segments"][-1]["to"] in D_set]
    transfer_cache: dict[tuple, dict] = {}

    def transfer(x: str, y: str) -> dict | None:
        if x == y:
            return None
        if (x, y) not in transfer_cache:
            ax, ay = airports.get(x), airports.get(y)
            transfer_cache[(x, y)] = {"from": x, "to": y, "kind": "airport change",
                                      **ground.estimate(airports.haversine_km(ax, ay), ay.country, usd)}
        return transfer_cache[(x, y)]

    combos = []
    for i in by1:
        a_list = sorted((it for it in by1[i] if eur(it)), key=lambda it: (eur(it), _key(it["segments"])))[:LEG_CANDIDATES]
        b_list = sorted((it for it in by2[i] if eur(it)), key=lambda it: (eur(it), _key(it["segments"])))
        for a in a_list:
            arr = _parse(a["segments"][-1]["arr"])
            kept = 0
            for b in b_list:
                dep = _parse(b["segments"][0]["dep"])
                tr = transfer(a["segments"][-1]["to"], b["segments"][0]["from"])
                need = MIN_CONNECT_H + (tr["hours"] if tr else 0)
                if not arr or not dep or not (timedelta(hours=need) <= dep - arr <= timedelta(hours=MAX_CONNECT_H)):
                    continue
                layover = round((dep - arr).total_seconds() / 3600, 1)
                where = (f"{tr['from']}→{tr['to']} (~{tr['road_km']} km, ~{tr['hours']} h, ~{tr['cost_eur']} EUR "
                         f"est.)" if tr else a["segments"][-1]["to"])
                combos.append({
                    "source": f"{a['source']}+{b['source']}", "price": round(eur(a) + eur(b), 2), "currency": "EUR",
                    "booking_url": None, "booking_urls": [a.get("booking_url"), b.get("booking_url")],
                    "segments": a["segments"] + b["segments"], "self_transfer": True,
                    "transfers": [tr] if tr else [],
                    "notes": [f"separate tickets, self-transfer {where}, {layover} h between flights"] + a["notes"] + b["notes"],
                })
                kept += 1
                if kept >= PAIRS_PER_LEG:
                    break

    # 5. Normalize, dedupe, filter to the window, score, rank.
    options, seen = [], set()
    for it in whole_res + combos:
        segs = it["segments"]
        price_eur = eur(it)
        if not segs or price_eur is None or (max_price_eur and price_eur > max_price_eur):
            continue
        if not (s_day <= (segs[0]["dep"] or "")[:10] <= e_day):
            continue
        k = _key(segs)
        if k in seen:
            continue
        seen.add(k)
        final = segs[-1]["to"]
        land = airports.get(final)
        gkm = ground_km.get(final)
        if gkm is None:  # an engine landed somewhere else entirely (e.g. city codes)
            gkm = round(airports.haversine_km(land, D)) if land else None
        hrs = _hours(segs)
        border = bool(gkm) and land is not None and land.country != D.country
        g = border_route(land.country) if border else None
        warn = borders.warnings(g) if g else []
        legs = list(it.get("transfers") or [])
        if gkm and land:
            legs.append({"from": final, "to": D.iata, "kind": "to destination",
                         **ground.estimate(gkm, land.country, usd)})
        g_cost = sum(x["cost_eur"] for x in legs)
        g_hours = sum(x["hours"] for x in legs if x["kind"] == "to destination")  # transfers are inside hrs
        score = (price_eur + g_cost + PER_HOUR * ((hrs or 24) + g_hours) + SELF_TRANSFER * bool(it["self_transfer"])
                 + AIRPORT_CHANGE * bool(it.get("transfers")) + PER_STOP * (len(segs) - 1)
                 + (_ground_penalty(g) if g else 0))
        extra = []
        if gkm:
            via = "→".join(g["path"]) if g and g["land_link"] else ""
            how = ("away, across the sea," if g and g.get("islands") else "away (no road route found)"
                   if g and not g["land_link"] else "by ground")
            extra.append(f"lands in {final} ({land.city if land else '?'}, {land.country if land else '?'}), "
                         f"~{gkm} km {how} to {D.iata}" + (f" incl. border crossing {via}" if via else ""))
        extra += warn
        options.append({
            "price_eur": price_eur, "price": it["price"], "currency": it["currency"], "source": it["source"],
            "ground_cost_eur_est": g_cost, "total_eur_est": round(price_eur + g_cost),
            "route": _route(segs),
            "arrives_at": final, "ground_km_to_destination": gkm, "self_transfer": it["self_transfer"],
            "ground_legs": legs, "total_hours_approx": hrs, "stops": len(segs) - 1, "segments": segs,
            "booking_urls": [u for u in it.get("booking_urls") or [it.get("booking_url")] if u],
            "notes": extra + it["notes"], "border_crossing": border,
            "border_crossings": [c["border"] for c in g["crossings"]] if g else [],
            "land_link": g["land_link"] if g else True, "island_leg": bool(g and g.get("islands")),
            "border_warnings": warn,
            "_unverified": [c["border"] for c in g["crossings"] if c["status"] is None] if g else [],
            "_score": round(score, 2),
        })
    options.sort(key=lambda o: (o["_score"], o["price_eur"], o["route"]))
    # Diversity: cap options per (arrival airport, last connection) so one cheap
    # pattern doesn't fill the whole list; the rest follow in score order.
    per_pattern: dict[tuple, int] = defaultdict(int)
    head, tail = [], []
    for o in options:
        pat = (o["arrives_at"], o["segments"][-1]["from"])
        per_pattern[pat] += 1
        (head if per_pattern[pat] <= PER_PATTERN else tail).append(o)
    options = head + tail
    unverified: set[str] = set()
    for i, o in enumerate(options, 1):
        o["rank"] = i
        o.pop("_score")
        if i <= 10:
            unverified.update(o.pop("_unverified"))
        else:
            o.pop("_unverified")
    cheapest = sorted(options, key=lambda o: (o["total_eur_est"], o["price_eur"], o["rank"]))[:3]

    # 6. Last-leg airlines nobody priced -> check their own websites. Only current
    # route data names airlines (AeroDataBox); historical routes carry no airlines here.
    priced = " ".join(f"{s.get('carrier') or ''} {s.get('flight') or ''}" for o in options
                      for s in o["segments"]).lower()
    manual = []
    for a in areas:
        for al in sorted(a["airlines"]):
            if al.lower() not in priced:
                manual.append({"airline": al, "from_hub": a["iata"], "to": sorted(a["serves"]),
                               "hint": "not priced by any engine; check the airline's website (or Playwright)"})
    for c in runner.coverage:  # cut off at the deadline while still running
        if not c["ok"] and "error" not in c:
            c["error"] = "time budget exhausted"
    unchecked = [f"{c['engine']} {c['query']}: {c['error']}" for c in runner.coverage
                 if not c["ok"] and c.get("error") != "engine not configured" and c["engine"] != "aerodatabox"]
    if unchecked:
        notes.append("Some searches failed (see `unchecked`): for those routes say 'could not be checked', "
                     "never 'no flights'.")
    if not options:
        notes.append("No priced options: widen dest_radius_km, widen the date window, or check manual_checks "
                     "carriers directly.")

    def line(o: dict, prefix: str = "") -> str:
        segs = o["segments"]
        dep, arr = (segs[0]["dep"] or "")[:16].replace("T", " "), (segs[-1]["arr"] or "")[:16].replace("T", " ")
        km = o["ground_km_to_destination"]
        gr = (f"+{km} km across the sea, island: check ferry" if o["island_leg"]
              else f"+{km} km, NO road route found: verify" if not o["land_link"]
              else f"+{km} km ground + border WARNING" if o["border_warnings"]
              else f"+{km} km ground + border" if o["border_crossing"] else f"+{km} km ground")
        flags = (["self-transfer"] if o["self_transfer"] else [])
        flags += [f"airport change {t['from']}→{t['to']}" for t in o["ground_legs"] if t["kind"] == "airport change"]
        flags += [gr] if km else []
        cost = (f" (+~{o['ground_cost_eur_est']:.0f} EUR ground est. = ~{o['total_eur_est']} EUR)"
                if o["ground_cost_eur_est"] else "")
        # Separate tickets: every ticket's link, else the itinerary reads as one booking.
        link = " + ".join(o["booking_urls"]) if len(o["booking_urls"]) > 1 else (o["booking_urls"] or ["-"])[0]
        return (f"{prefix}#{o['rank']} {o['price_eur']:.0f} EUR{cost} | {o['route']} | {dep} → {arr} | "
                f"{o['stops']} stop(s){', ' + ', '.join(flags) if flags else ''} | {o['source']} | {link}")

    return {
        # Compact digest first: survives truncation and is what an agent should read.
        "summary": ([line(o) for o in options[:10]]
                    + [line(o, "CHEAPEST: ") for o in cheapest if o["rank"] > 10]) or ["no priced options"],
        "ranking": "by value: price + estimated ground cost + time + risk; CHEAPEST lines = lowest total",
        "query": {"origin": origin, "destination": destination, "window": [s_day, e_day], "adults": adults},
        "origin": O.brief(), "destination": D.brief(),
        "origin_alternatives": [{**a.brief(), "km": round(km)} for a, km in o_alts],
        "destination_alternatives": [{**a.brief(), "km": round(km),
                                      **({"border_warnings": w} if (w := borders.warnings(border_route(a.country))) else {})}
                                     for a, km in d_alts],
        # Remembered border closures/restrictions and island legs among the shown options.
        "border_warnings": list(dict.fromkeys(w for o in options[:10] for w in o["border_warnings"])),
        **({"unverified_borders": {
            "borders": sorted(unverified),
            "hint": "no remembered status for these land borders on the ground legs above: if the answer "
                    "depends on them, web-search whether they are open and record the finding with the "
                    "flight-research report_border_status tool (with the source URL)"}} if unverified else {}),
        "hubs": [{"iata": a["iata"], "area": a["airports"], "city": a["city"], "country": a["country"],
                  "sources": sorted(a["sources"]), "airlines": sorted(a["airlines"]), "serves": sorted(a["serves"]),
                  **({"daily_flights": round(a["daily_flights"], 1)} if a["daily_flights"] else {})} for a in areas],
        "searched_days": best_days,
        "options": options[:10] + [o for o in cheapest if o["rank"] > 10], "options_total": len(options),
        "unchecked": unchecked,
        "coverage": runner.coverage, "manual_checks": manual[:12], "notes": notes,
        "elapsed_s": round(time.monotonic() - t0, 1), "aerodatabox_calls": adb.calls - calls_before,
    }
