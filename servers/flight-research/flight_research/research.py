"""research_route: systematic search for hard-to-reach destinations.

Pipeline (one call, bounded time budget):
  1. Expand airports: nearby scheduled-service airports around origin and
     destination (a destination alternative implies a ground transfer).
  2. Find hubs: which airports actually fly into the destination set
     (AeroDataBox route stats), busiest first.
  3. Query engines concurrently: whole-trip searches (Kiwi with self-transfer,
     Google, Skiplagged, Duffel) plus per-hub legs origin→hub and hub→destination.
  4. Assemble hub legs into self-transfer combinations (min connection time).
  5. Rank by a cost score (price + time + risk penalties), deterministically.
  6. Report coverage (what was queried, what failed) and manual checks:
     airlines flying the last leg that no engine priced — candidates for
     checking on the carrier's own website.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from datetime import date as Date, datetime, timedelta

from flight_research import airports, engines, fx
from flight_research.aerodatabox import AeroDataBox

BUDGET_S = float(os.environ.get("RESEARCH_BUDGET_S", "75"))
CONCURRENCY = int(os.environ.get("RESEARCH_CONCURRENCY", "6"))
MAX_HUBS = 5
MIN_CONNECT_H = 3      # self-transfer on separate tickets
MAX_CONNECT_H = 36
LEG_CANDIDATES = 6     # cheapest legs per side kept for combination
# Score = price in EUR + penalties (EUR-equivalent).
PER_HOUR = 8.0
SELF_TRANSFER = 25.0
PER_STOP = 10.0
PER_100KM_GROUND = 35.0
BORDER_CROSSING = 80.0  # ground leg into another country (visas, closures, time)
# Ranked list diversity: at most this many options per (arrival airport, last hub).
PER_PATTERN = 2


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


class _Runner:
    """Runs engine calls under a semaphore and a shared deadline, logging coverage."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.coverage: list[dict] = []

    async def run(self, engine: str, label: str, coro_fn, *args, **kw) -> list[dict]:
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
            while isinstance(e, BaseExceptionGroup) and e.exceptions:  # anyio TaskGroup wrapping
                e = e.exceptions[0]
            entry["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        finally:
            entry["ms"] = int((time.monotonic() - t) * 1000)
        return []


async def research_route(origin: str, destination: str, day: str, flex_days: int = 2, adults: int = 1,
                         max_price_eur: float | None = None, dest_radius_km: float = 400,
                         origin_radius_km: float = 200, allow_ground: bool = True,
                         adb: AeroDataBox | None = None) -> dict:
    t0 = time.monotonic()
    adb = adb or AeroDataBox()
    calls_before = adb.calls
    Date.fromisoformat(day)  # validate early
    o_hits, d_hits = airports.resolve(origin), airports.resolve(destination)
    if not o_hits or not d_hits:
        return {"error": f"unknown {'origin' if not o_hits else 'destination'}: {origin if not o_hits else destination!r}"}
    O, D = o_hits[0], d_hits[0]
    o_alts = [(a, km) for a, km in airports.nearby(O, origin_radius_km) if a.type != "small_airport"][:3]
    d_alts = airports.nearby(D, dest_radius_km)[:6] if allow_ground else []
    O_set = [O.iata] + [a.iata for a, _ in o_alts]
    D_set = [D.iata] + [a.iata for a, _ in d_alts]
    ground_km = {D.iata: 0.0, **{a.iata: round(km) for a, km in d_alts}}
    runner = _Runner(t0 + BUDGET_S)
    notes: list[str] = []

    # 2. Hubs flying into the destination set.
    hubs: dict[str, dict] = {}
    if adb.enabled:
        async def routes(code: str):
            try:
                return code, await adb.routes(code), None
            except Exception as e:
                return code, [], f"{type(e).__name__}: {str(e)[:160]}"
        for code, rts, err in await asyncio.gather(*(routes(c) for c in D_set)):
            runner.coverage.append({"engine": "aerodatabox", "query": f"routes {code}", "ok": err is None,
                                    "results": len(rts), **({"error": err} if err else {})})
            for r in rts:
                h = r["iata"]
                if not h or h in D_set or h in O_set:
                    continue
                e = hubs.setdefault(h, {"iata": h, "city": r["city"], "country": r["country"],
                                        "daily_flights": 0.0, "airlines": set(), "serves": set()})
                e["daily_flights"] += r["daily_flights"]
                e["airlines"].update(r["airlines"])
                e["serves"].add(code)
    else:
        notes.append("AERODATABOX_KEY not set: hubs inferred from engine connections; no manual_checks.")
    top_hubs = sorted(hubs.values(), key=lambda h: (-h["daily_flights"], h["iata"]))[:MAX_HUBS]

    # 3. Engine queries (whole trip + per-hub legs), all concurrent.
    nxt = (Date.fromisoformat(day) + timedelta(days=1)).isoformat()
    whole = [
        runner.run("kiwi", f"{','.join(O_set)}→{D.iata} ±{flex_days}d", engines.kiwi, O_set, [D.iata], day, flex_days, adults, True),
        runner.run("google", f"{O.iata}→{D.iata}", engines.google, [O.iata], [D.iata], day, adults),
        runner.run("skiplagged", f"{O.iata}→{D.iata}", engines.skiplagged, O.iata, D.iata, day, adults),
        runner.run("duffel", f"{O.iata}→{D.iata}", engines.duffel, O.iata, D.iata, day, adults),
    ]
    if len(D_set) > 1:
        whole += [
            runner.run("kiwi", f"{','.join(O_set)}→{','.join(D_set[1:])} ±{flex_days}d", engines.kiwi, O_set, D_set[1:], day, flex_days, adults, True),
            runner.run("google", f"{','.join(O_set)}→{','.join(D_set[1:])}", engines.google, O_set, D_set[1:], day, adults),
        ]
    whole_results = await asyncio.gather(*whole) if not top_hubs else None
    if whole_results is not None:
        # No route data: the last connection airport before the destination set in
        # the whole-trip results is a hub that demonstrably serves it.
        seen_hubs: dict[str, dict] = {}
        for it in (it for r in whole_results for it in r):
            segs = it["segments"]
            if len(segs) > 1 and segs[-1]["to"] in D_set and segs[-1]["from"] not in O_set:
                h = seen_hubs.setdefault(segs[-1]["from"], {"iata": segs[-1]["from"], "city": None, "country": None,
                                                            "daily_flights": 0.0, "airlines": set(), "serves": set()})
                h["daily_flights"] += 1  # here: number of itineraries using it
                h["serves"].add(segs[-1]["to"])
        top_hubs = sorted(seen_hubs.values(), key=lambda h: (-h["daily_flights"], h["iata"]))[:3]
    legs1, legs2 = [], []
    for h in top_hubs:
        serves = sorted(h["serves"])
        legs1.append((h["iata"], runner.run("kiwi", f"{','.join(O_set)}→{h['iata']}", engines.kiwi, O_set, [h["iata"]], day, 0, adults)))
        legs1.append((h["iata"], runner.run("google", f"{O.iata}→{h['iata']}", engines.google, [O.iata], [h["iata"]], day, adults)))
        legs2.append((h["iata"], runner.run("kiwi", f"{h['iata']}→{','.join(serves)} +1d", engines.kiwi, [h["iata"]], serves, nxt, 1, adults)))
        legs2.append((h["iata"], runner.run("google", f"{h['iata']}→{','.join(serves)}", engines.google, [h["iata"]], serves, day, adults)))
    if whole_results is None:
        results = await asyncio.gather(*whole, *(c for _, c in legs1), *(c for _, c in legs2))
        whole_results, results = results[:len(whole)], results[len(whole):]
    else:
        results = await asyncio.gather(*(c for _, c in legs1), *(c for _, c in legs2))
    whole_res = [it for r in whole_results for it in r]
    leg1_res, leg2_res = results[:len(legs1)], results[len(legs1):]

    rates = await fx.rates()

    def eur(it: dict) -> float | None:
        return fx.to_eur(it.get("price"), it.get("currency"), rates)

    # 4. Combine hub legs into self-transfer itineraries.
    by_hub1, by_hub2 = defaultdict(list), defaultdict(list)
    for (h, _), res in zip(legs1, leg1_res):
        by_hub1[h] += [it for it in res if it["segments"] and it["segments"][-1]["to"] == h]
    for (h, _), res in zip(legs2, leg2_res):
        by_hub2[h] += [it for it in res if it["segments"] and it["segments"][0]["from"] == h]
    combos = []
    for h in by_hub1:
        a_list = sorted((it for it in by_hub1[h] if eur(it)), key=lambda it: (eur(it), _key(it["segments"])))[:LEG_CANDIDATES]
        b_list = sorted((it for it in by_hub2[h] if eur(it)), key=lambda it: (eur(it), _key(it["segments"])))[:LEG_CANDIDATES]
        for a in a_list:
            arr = _parse(a["segments"][-1]["arr"])
            for b in b_list:
                dep = _parse(b["segments"][0]["dep"])
                if not arr or not dep or not (timedelta(hours=MIN_CONNECT_H) <= dep - arr <= timedelta(hours=MAX_CONNECT_H)):
                    continue
                layover = round((dep - arr).total_seconds() / 3600, 1)
                combos.append({
                    "source": f"{a['source']}+{b['source']}", "price": round(eur(a) + eur(b), 2), "currency": "EUR",
                    "booking_url": None, "booking_urls": [a.get("booking_url"), b.get("booking_url")],
                    "segments": a["segments"] + b["segments"], "self_transfer": True,
                    "notes": [f"separate tickets, self-transfer in {h} ({layover} h)"] + a["notes"] + b["notes"],
                })

    # 5. Normalize, dedupe, score, rank.
    options, seen = [], set()
    for it in whole_res + combos:
        segs = it["segments"]
        price_eur = eur(it)
        if not segs or price_eur is None or (max_price_eur and price_eur > max_price_eur):
            continue
        k = _key(segs)
        if k in seen:
            continue
        seen.add(k)
        final = segs[-1]["to"]
        gkm = ground_km.get(final)
        if gkm is None:  # an engine landed somewhere else entirely (e.g. city codes)
            ap = airports.get(final)
            gkm = round(airports.haversine_km(ap, D)) if ap else None
        hrs = _hours(segs)
        land = airports.get(final)
        border = bool(gkm) and land is not None and land.country != D.country
        score = (price_eur + PER_HOUR * (hrs or 24) + SELF_TRANSFER * bool(it["self_transfer"])
                 + PER_STOP * (len(segs) - 1) + PER_100KM_GROUND * ((gkm or 0) / 100)
                 + BORDER_CROSSING * border)
        extra = []
        if gkm:
            extra.append(f"lands in {final} ({land.city if land else '?'}, {land.country if land else '?'}), "
                         f"~{gkm} km by ground to {D.iata}" + (f" incl. border crossing {land.country}→{D.country}" if border else ""))
        options.append({
            "price_eur": price_eur, "price": it["price"], "currency": it["currency"], "source": it["source"],
            "route": "-".join([segs[0]["from"]] + [s["to"] for s in segs]),
            "arrives_at": final, "ground_km_to_destination": gkm, "self_transfer": it["self_transfer"],
            "total_hours_approx": hrs, "stops": len(segs) - 1, "segments": segs,
            "booking_urls": [u for u in it.get("booking_urls") or [it.get("booking_url")] if u],
            "notes": extra + it["notes"], "border_crossing": border, "_score": round(score, 2),
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
    for i, o in enumerate(options, 1):
        o["rank"] = i
        o.pop("_score")

    # 6. Last-leg airlines nobody priced -> check their own websites.
    priced = " ".join(str(s.get("carrier") or "") for o in options for s in o["segments"]).lower()
    manual = []
    for h in top_hubs:
        for al in sorted(h["airlines"]):
            if al.lower() not in priced:
                manual.append({"airline": al, "from_hub": h["iata"], "to": sorted(h["serves"]),
                               "hint": "not priced by any engine; check the airline's website (or Playwright)"})
    if not options:
        notes.append("No priced options: widen dest_radius_km, add flex_days, or check manual_checks carriers directly.")

    def line(o: dict) -> str:
        segs = o["segments"]
        dep, arr = (segs[0]["dep"] or "")[:16].replace("T", " "), (segs[-1]["arr"] or "")[:16].replace("T", " ")
        flags = (["self-transfer"] if o["self_transfer"] else []) + (
            [f"+{o['ground_km_to_destination']} km ground{' + border' if o['border_crossing'] else ''}"]
            if o["ground_km_to_destination"] else [])
        link = (o["booking_urls"] or ["-"])[0]
        return (f"#{o['rank']} {o['price_eur']:.0f} EUR | {o['route']} | {dep} → {arr} | "
                f"{o['stops']} stop(s){', ' + ', '.join(flags) if flags else ''} | {o['source']} | {link}")

    return {
        # Compact digest first: survives truncation and is what an agent should read.
        "summary": [line(o) for o in options[:10]] or ["no priced options"],
        "query": {"origin": origin, "destination": destination, "date": day, "flex_days": flex_days, "adults": adults},
        "origin": O.brief(), "destination": D.brief(),
        "origin_alternatives": [{**a.brief(), "km": round(km)} for a, km in o_alts],
        "destination_alternatives": [{**a.brief(), "km": round(km)} for a, km in d_alts],
        "hubs": [{**h, "airlines": sorted(h["airlines"]), "serves": sorted(h["serves"]),
                  "daily_flights": round(h["daily_flights"], 1)} for h in top_hubs],
        "options": options[:10], "options_total": len(options),
        "coverage": runner.coverage, "manual_checks": manual[:12], "notes": notes,
        "elapsed_s": round(time.monotonic() - t0, 1), "aerodatabox_calls": adb.calls - calls_before,
    }
