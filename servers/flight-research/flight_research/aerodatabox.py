"""AeroDataBox (RapidAPI): which routes an airport serves, and its daily schedule.

This answers "what actually flies into airport X" — the step fare engines can't
do. Responses are cached (routes 7 days, schedules 6 hours) to stay inside the
free tier; every result reports how many API calls it spent.
"""

from __future__ import annotations

import os
from datetime import date as Date, datetime, timedelta

import httpx

from flight_research import cache

HOST = os.environ.get("AERODATABOX_HOST", "aerodatabox.p.rapidapi.com")
ROUTES_TTL = 7 * 24 * 3600
SCHEDULE_TTL = 6 * 3600


class AeroDataBoxError(RuntimeError):
    pass


class AeroDataBox:
    def __init__(self, key: str | None = None) -> None:
        self.key = key if key is not None else os.environ.get("AERODATABOX_KEY", "")
        self.calls = 0  # real (uncached) API calls since start

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    async def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.enabled:
            raise AeroDataBoxError("AERODATABOX_KEY is not configured")
        self.calls += 1
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"https://{HOST}{path}", params=params,
                            headers={"X-RapidAPI-Key": self.key, "X-RapidAPI-Host": HOST})
        if r.status_code == 204:
            return {}
        if r.status_code >= 400:
            raise AeroDataBoxError(f"AeroDataBox {r.status_code}: {r.text[:200]}")
        return r.json()

    async def routes(self, iata: str) -> list[dict]:
        """Destinations served from `iata` (≈ origins flying into it), busiest first."""
        iata = iata.upper()
        key = f"adb:routes:{iata}"
        if (hit := cache.get(key)) is not None:
            return hit
        data = await self._get(f"/airports/iata/{iata}/stats/routes/daily")
        out = []
        for r in data.get("routes") or []:
            d = r.get("destination") or {}
            out.append({
                "iata": d.get("iata") or d.get("icao"),
                "name": d.get("name"),
                "city": d.get("municipalityName"),
                "country": d.get("countryCode"),
                "daily_flights": round(float(r.get("averageDailyFlights") or 0), 2),
                "airlines": sorted({o.get("name") for o in r.get("operators") or [] if o.get("name")}),
            })
        out.sort(key=lambda x: (-x["daily_flights"], x["iata"] or ""))
        cache.put(key, out, ROUTES_TTL)
        return out

    async def schedule(self, iata: str, day: str, direction: str = "Both") -> dict:
        """Scheduled departures/arrivals for a local date (two 12-hour windows)."""
        iata = iata.upper()
        direction = {"departures": "Departure", "arrivals": "Arrival"}.get(direction.lower(), "Both")
        key = f"adb:fids:{iata}:{day}:{direction}"
        if (hit := cache.get(key)) is not None:
            return hit
        d0 = datetime.combine(Date.fromisoformat(day), datetime.min.time())
        out: dict[str, list] = {"departures": [], "arrivals": []}
        for start in (d0, d0 + timedelta(hours=12)):
            end = start + timedelta(hours=11, minutes=59)
            data = await self._get(
                f"/flights/airports/iata/{iata}/{start:%Y-%m-%dT%H:%M}/{end:%Y-%m-%dT%H:%M}",
                {"direction": direction, "withLeg": "true", "withCancelled": "false",
                 "withCodeshared": "false", "withCargo": "false", "withPrivate": "false"},
            )
            for kind in ("departures", "arrivals"):
                for f in data.get(kind) or []:
                    other = f.get("arrival" if kind == "departures" else "departure") or {}
                    here = f.get("departure" if kind == "departures" else "arrival") or f.get("movement") or {}
                    out[kind].append({
                        "flight": f.get("number"),
                        "airline": (f.get("airline") or {}).get("name"),
                        "other_airport": (other.get("airport") or {}).get("iata"),
                        "other_city": (other.get("airport") or {}).get("municipalityName"),
                        "time_local": ((here.get("scheduledTime") or {}).get("local")),
                        "status": f.get("status"),
                    })
        for kind in out:
            out[kind].sort(key=lambda x: (x["time_local"] or "", x["flight"] or ""))
        cache.put(key, out, SCHEDULE_TTL)
        return out
