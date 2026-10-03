"""Currency conversion to EUR so prices from different engines are comparable
(Kiwi: EUR, Google Flights: locale currency, Skiplagged: USD). ECB daily
reference rates, cached for a day; static fallback if the ECB is unreachable."""

from __future__ import annotations

import re

import httpx

from flight_research import cache

_ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
# Units of currency per 1 EUR — rough fallback only.
_FALLBACK = {"EUR": 1.0, "USD": 1.08, "PLN": 4.3, "GBP": 0.85, "CHF": 0.95, "TRY": 37.0, "CZK": 25.0}


async def rates() -> dict[str, float]:
    cached = cache.get("fx:ecb")
    if cached:
        return cached
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            xml = (await c.get(_ECB_URL)).text
        r = {cur: float(val) for cur, val in re.findall(r"currency='([A-Z]{3})' rate='([0-9.]+)'", xml)}
        r["EUR"] = 1.0
        if len(r) > 5:
            cache.put("fx:ecb", r, 24 * 3600)
            return r
    except Exception:
        pass
    return dict(_FALLBACK)


def to_eur(amount: float | None, currency: str | None, table: dict[str, float]) -> float | None:
    if amount is None:
        return None
    rate = table.get((currency or "EUR").upper())
    return round(amount / rate, 2) if rate else None
