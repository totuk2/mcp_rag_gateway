"""Ground-transfer estimates: time and rough cost of getting between two airports
(a self-transfer between airports of one area, e.g. SHJ -> DXB) or from the landing
airport to the real destination.

Nothing is per-city: road distance ~= great-circle km x GROUND_ROAD_FACTOR; time =
fixed overhead (leaving the airport, waiting) + road km / GROUND_KMH; cost = a taxi
fare for the first GROUND_TAXI_MAX_KM and a bus/shared-taxi rate beyond, priced at
US levels and scaled by the country's price level (World Bank, airports.price_level).
These are estimates and are labelled as such wherever shown.
"""

from __future__ import annotations

import os

from flight_research import airports

ROAD_FACTOR = float(os.environ.get("GROUND_ROAD_FACTOR", "1.3"))
KMH = float(os.environ.get("GROUND_KMH", "65"))
FIXED_H = float(os.environ.get("GROUND_FIXED_H", "0.75"))
TAXI_FLAG_USD = float(os.environ.get("GROUND_TAXI_FLAG_USD", "4"))
TAXI_KM_USD = float(os.environ.get("GROUND_TAXI_KM_USD", "1.4"))
TAXI_MAX_KM = float(os.environ.get("GROUND_TAXI_MAX_KM", "30"))
LONG_KM_USD = float(os.environ.get("GROUND_LONG_KM_USD", "0.12"))


def estimate(km: float, country: str, usd_per_eur: float | None) -> dict:
    """{"km", "road_km", "hours", "cost_eur"} for a ground leg of `km` great-circle km
    in `country` (ISO code). `usd_per_eur` from fx.rates()["USD"]."""
    road = km * ROAD_FACTOR
    usd = airports.price_level(country) * (TAXI_FLAG_USD + TAXI_KM_USD * min(road, TAXI_MAX_KM)
                                           + LONG_KM_USD * max(0.0, road - TAXI_MAX_KM))
    return {"km": round(km), "road_km": round(road), "hours": round(FIXED_H + road / KMH, 1),
            "cost_eur": round(usd / (usd_per_eur or 1.1))}
