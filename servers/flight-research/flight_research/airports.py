"""Airport lookup and proximity search over the OurAirports dataset.

The CSV (https://ourairports.com/data/, public domain) is downloaded into the
image at build time; AIRPORTS_CSV overrides the path. Only airports with an
IATA code are kept — those are the ones fare engines can price.

Everything else is derived from data, not hand-kept tables (each file optional):
- COUNTRY_NAMES_CSV: country names in every CLDR language (build_data.py).
- ROUTES_DAT: OpenFlights routes (ODbL); routes per airport = its traffic rank,
  which picks a country's main airport (OurAirports has no traffic data).
- AIRPORT_OVERRIDES: `kind,key,value` rows for what data gets wrong
  (`country_alias,anglia,GB`, `primary,US,JFK`; borders.py reads `land_link`
  and `closed_border`).
"""

from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

CSV_PATH = os.environ.get("AIRPORTS_CSV", "/app/data/airports.csv")
COUNTRIES_CSV = os.environ.get("COUNTRIES_CSV", "/app/data/countries.csv")
COUNTRY_NAMES_CSV = os.environ.get("COUNTRY_NAMES_CSV", "/app/data/country_names.csv")
ROUTES_DAT = os.environ.get("ROUTES_DAT", "/app/data/routes.dat")
OVERRIDES_CSV = os.environ.get("AIRPORT_OVERRIDES", os.path.join(os.path.dirname(__file__), "overrides.csv"))
# A name word in more than this share of airport names ("airport", "regional",
# "field") says nothing about which airport is meant.
GENERIC_DF = float(os.environ.get("AIRPORT_GENERIC_DF", "0.005"))
# Lower is bigger. Small airports matter only when they have scheduled service.
TYPE_RANK = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    type: str
    scheduled: bool

    def brief(self) -> dict:
        return {"iata": self.iata, "name": self.name, "city": self.city, "country": self.country,
                "type": self.type.replace("_airport", ""), "scheduled_service": self.scheduled}


_by_iata: dict[str, Airport] = {}
_all: list[Airport] = []
_by_name: dict[str, Airport] = {}
# Normalized alternate names / metro codes (OurAirports `keywords`, e.g. "MIL" for MXP).
_keywords: dict[str, set[str]] = {}
# Normalized country name / keyword / alias -> ISO code.
_countries: dict[str, str] = {}
# IATA -> number of routes touching it (OpenFlights); 0 when unknown.
_traffic: dict[str, int] = {}
# Country ISO code -> pinned main airport (overrides only; else busiest wins).
_primary: dict[str, str] = {}
# Name words too common to identify an airport (computed in load()).
_generic: set[str] = set()


# Letters NFKD doesn't decompose into base + accent (else "Wałęsa" -> "waesa").
_TRANSLIT = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss",
                           "æ": "ae", "Æ": "AE", "ı": "i", "ð": "d", "þ": "th", "œ": "oe"})


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").translate(_TRANSLIT)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def load(path: str = CSV_PATH) -> None:
    if _all:
        return
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iata = (row.get("iata_code") or "").strip().upper()
            if len(iata) != 3 or row.get("type") not in TYPE_RANK:
                continue
            try:
                ap = Airport(iata, row["name"], row.get("municipality") or "", row.get("iso_country") or "",
                             float(row["latitude_deg"]), float(row["longitude_deg"]), row["type"],
                             row.get("scheduled_service") == "yes")
            except (KeyError, ValueError):
                continue
            # Several rows can share an IATA (closed/duplicate entries): keep the busiest kind.
            prev = _by_iata.get(iata)
            if prev is None or (not prev.scheduled and ap.scheduled) or TYPE_RANK[ap.type] < TYPE_RANK[prev.type]:
                _by_iata[iata] = ap
                _keywords[iata] = {_norm(k) for k in (row.get("keywords") or "").split(",") if _norm(k)}
    _all.extend(_by_iata.values())
    for ap in _all:
        _by_name.setdefault(_norm(ap.name), ap)
    df = Counter(t for ap in _all for t in set(_norm(ap.name).split()))
    _generic.update(t for t, n in df.items() if n > len(_all) * GENERIC_DF)
    # English names and OurAirports keywords first; CLDR translations only fill gaps.
    for path, names in ((COUNTRIES_CSV, lambda r: [r.get("name") or ""] + (r.get("keywords") or "").split(",")),
                        (COUNTRY_NAMES_CSV, lambda r: [r.get("name") or ""])):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    for n in names(row):
                        if _norm(n):
                            _countries.setdefault(_norm(n), row.get("code") or "")
    if os.path.exists(ROUTES_DAT):
        with open(ROUTES_DAT, encoding="utf-8") as f:
            routes = {(r[2], r[4]) for r in csv.reader(f) if len(r) > 4}
        _traffic.update(Counter(code for route in routes for code in route))
    for key, value in override_rows("country_alias"):
        _countries[_norm(key)] = value.upper()
    for key, value in override_rows("primary"):
        _primary[key.upper()] = value.upper()


def override_rows(kind: str) -> list[tuple[str, str]]:
    """(key, value) rows of one `kind` from AIRPORT_OVERRIDES; [] without the file."""
    if not os.path.exists(OVERRIDES_CSV):
        return []
    with open(OVERRIDES_CSV, encoding="utf-8") as f:
        return [(row.get("key") or "", row.get("value") or "") for row in csv.DictReader(f) if row.get("kind") == kind]


def _rank(ap: Airport) -> tuple:
    return (not ap.scheduled, TYPE_RANK[ap.type], -_traffic.get(ap.iata, 0), ap.iata)


def get(iata: str) -> Airport | None:
    load()
    return _by_iata.get((iata or "").strip().upper())


def resolve(location: str) -> list[Airport]:
    """IATA code, or a city/airport name ("Aleppo", "Gdańsk") -> best matches."""
    load()
    loc = (location or "").strip()
    if re.fullmatch(r"[A-Za-z]{3}", loc) and (ap := get(loc)):
        return [ap]
    key = _norm(loc)
    if not key:
        return []
    exact = [a for a in _all if _norm(a.city) == key or key in _keywords.get(a.iata, ())]
    if exact:
        return sorted(exact, key=_rank)
    partial = [a for a in _all if key in _norm(a.city) or key in _norm(a.name)]
    return sorted(partial, key=_rank)[:10]


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t not in _generic}


def country_code(location: str) -> str | None:
    """ISO code for a country name in any language ("Polska", "Allemagne") or a
    2-letter code ("PL"); None if `location` isn't a country."""
    load()
    loc = (location or "").strip()
    return loc.upper() if re.fullmatch(r"[A-Za-z]{2}", loc) else _countries.get(_norm(loc))


def country_airports(location: str, limit: int = 12) -> list[Airport]:
    """The main scheduled airports of a country given by name ("Poland", "Polska")
    or ISO code ("PL"); [] if `location` isn't a country. Busiest first — the
    first one is what single-airport engines use."""
    code = country_code(location)
    if not code:
        return []
    aps = [a for a in _all if a.country == code and a.scheduled and a.type != "small_airport"]
    # Traffic before type: OurAirports calls e.g. every Polish airport "large".
    return sorted(aps, key=lambda a: (a.iata != _primary.get(code), -_traffic.get(a.iata, 0), *_rank(a)))[:limit]


def code_for_name(name: str) -> str:
    """Best-effort IATA for an airport *name* (Google Flights legs carry names only).
    Exact normalized name, else substring, else all significant words contained
    ("Ben Gurion Airport" -> "Ben Gurion International Airport")."""
    load()
    key = _norm(name)
    if (ap := _by_name.get(key)):
        return ap.iata
    hits = [a for a in _all if a.scheduled and (key in _norm(a.name) or _norm(a.name) in key)]
    if not hits and (toks := _tokens(name)):
        hits = [a for a in _all if a.scheduled and toks <= _tokens(a.name)]
        if not hits:  # the query has extra words ("... Kraków Balice ...")
            hits = [a for a in _all if a.scheduled and len(_tokens(a.name)) >= 2 and _tokens(a.name) <= toks]
    return sorted(hits, key=_rank)[0].iata if hits else name


def haversine_km(a: Airport, b: Airport) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def nearby(center: Airport, radius_km: float, scheduled_only: bool = True, limit: int = 15) -> list[tuple[Airport, float]]:
    """Other airports within radius, nearest first (scheduled-service ones by default)."""
    load()
    out = []
    for ap in _all:
        if ap.iata == center.iata or (scheduled_only and not ap.scheduled):
            continue
        km = haversine_km(center, ap)
        if km <= radius_km:
            out.append((ap, km))
    out.sort(key=lambda x: (x[1], x[0].iata))
    return out[:limit]
