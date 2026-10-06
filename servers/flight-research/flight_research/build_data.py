"""Build-time data prep (run in the Dockerfile).

- COUNTRY_NAMES_CSV (`code,name`) from Babel's CLDR territory names, so users can
  type "Niemcy", "Allemagne" or "Alemania" for DE. Names that mean different
  countries in different languages are dropped rather than guessed.
- PRICE_LEVELS_CSV (`code,level`): price level vs the US = World Bank household
  consumption PPP factor / market exchange rate (CC BY 4.0), latest year per
  country. Scales ground-transfer cost estimates. Best effort: without network the
  file is skipped and estimates use a flat level.

    python -m flight_research.build_data
"""

from __future__ import annotations

import csv
from collections import defaultdict

import httpx
from babel import Locale, localedata

from flight_research.airports import COUNTRIES_CSV, COUNTRY_NAMES_CSV, PRICE_LEVELS_CSV, _norm

_WB = "https://api.worldbank.org/v2/country/all/indicator/{}?format=json&per_page=1000&mrnev=1"


def _wb(indicator: str) -> dict[str, float]:
    data = httpx.get(_WB.format(indicator), timeout=60, follow_redirects=True).json()
    return {r["country"]["id"]: float(r["value"]) for r in data[1] or []
            if r.get("value") is not None and len(r["country"]["id"]) == 2}


def price_levels() -> None:
    try:
        ppp, fx = _wb("PA.NUS.PRVT.PP"), _wb("PA.NUS.FCRF")
    except Exception as e:  # build stays offline-capable
        print(f"{PRICE_LEVELS_CSV}: skipped ({type(e).__name__}: {e})")
        return
    rows = sorted((c, round(ppp[c] / fx[c], 3)) for c in ppp.keys() & fx.keys() if fx[c] > 0)
    with open(PRICE_LEVELS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "level"])
        w.writerows(rows)
    print(f"{PRICE_LEVELS_CSV}: {len(rows)} countries")


def main() -> None:
    price_levels()
    with open(COUNTRIES_CSV, encoding="utf-8") as f:
        known = {row["code"] for row in csv.DictReader(f)}
    codes: dict[str, set[str]] = defaultdict(set)
    for loc in localedata.locale_identifiers():
        try:
            territories = Locale.parse(loc).territories
        except Exception:  # a few identifiers don't parse on every Babel version
            continue
        for code, name in territories.items():
            if code in known and _norm(name):
                codes[_norm(name)].add(code)
    rows = sorted((next(iter(c)), n) for n, c in codes.items() if len(c) == 1)
    with open(COUNTRY_NAMES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name"])
        w.writerows(rows)
    print(f"{COUNTRY_NAMES_CSV}: {len(rows)} names, {len(codes) - len(rows)} ambiguous dropped")


if __name__ == "__main__":
    main()
