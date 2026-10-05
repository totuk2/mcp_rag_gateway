"""Build-time data prep (run in the Dockerfile): country names in every CLDR language.

Writes COUNTRY_NAMES_CSV (`code,name`) from Babel's CLDR territory names, so users
can type "Niemcy", "Allemagne" or "Alemania" for DE. Names that mean different
countries in different languages are dropped rather than guessed.

    python -m flight_research.build_data
"""

from __future__ import annotations

import csv
from collections import defaultdict

from babel import Locale, localedata

from flight_research.airports import COUNTRIES_CSV, COUNTRY_NAMES_CSV, _norm


def main() -> None:
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
