---
name: hard-flight-routing
description: >-
  Playbook for finding flights on hard or unusual routes — small or remote airports,
  conflict regions, destinations few airlines serve, or when a first flight search
  returns nothing (e.g. Gdańsk to Aleppo, to Kabul, to an island). Covers nearby
  airports, hubs, self-transfer combinations, ground legs and airline websites.
  Also use for any multi-engine flight price comparison.
requires_servers: [flight-research]
---

# Hard flight routing

Use this when a user asks for flights and the route is not a simple hub-to-hub trip,
or when a first search came back empty. **Do not give up after one empty search** —
thin routes almost always have *some* way in; your job is to find and compare them.

## 1. Run the systematic search first

Call `flight-research__research_route` (via `run_tool`) with origin, destination
(IATA, metro code or English city name — translate e.g. "Mediolan" → "Milan") and
date (`YYYY-MM-DD`). Defaults are sensible; pass `flex_days` if the user is flexible.
It takes up to ~75 s. It returns:

- `options` — ranked itineraries (price in EUR, route, segments, booking links, notes);
- `hubs` — airports that actually fly into the destination area;
- `coverage` — which engine/route queries ran and which failed;
- `manual_checks` — airlines flying the last leg that no engine priced.

Read `summary` first — one line per ranked option (price, route, times, flags, link).
Quote prices, routes and links **exactly** from `summary`/`options`; never estimate or
reconstruct them. If it returns good options, present them (see "Answer format") — you're
done.

## 2. If results are thin, dig deeper

1. **What flies in?** `flight-research__airport_routes` for the destination and its
   neighbours (`flight-research__nearby_airports`). The airlines and origin airports
   listed there are the only ways in — build the trip around them.
2. **Price the legs yourself.** For each promising hub H: search origin→H and H→destination
   separately with `kiwi__search-flight` (supports `allow_self_transfer`,
   `allow_diff_airport_connection`, `departureDateFlexDays`) and
   `google-flights__search_flights` (comma-separated airports allowed). Leave ≥3 h
   between separately-ticketed flights, more if changing airports.
3. **Check operating days.** `flight-research__airport_schedule` shows which days a thin
   route actually operates — shift the date if the last leg doesn't fly that day.
4. **Ground options.** A nearby airport plus a bus/taxi can beat the direct airport.
   Always state the distance and whether a border crossing is involved.

5. **Airline codes.** Many airlines fly under several IATA codes; filter with all of
   them (`select_airlines`): Wizz Air `W6,W4,W9,5W` (W4 = Wizz Air Malta), Ryanair
   group `FR,RK,AL,LW`, easyJet `U2,EC,DS`. When unsure, search unfiltered and read the
   carriers in the results.
6. **Does the leg exist?** Before reporting "no fares" for a requested direct leg, check
   `flight-research__airport_routes` on one end. If it isn't served, say so and offer the
   closest alternatives (same airline with a connection, or another airline).

## 3. Airline websites (last resort)

For carriers in `manual_checks` (often small national or regional airlines), open the
airline's own booking page with the Playwright tools (`playwright__browser_navigate`,
`browser_snapshot`, `browser_fill_form`, `browser_click`) and search the leg there.

**Use and grow the site memory:**
- Before opening the site, call `get_site_recipes` with the airline name or domain. If a
  recipe exists, follow it (fill its `url_template` placeholders, or replay its `steps`).
  Recipes are also appended automatically when you navigate to a known site.
- When you got real results from the site, call `save_site_recipe`: `site` (domain),
  `label` (airline name), `task` (e.g. "search one-way flights"), `url_template` if the
  results page has a reusable URL (placeholders `{origin}`, `{destination}`,
  `{date:YYYY-MM-DD}` — state the site's date format), and `steps` (each a short action
  naming the visible label/button, e.g. "click 'One way'", "type origin IATA into 'From'
  and pick the first suggestion"), plus `notes` (cookie banner, slow loading…).
- After following an existing recipe, report it (`recipe_id`, `worked` true/false); if it
  broke, save the corrected version. Never store credentials or personal data.

- If you hit a CAPTCHA, bot check or login wall: **stop on that site** and tell the user
  which site and leg to check manually. Never try to bypass it.
- Do not book or pay for anything; only read schedules and prices.

## Answer format

- A table of the best 3–6 options: total price (EUR + original currency), route,
  dates/times, number of tickets, self-transfer/ground legs, source, booking link.
- Clearly mark **separate tickets** (missed connection = your problem) and **ground
  legs / border crossings**.
- One line on what was searched and what couldn't be checked (from `coverage`).
- For destinations with travel advisories or visa requirements (e.g. Syria, Afghanistan,
  Yemen), add a short note to check the official travel advisory and visa rules before
  booking.
- Never state as fact whether a border crossing is open, a visa is available or a route
  is safe — no tool here checks that. Say it must be verified (official sources).
