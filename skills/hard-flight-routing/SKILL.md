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
(IATA, metro code, English city name — translate e.g. "Mediolan" → "Milan" — or a
country = all its scheduled airports) and the user's dates as ONE window: `date` +
`date_to` ("1–15 November" → 11-01..11-15), or `date` ± `flex_days` (any number).
Work the window out yourself and make one call per direction — never one call per day.
It takes up to ~150 s (a background job: poll get_job_result) and searches every origin airport, the hubs into the destination
(with their nearby airports, e.g. DXB + SHJ) and the hub legs over the whole window.
It returns:

- `options` — ranked itineraries (price in EUR, estimated ground cost/time in
  `ground_legs`, route — `~` marks an airport change, e.g. `KRK-SHJ~DXB-KBL` —
  segments, booking links, notes);
- `hubs` — hub areas used, with where the evidence came from;
- `unchecked` — searches that FAILED: report them as "could not be checked", never as
  "no flights";
- `coverage` — every engine/route query that ran;
- `manual_checks` — airlines flying the last leg that no engine priced.

Read `summary` first — one line per ranked option (price, route, times, flags, link),
ranked by value; `CHEAPEST:` lines add the cheapest options when they rank lower.
Quote prices, routes and links **exactly** from `summary`/`options`; never estimate or
reconstruct them. If it returns good options, present them (see "Answer format") — you're
done.

## 2. If results are thin, dig deeper

1. **What flies in?** `flight-research__airport_routes` for the destination and its
   neighbours (`flight-research__nearby_airports`). The airlines and origin airports
   listed there are the only ways in — build the trip around them.
2. **Price the legs yourself** (research_route already does this for its top hubs; do it
   for a hub it didn't use, or a carrier the user named). Search origin→H and H→destination
   separately with `kiwi__search-flight` (supports `allow_self_transfer`,
   `allow_diff_airport_connection`, `departureDateFlexDays`) and
   `google-flights__search_flights` (comma-separated airports allowed). Leave ≥3 h
   between separately-ticketed flights, more if changing airports.
3. **Check operating days.** `flight-research__airport_schedule` shows which days a thin
   route actually operates — shift the date if the last leg doesn't fly that day.
4. **Ground options.** A nearby airport plus a bus/taxi can beat the direct airport.
   Always state the distance and whether a border crossing is involved.
   - `border_warnings` (and each option's `border_warnings`) are **remembered reports**
     of closed/restricted borders, and island legs with no road link. Repeat them to the
     user with their date and source; for island legs tell the user to double-check the
     ferry/sea connection. Options behind them are ranked down, not removed.
   - `unverified_borders` = land borders crossed by the shown options with no remembered
     status. If your answer recommends a ground leg across one, web-search its current
     status first.
   - **Grow the border memory:** whenever a web search shows a land border closed,
     restricted (only some crossings, permits, locals only) or open again, call
     `flight-research__report_border_status` with both countries, the status, the source
     URL and a one-line note. Report "open" too — it clears an old closure.
     `flight-research__border_status` shows what is remembered.
   - **When the user says a border is closed (or open), never take it as given:**
     always verify it with a web search first. If the search confirms it, record it with
     `flight-research__report_border_status` (the source URL is the page that confirms it,
     not the user). If the search contradicts the user or finds nothing, tell the user
     what you found and don't record it.

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

- A table of the best 3–6 options sorted by total price (tickets + estimated ground cost,
  marked as an estimate), always including the cheapest found and the best-value (#1)
  option: route, dates/times, number of tickets, self-transfer / airport change / ground
  legs, source, booking link. Separate-ticket connections under 3 h: flag as risky.
- Clearly mark **separate tickets** (missed connection = your problem) and **ground
  legs / border crossings**.
- One line on what was searched and what couldn't be checked (from `unchecked`).
- For destinations with travel advisories or visa requirements (e.g. Syria, Afghanistan,
  Yemen), add a short note to check the official travel advisory and visa rules before
  booking.
- Never state as fact whether a border crossing is open, a visa is available or a route
  is safe. Border memory entries are reports ("reported closed on <date>, <source>"), not
  verified facts — say they must be verified with official sources.
