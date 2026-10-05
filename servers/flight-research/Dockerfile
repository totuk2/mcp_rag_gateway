# flight-research: our own MCP server (aviation data + research_route pipeline).
FROM python:3.12-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir "mcp>=1.10,<2" httpx babel
# OurAirports (public domain): refreshed on every image build.
ADD https://davidmegginson.github.io/ourairports-data/airports.csv /app/data/airports.csv
ADD https://davidmegginson.github.io/ourairports-data/countries.csv /app/data/countries.csv
# OpenFlights routes (ODbL): route counts rank airports by traffic.
ADD https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat /app/data/routes.dat
# GeoNames country info (CC BY 4.0): land neighbours -> ground legs and island states.
ADD https://download.geonames.org/export/dump/countryInfo.txt /app/data/countryInfo.txt
COPY flight_research/ ./flight_research/
# Country names in every CLDR language -> /app/data/country_names.csv.
RUN python -m flight_research.build_data
ENV FLIGHT_RESEARCH_CACHE=/data/cache.db PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "flight_research.server"]
