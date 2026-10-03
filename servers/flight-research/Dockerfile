# flight-research: our own MCP server (aviation data + research_route pipeline).
FROM python:3.12-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir "mcp>=1.10,<2" httpx
# OurAirports (public domain): refreshed on every image build.
ADD https://davidmegginson.github.io/ourairports-data/airports.csv /app/data/airports.csv
COPY flight_research/ ./flight_research/
ENV FLIGHT_RESEARCH_CACHE=/data/cache.db PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "flight_research.server"]
