FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ ./gateway/
COPY servers/ ./servers/
COPY gateway/ ./gateway/
COPY tool_rag/ ./tool_rag/
COPY servers/ ./servers/
COPY config/ ./config/

ENV MCP_GATEWAY_CONFIG_DIR=/app/config
ENV MCP_GATEWAY_HOST=0.0.0.0
ENV MCP_GATEWAY_PORT=8765

EXPOSE 8765

CMD ["python", "-m", "gateway"]
