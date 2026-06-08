FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    'httpx>=0.28.1' \
    'mcp[cli]>=1.2.1' \
    'pillow>=11.1.0'

COPY mcp_image.py ./

RUN mkdir -p /app/data /app/Temp

ENV MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

EXPOSE 8000

CMD ["python", "mcp_image.py"]
