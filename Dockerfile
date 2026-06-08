FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optionally bake the cross-encoder reranker into the image so it is warm at boot
# and needs no HF Hub access at runtime. Gated on TOOL_RAG_RERANKER (compose
# passes it from .env): the model is downloaded only when it equals "local".
# The named hf-cache volume seeds itself from this baked cache on first run.
ARG TOOL_RAG_RERANKER=off
ARG TOOL_RAG_RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RUN if [ "$TOOL_RAG_RERANKER" = "local" ]; then \
        echo "Baking reranker model: $TOOL_RAG_RERANKER_MODEL" && \
        python -c "from sentence_transformers import CrossEncoder; CrossEncoder('$TOOL_RAG_RERANKER_MODEL')"; \
    else \
        echo "TOOL_RAG_RERANKER=$TOOL_RAG_RERANKER — skipping reranker bake"; \
    fi

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
