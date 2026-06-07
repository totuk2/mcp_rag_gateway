"""Run the gateway with uvicorn: `python -m gateway` from the mcp_server directory."""

from __future__ import annotations

import logging
import os

import uvicorn

from gateway.app import app_from_env

logging.basicConfig(level=os.environ.get("MCP_GATEWAY_LOG_LEVEL", "INFO"))


def main() -> None:
    host = os.environ.get("MCP_GATEWAY_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_GATEWAY_PORT", "8765"))
    uvicorn.run(
        app_from_env,
        host=host,
        port=port,
        factory=True,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
