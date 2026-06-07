"""Starlette ASGI app: Bearer auth + Streamable HTTP MCP endpoint + Tool-RAG."""
from __future__ import annotations
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from gateway.auth import KeyStore, load_keys
from gateway.context import current_policy
from gateway.health import ServerHealth, health_loop
from gateway.index_publisher import IndexPublisher
from gateway.registry import Registry, load_registries, load_registry
from gateway.server import build_gateway_server
from gateway.sync_adapter import SyncAdapter
from gateway.tool_db import ToolDb
from tool_rag.embedder import create_embedder
from tool_rag.indexer import ToolRagIndexer
from tool_rag.retriever import Retriever
from tool_rag.router import ToolRagRouter
logger = logging.getLogger(__name__)
DEFAULT_MCP_PATH = "/mcp"

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require Authorization: Bearer <secret> for MCP traffic; skip health and optional tool-rag."""

    def __init__(self, app, key_store, skip_prefixes=("/health",)):
        super().__init__(app)
        self._key_store = key_store
        self._skip_prefixes = skip_prefixes

    async def dispatch(self, request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in self._skip_prefixes):
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Missing or invalid Authorization"}, status_code=401)
        token = auth[7:].strip()
        policy = self._key_store.resolve(token)
        if policy is None:
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)
        logger.debug("Authenticated key_id=%s", policy.key_id)
        tok = current_policy.set(policy)
        try:
            return await call_next(request)
        finally:
            current_policy.reset(tok)


async def health(_):
    return JSONResponse({"status": "ok"})


def _skip_prefixes():
    prefixes = ["/health"]
    if os.environ.get("TOOL_RAG_WITHOUT_AUTH", "").lower() in ("1", "true"):
        prefixes.append("/tool-rag")
    return tuple(prefixes)


async def resync_loop(registry, tool_db, indexer, interval):
    """Opt-in background re-pull of upstream tool lists (TOOL_RAG_RESYNC_INTERVAL).

    Re-runs full_sync (re-query upstreams + reconcile) then a clean full_reindex,
    so tool add/deprecate on a *running* upstream is picked up without a restart.
    Off by default; restart is the default way to refresh the catalog.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            result = await SyncAdapter(registry, tool_db).full_sync()
            indexer.full_reindex()
            logger.info(
                "Resync: %d servers, %d added, %d updated, %d removed",
                result.servers_synced, result.tools_added, result.tools_updated, result.tools_removed,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background resync failed")


def build_starlette_app(registry, key_store, mcp_path=DEFAULT_MCP_PATH, tool_rag_enabled=False, tool_db=None, tool_rag_router=None, server_health=None, retriever=None):
    mcp = build_gateway_server(registry, retriever=retriever)
    session_manager = StreamableHTTPSessionManager(app=mcp, stateless=True, json_response=False)
    streamable_http_app = StreamableHTTPASGIApp(session_manager)
    is_tr = tool_rag_enabled and tool_rag_router is not None

    @asynccontextmanager
    async def lifespan(_):
        tasks: list[asyncio.Task] = []
        async with session_manager.run():
            if is_tr and tool_db is not None and tool_rag_router is not None:
                logger.info("Tool-RAG: initializing sync + index ...")
                indexer = tool_rag_router._indexer
                try:
                    sync = SyncAdapter(registry, tool_db)
                    result = await sync.full_sync()
                    logger.info(
                        "Tool-RAG: synced %d servers, %d added, %d updated, %d removed",
                        result.servers_synced, result.tools_added, result.tools_updated, result.tools_removed,
                    )
                    mode = os.environ.get("TOOL_RAG_STARTUP_REINDEX", "full").lower()
                    if mode == "incremental":
                        IndexPublisher(tool_db, indexer).publish_sync_results()
                    elif mode == "off":
                        logger.info("Tool-RAG: startup reindex disabled (TOOL_RAG_STARTUP_REINDEX=off)")
                    else:
                        if mode != "full":
                            logger.warning("Unknown TOOL_RAG_STARTUP_REINDEX=%r, using 'full'", mode)
                        indexer.full_reindex()
                except Exception:
                    logger.exception("Tool-RAG initial sync failed")

                # Runtime liveness: initial probe, then optional background loops.
                if server_health is not None:
                    hc_interval = float(os.environ.get("TOOL_RAG_HEALTHCHECK_INTERVAL", "30"))
                    hc_timeout = float(os.environ.get("TOOL_RAG_HEALTHCHECK_TIMEOUT", "5"))
                    try:
                        await server_health.probe_all(hc_timeout)
                    except Exception:
                        logger.exception("Initial health probe failed")
                    if hc_interval > 0:
                        tasks.append(asyncio.create_task(health_loop(server_health, hc_interval, hc_timeout)))
                resync_interval = float(os.environ.get("TOOL_RAG_RESYNC_INTERVAL", "0"))
                if resync_interval > 0:
                    tasks.append(asyncio.create_task(resync_loop(registry, tool_db, indexer, resync_interval)))
            try:
                yield
            finally:
                for t in tasks:
                    t.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

    routes = [
        Route("/health", health, methods=["GET"]),
        Route(mcp_path, endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
    ]
    if is_tr:
        routes.extend([
            Route("/tool-rag/retrieve", endpoint=tool_rag_router.retrieve, methods=["POST"]),
            Route("/tool-rag/reindex", endpoint=tool_rag_router.reindex, methods=["POST"]),
            Route("/tool-rag/health", endpoint=tool_rag_router.health, methods=["GET"]),
            Route("/tool-rag/metrics", endpoint=tool_rag_router.metrics, methods=["GET"]),
        ])
    return Starlette(routes=routes, lifespan=lifespan, middleware=[Middleware(APIKeyMiddleware, key_store=key_store, skip_prefixes=_skip_prefixes())])


def _config_dir():
    return Path(os.environ.get("MCP_GATEWAY_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))


def app_from_env():
    base = _config_dir()
    reg = Path(os.environ.get("MCP_GATEWAY_REGISTRY", base / "registry.yaml"))
    reg_generated = Path(os.environ.get("MCP_GATEWAY_REGISTRY_GENERATED", base / "registry.generated.yaml"))
    keys = Path(os.environ.get("MCP_GATEWAY_KEYS", base / "keys.yaml"))
    mcp_path = os.environ.get("MCP_GATEWAY_MCP_PATH", DEFAULT_MCP_PATH)
    registry = load_registries(reg, reg_generated)
    key_store = load_keys(keys)
    tool_rag_enabled = os.environ.get("TOOL_RAG_ENABLED", "1").lower() in ("1", "true")
    tool_db = None
    tool_rag_router = None
    retriever = None
    server_health = ServerHealth(registry)
    if tool_rag_enabled:
        db_path = os.environ.get("TOOL_RAG_DB", "tool_registry.db")
        tool_db = ToolDb(db_path)
        embedder = create_embedder()
        indexer = ToolRagIndexer(embedder, tool_db)
        retriever = Retriever(embedder, indexer, tool_db, server_health=server_health)
        tool_rag_router = ToolRagRouter(tool_db, embedder, indexer, retriever, server_health=server_health)
    return build_starlette_app(registry, key_store, mcp_path=mcp_path, tool_rag_enabled=tool_rag_enabled, tool_db=tool_db, tool_rag_router=tool_rag_router, server_health=server_health, retriever=retriever)
