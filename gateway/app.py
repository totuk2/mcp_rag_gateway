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
from gateway.reloader import ConfigReloader
from gateway.server import build_gateway_server
from gateway.session_pool import StickySessionPool
from gateway.recipes import RecipeStore
from gateway.skills import SkillStore
from gateway.sync_adapter import SyncAdapter
from gateway.tool_db import ToolDb
from tool_rag.catalog import ToolCatalog
from tool_rag.embedder import create_embedder
from tool_rag.enrichment import refresh_catalog
from tool_rag.reranker import create_reranker
from tool_rag.planner import create_planner
from tool_rag.indexer import ToolRagIndexer
from tool_rag.retriever import Retriever
from tool_rag.router import ToolRagRouter
logger = logging.getLogger(__name__)
DEFAULT_MCP_PATH = "/mcp"

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require Authorization: Bearer <secret> for MCP traffic; skip health and optional tool-rag.

    When anon_policy is set, unauthenticated requests use that policy instead of returning 401.
    This allows MCP clients that don't support simple Bearer auth (e.g. LibreChat, which requires
    full OAuth) to connect without credentials.
    """

    def __init__(self, app, key_store, skip_prefixes=("/health",), anon_policy=None, anon_key_id=None):
        super().__init__(app)
        self._key_store = key_store
        self._skip_prefixes = skip_prefixes
        self._anon_static = anon_policy
        self._anon_key_id = anon_key_id

    @property
    def _anon_policy(self):
        # Resolved per request so a hot-reloaded keys.yaml applies to anonymous access too.
        if self._anon_key_id:
            return self._key_store.by_id(self._anon_key_id)
        return self._anon_static

    async def dispatch(self, request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in self._skip_prefixes):
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            anon = self._anon_policy
            if anon is not None:
                logger.debug("Unauthenticated request, using anon policy key_id=%s", anon.key_id)
                tok = current_policy.set(anon)
                try:
                    return await call_next(request)
                finally:
                    current_policy.reset(tok)
            return JSONResponse(
                {"detail": "Missing or invalid Authorization"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'},
            )
        token = auth[7:].strip()
        policy = self._key_store.resolve(token)
        if policy is None:
            return JSONResponse(
                {"detail": "Invalid API key"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway", error="invalid_token"'},
            )
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


async def _refresh_catalog_safely(registry, tool_db, planner):
    """Recompute catalog categories/tags if the tool set changed. Never raises:
    the catalog is derived data, it must not block sync or indexing. Returns
    the RefreshResult, or None on an unexpected error."""
    try:
        return await refresh_catalog(registry, tool_db, planner)
    except Exception:
        logger.exception("Tool catalog refresh failed")
        return None


async def catalog_retry_loop(registry, tool_db, indexer, planner, delays=(60, 120, 240)):
    """After an incomplete refresh (LLM rate limit / outage), retry a few times
    with backoff instead of waiting for the next restart or resync."""
    for delay in delays:
        await asyncio.sleep(delay)
        res = await _refresh_catalog_safely(registry, tool_db, planner)
        if res is not None and res.tags_changed:
            indexer.incremental_reindex()  # tags are part of the embedding text
        if res is not None and not res.incomplete:
            logger.info("Tool catalog retry succeeded")
            return
    logger.warning("Tool catalog still incomplete after %d retries; next refresh on restart/resync", len(delays))


async def resync_loop(registry, tool_db, indexer, interval, planner=None):
    """Opt-in background re-pull of upstream tool lists (TOOL_RAG_RESYNC_INTERVAL).

    Re-runs full_sync (re-query upstreams + reconcile), refreshes the catalog
    (categories follow added/removed tools), then a clean full_reindex, so tool
    add/deprecate on a *running* upstream is picked up without a restart.
    Off by default; restart is the default way to refresh the catalog.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            result = await SyncAdapter(registry, tool_db).full_sync()
            await _refresh_catalog_safely(registry, tool_db, planner)
            indexer.full_reindex()
            logger.info(
                "Resync: %d servers, %d added, %d updated, %d removed",
                result.servers_synced, result.tools_added, result.tools_updated, result.tools_removed,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background resync failed")


def build_starlette_app(registry, key_store, mcp_path=DEFAULT_MCP_PATH, tool_rag_enabled=False, tool_db=None, tool_rag_router=None, server_health=None, retriever=None, anon_policy=None, planner=None, max_parallel=8, catalog=None, skills=None, recipes=None, config_paths=None, anon_key_id=None):
    # Sticky upstream sessions for servers flagged `stateful` (e.g. Playwright); always
    # created so a stateful server added by hot reload works without a restart.
    session_pool = StickySessionPool()
    mcp = build_gateway_server(registry, retriever=retriever, planner=planner, max_parallel=max_parallel, catalog=catalog, skills=skills, recipes=recipes, session_pool=session_pool)
    session_manager = StreamableHTTPSessionManager(app=mcp, stateless=False, json_response=False)
    streamable_http_app = StreamableHTTPASGIApp(session_manager)
    is_tr = tool_rag_enabled and tool_rag_router is not None

    async def _on_registry_change(diff):
        """Hot reload: re-sync upstreams (reconciles removed servers), refresh the
        catalog, reindex, re-probe liveness, drop sticky sessions of changed servers."""
        for sid in diff.get("removed", []) + diff.get("changed", []):
            session_pool.close_server(sid)
        if not (is_tr and tool_db is not None):
            return
        indexer = tool_rag_router._indexer
        result = await SyncAdapter(registry, tool_db).full_sync()
        await _refresh_catalog_safely(registry, tool_db, planner)
        indexer.full_reindex()
        if server_health is not None:
            await server_health.probe_all(float(os.environ.get("TOOL_RAG_HEALTHCHECK_TIMEOUT", "5")))
        logger.info("Hot reload: synced %d servers (%d added, %d removed tools)",
                    result.servers_synced, result.tools_added, result.tools_removed)

    reloader = None
    if config_paths is not None:
        reg_path, gen_path, keys_path = config_paths
        reloader = ConfigReloader(registry, key_store, reg_path, gen_path, keys_path, skills, _on_registry_change)

    async def reload_endpoint(request):
        policy = current_policy.get()
        if policy is None or not policy.admin:
            return JSONResponse({"detail": "admin key required"}, status_code=403)
        if reloader is None:
            return JSONResponse({"detail": "hot reload not configured"}, status_code=404)
        return JSONResponse(await reloader.check(force=True))

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
                    # Before reindexing: category tags are part of the embedding text.
                    catalog_res = await _refresh_catalog_safely(registry, tool_db, planner)
                    if catalog_res is None or catalog_res.incomplete:
                        tasks.append(asyncio.create_task(catalog_retry_loop(registry, tool_db, indexer, planner)))
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
                reload_interval = float(os.environ.get("CONFIG_RELOAD_INTERVAL", "10"))
                if reloader is not None and reload_interval > 0:
                    tasks.append(asyncio.create_task(reloader.loop(reload_interval)))
                resync_interval = float(os.environ.get("TOOL_RAG_RESYNC_INTERVAL", "0"))
                if resync_interval > 0:
                    tasks.append(asyncio.create_task(resync_loop(registry, tool_db, indexer, resync_interval, planner)))
            try:
                yield
            finally:
                for t in tasks:
                    t.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if retriever is not None:
                    await retriever.aclose()
                await session_pool.aclose()

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
            Route("/tool-rag/tool/{tool_id:path}", endpoint=tool_rag_router.describe, methods=["GET"]),
            Route("/tool-rag/catalog", endpoint=tool_rag_router.catalog, methods=["GET"]),
            Route("/tool-rag/catalog/refresh", endpoint=tool_rag_router.catalog_refresh, methods=["POST"]),
            Route("/tool-rag/reload", endpoint=reload_endpoint, methods=["POST"]),
            Route("/tool-rag/reviews", endpoint=tool_rag_router.reviews, methods=["GET"]),
            Route("/tool-rag/reviews/{action}", endpoint=tool_rag_router.reviews_decide, methods=["POST"]),
        ])
    return Starlette(routes=routes, lifespan=lifespan, middleware=[Middleware(APIKeyMiddleware, key_store=key_store, skip_prefixes=_skip_prefixes(), anon_policy=anon_policy, anon_key_id=anon_key_id)])


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
    planner = None
    catalog = None
    skills = None
    recipes = None
    try:
        max_parallel = int(os.environ.get("TOOL_RAG_MAX_PARALLEL", "8"))
    except ValueError:
        max_parallel = 8
    server_health = ServerHealth(registry)
    if tool_rag_enabled:
        db_path = os.environ.get("TOOL_RAG_DB", "tool_registry.db")
        tool_db = ToolDb(db_path)
        embedder = create_embedder()
        # FAISS index lives next to the DB (same persistent volume in compose).
        data_dir = Path(db_path).parent
        indexer = ToolRagIndexer(
            embedder, tool_db, index_path=data_dir / "tool_rag.index", meta_path=data_dir / "tool_rag.meta"
        )
        reranker = create_reranker()
        planner = create_planner()
        retriever = Retriever(embedder, indexer, tool_db, server_health=server_health, reranker=reranker)
        skills = SkillStore(embedder=embedder)
        if os.environ.get("RECIPES_ENABLED", "1").lower() in ("1", "true", "on"):
            recipes = RecipeStore(tool_db)
        catalog = ToolCatalog(tool_db, server_health=server_health, retriever=retriever, planner=planner, skills=skills)
        tool_rag_router = ToolRagRouter(
            tool_db, embedder, indexer, retriever, server_health=server_health,
            catalog=catalog, registry=registry, planner=planner,
        )
    anon_key_id = os.environ.get("GATEWAY_ANON_KEY", "").strip()
    anon_policy = key_store.by_id(anon_key_id) if anon_key_id else None
    if anon_key_id and anon_policy is None:
        logger.warning("GATEWAY_ANON_KEY=%r not found in keys.yaml; anonymous access disabled", anon_key_id)
    elif anon_policy is not None:
        logger.info("Anonymous access enabled via key_id=%s (GATEWAY_ANON_KEY)", anon_policy.key_id)
    return build_starlette_app(registry, key_store, mcp_path=mcp_path, tool_rag_enabled=tool_rag_enabled, tool_db=tool_db, tool_rag_router=tool_rag_router, server_health=server_health, retriever=retriever, anon_policy=anon_policy, planner=planner, max_parallel=max_parallel, catalog=catalog, skills=skills, recipes=recipes, config_paths=(reg, reg_generated, keys), anon_key_id=anon_key_id or None)
