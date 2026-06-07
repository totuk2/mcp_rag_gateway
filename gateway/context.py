"""Request-scoped context for the active API key policy (set by ASGI middleware)."""

from __future__ import annotations

import contextvars

from gateway.policy import AccessPolicy

current_policy: contextvars.ContextVar[AccessPolicy | None] = contextvars.ContextVar(
    "mcp_gateway_policy", default=None
)


def get_policy() -> AccessPolicy:
    p = current_policy.get()
    if p is None:
        raise RuntimeError("MCP gateway policy is not set (middleware bug)")
    return p
