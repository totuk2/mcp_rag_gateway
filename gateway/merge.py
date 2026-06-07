"""Namespacing merged tools, prompts, and gateway resource URIs."""

from __future__ import annotations

import base64
import re
from urllib.parse import urlparse

from pydantic import AnyUrl

# Merged tool/prompt names: {server_id}__{original_name}
# server_id must not contain "__" (validated in registry / docs).
_SEP = "__"


def merged_tool_name(server_id: str, original: str) -> str:
    if _SEP in server_id:
        raise ValueError(f"server_id must not contain {_SEP!r}: {server_id!r}")
    return f"{server_id}{_SEP}{original}"


def split_merged_name(merged: str) -> tuple[str, str]:
    if _SEP not in merged:
        raise ValueError("Invalid merged name (missing server prefix)")
    server_id, rest = merged.split(_SEP, 1)
    if not server_id:
        raise ValueError("Invalid merged name (empty server id)")
    return server_id, rest


def merged_prompt_name(server_id: str, original: str) -> str:
    return merged_tool_name(server_id, original)


def gateway_resource_uri(server_id: str, original_uri: str) -> AnyUrl:
    """Opaque gateway URI that routes read_resource to one upstream."""
    if _SEP in server_id:
        raise ValueError(f"server_id must not contain {_SEP!r}: {server_id!r}")
    payload = original_uri.encode("utf-8")
    b64 = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    # host = server_id, path = / + b64 (URL-safe)
    return AnyUrl(f"gateway://{server_id}/{b64}")


def parse_gateway_resource_uri(uri: str | AnyUrl) -> tuple[str, str]:
    u = str(uri)
    parsed = urlparse(u)
    if parsed.scheme != "gateway":
        raise ValueError("Not a gateway resource URI")
    server_id = parsed.netloc or parsed.hostname or ""
    if not server_id:
        raise ValueError("Gateway URI missing server id (host)")
    path = (parsed.path or "").lstrip("/")
    if not path:
        raise ValueError("Gateway URI missing path segment")
    # restore base64 padding
    pad = "=" * ((4 - len(path) % 4) % 4)
    raw = base64.urlsafe_b64decode(path + pad)
    return server_id, raw.decode("utf-8")


_SERVER_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def assert_safe_server_id(server_id: str) -> None:
    if not _SERVER_ID_RE.match(server_id):
        raise ValueError(
            f"Invalid server_id {server_id!r}: use only ASCII letters, digits, ._- "
            f"(and no {_SEP})"
        )
