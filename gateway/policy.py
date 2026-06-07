"""Per-API-key access rules: allowed servers and optional prefix filters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServerRule:
    """Restrictions for one upstream server (default: full access)."""

    tool_prefixes: tuple[str, ...] = ()
    uri_prefixes: tuple[str, ...] = ()
    prompt_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AccessPolicy:
    """Resolved policy for one HTTP request (one API key)."""

    key_id: str
    servers: dict[str, ServerRule] = field(default_factory=dict)
    admin: bool = False

    def allows_server(self, server_id: str) -> bool:
        return server_id in self.servers

    def tool_visible(self, server_id: str, tool_name: str) -> bool:
        rule = self.servers.get(server_id)
        if rule is None:
            return False
        if not rule.tool_prefixes:
            return True
        return any(tool_name.startswith(p) for p in rule.tool_prefixes)

    def uri_visible(self, server_id: str, uri: str) -> bool:
        rule = self.servers.get(server_id)
        if rule is None:
            return False
        if not rule.uri_prefixes:
            return True
        return any(uri.startswith(p) for p in rule.uri_prefixes)

    def prompt_visible(self, server_id: str, prompt_name: str) -> bool:
        rule = self.servers.get(server_id)
        if rule is None:
            return False
        if not rule.prompt_prefixes:
            return True
        return any(prompt_name.startswith(p) for p in rule.prompt_prefixes)
