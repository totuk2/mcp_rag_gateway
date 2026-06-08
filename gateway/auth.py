"""Load API keys and resolve Bearer tokens to AccessPolicy."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any

import yaml

from gateway.policy import AccessPolicy, ServerRule


def _parse_server_rule(raw: Any) -> ServerRule:
    if raw is None or raw == {}:
        return ServerRule()
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid server rule (expected mapping): {raw!r}")
    return ServerRule(
        tool_prefixes=tuple(raw.get("tool_prefixes") or ()),
        uri_prefixes=tuple(raw.get("uri_prefixes") or ()),
        prompt_prefixes=tuple(raw.get("prompt_prefixes") or ()),
    )


class KeyStore:
    """Resolve raw Bearer token to AccessPolicy."""

    def __init__(self, plaintext: dict[str, AccessPolicy], hashed: list[tuple[bytes, AccessPolicy]]):
        self._plaintext = plaintext
        self._hashed = hashed

    @classmethod
    def from_yaml(cls, path: Path) -> KeyStore:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("Keys file must be a YAML mapping")
        plaintext: dict[str, AccessPolicy] = {}
        hashed: list[tuple[bytes, AccessPolicy]] = []
        for entry in doc.get("keys") or []:
            key_id = entry["id"]
            servers_raw = entry.get("servers") or {}
            if not isinstance(servers_raw, dict):
                raise ValueError(f"keys[{key_id!r}].servers must be a mapping")
            servers: dict[str, ServerRule] = {}
            for sid, rule in servers_raw.items():
                servers[sid] = _parse_server_rule(rule)
            admin = bool(entry.get("admin", False))
            policy = AccessPolicy(key_id=key_id, servers=servers, admin=admin)
            if "secret" in entry:
                plaintext[str(entry["secret"])] = policy
            elif "secret_hash" in entry:
                h = str(entry["secret_hash"])
                if not h.startswith("sha256:"):
                    raise ValueError("secret_hash must be sha256:<hex>")
                digest = bytes.fromhex(h.split(":", 1)[1])
                hashed.append((digest, policy))
            else:
                raise ValueError(f"Key {key_id!r} needs secret or secret_hash")
        return cls(plaintext, hashed)

    def resolve(self, token: str) -> AccessPolicy | None:
        hit = self._plaintext.get(token)
        if hit:
            return hit
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for expected, policy in self._hashed:
            if hmac.compare_digest(expected, digest):
                return policy
        return None

    def by_id(self, key_id: str) -> AccessPolicy | None:
        for policy in self._plaintext.values():
            if policy.key_id == key_id:
                return policy
        for _, policy in self._hashed:
            if policy.key_id == key_id:
                return policy
        return None


def load_keys(path: Path) -> KeyStore:
    return KeyStore.from_yaml(path)
