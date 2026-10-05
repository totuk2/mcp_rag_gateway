"""Hot reload of keys, registry and skills — no gateway restart.

A restart drops every client's MCP session (LibreChat then fails with
"Connection closed" until it reconnects), so config changes are applied in
place instead: every CONFIG_RELOAD_INTERVAL seconds (default 10; 0 = off) the
reloader compares file mtimes and
  - keys.yaml            -> swaps the KeyStore's keys (next request uses them);
  - registry*.yaml       -> swaps the shared Registry's servers in place, then the
                            app re-syncs upstreams, refreshes the catalog and
                            reindexes (callback), and closes sticky sessions of
                            changed/removed servers;
  - skills/*/SKILL.md    -> reloads the SkillStore.
A file that fails to parse is reported and the previous config stays active.
POST /tool-rag/reload (admin) forces a check.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from gateway.auth import KeyStore, load_keys
from gateway.registry import Registry, load_registries

logger = logging.getLogger(__name__)


def _mtime(p: Path | None) -> float | None:
    try:
        return p.stat().st_mtime if p is not None else None
    except OSError:
        return None


class ConfigReloader:
    def __init__(self, registry: Registry, key_store: KeyStore, registry_path: Path,
                 generated_path: Path | None, keys_path: Path, skills: Any = None,
                 on_registry_change: Callable[[dict[str, list[str]]], Awaitable[None]] | None = None) -> None:
        self.registry = registry
        self.key_store = key_store
        self.registry_path = registry_path
        self.generated_path = generated_path
        self.keys_path = keys_path
        self.skills = skills
        self.on_registry_change = on_registry_change
        self._lock = asyncio.Lock()
        self._stamps = self._current()

    def _skills_stamp(self) -> tuple:
        d = getattr(self.skills, "_dir", None)
        if d is None or not Path(d).is_dir():
            return ()
        return tuple(sorted((str(p), _mtime(p)) for p in Path(d).glob("*/SKILL.md")))

    def _current(self) -> dict[str, Any]:
        return {
            "registry": (_mtime(self.registry_path), _mtime(self.generated_path)),
            "keys": _mtime(self.keys_path),
            "skills": self._skills_stamp(),
        }

    async def check(self, force: bool = False) -> dict[str, Any]:
        """Apply whatever changed (everything when `force`). Returns a summary."""
        async with self._lock:
            now = self._current()
            done: dict[str, Any] = {}
            if force or now["keys"] != self._stamps["keys"]:
                try:
                    new = load_keys(self.keys_path)
                    self.key_store.replace(new)
                    done["keys"] = "reloaded"
                    logger.info("config reload: keys.yaml reloaded")
                except Exception as e:
                    done["keys"] = f"error: {e}"
                    logger.error("config reload: keys.yaml invalid, keeping previous keys: %s", e)
            if force or now["registry"] != self._stamps["registry"]:
                try:
                    new_reg = load_registries(self.registry_path, self.generated_path)
                    old = dict(self.registry.servers)
                    diff = {
                        "added": sorted(set(new_reg.servers) - set(old)),
                        "removed": sorted(set(old) - set(new_reg.servers)),
                        "changed": sorted(s for s in set(old) & set(new_reg.servers) if old[s] != new_reg.servers[s]),
                    }
                    self.registry.servers.clear()
                    self.registry.servers.update(new_reg.servers)
                    done["registry"] = diff
                    logger.info("config reload: registry %s", diff)
                    if self.on_registry_change is not None and (force or any(diff.values())):
                        await self.on_registry_change(diff)
                except Exception as e:
                    done["registry"] = f"error: {e}"
                    logger.error("config reload: registry invalid, keeping previous servers: %s", e)
            if self.skills is not None and (force or now["skills"] != self._stamps["skills"]):
                try:
                    self.skills.load()
                    done["skills"] = "reloaded"
                except Exception as e:
                    done["skills"] = f"error: {e}"
                    logger.error("config reload: skills failed to load: %s", e)
            self._stamps = now
            return done

    async def loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("config reload check failed")
