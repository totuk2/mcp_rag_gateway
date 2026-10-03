"""Skills: markdown playbooks that tell an agent HOW to tackle a class of tasks.

Tools say what is possible; a skill encodes the procedure (which tools, in what
order, what to do when results are thin, when to stop). Each skill lives in
`skills/<name>/SKILL.md` with YAML frontmatter:

    ---
    name: hard-flight-routing
    description: one paragraph — when to use it (also used for matching)
    requires_servers: [flight-research]
    ---
    <markdown body>

A skill is visible to a key only if the key is granted every `requires_servers`
entry. Skills surface via the `get_skill` meta-tool, the `browse_tools` overview,
coverage verdicts and `find_tools` hints (`match()`, embedding similarity).
SKILLS_DIR overrides the directory (default: <repo>/skills).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from gateway.policy import AccessPolicy
    from tool_rag.embedder import Embedder

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "skills"
# Cosine (normalized embeddings) above which a query "matches" a skill. Override
# with SKILLS_MATCH_THRESHOLD.
DEFAULT_MATCH_THRESHOLD = 0.5
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    requires_servers: tuple[str, ...]
    body: str

    def brief(self) -> dict:
        return {"name": self.name, "description": self.description}


def _visible(skill: Skill, policy: "AccessPolicy | None") -> bool:
    if policy is None:  # TOOL_RAG_WITHOUT_AUTH: no policy in context by design
        return True
    return all(policy.allows_server(s) for s in skill.requires_servers)


class SkillStore:
    def __init__(self, directory: str | Path | None = None, embedder: "Embedder | None" = None) -> None:
        self._dir = Path(directory or os.environ.get("SKILLS_DIR") or DEFAULT_DIR)
        self._embedder = embedder
        self._skills: dict[str, Skill] = {}
        self._vectors: dict[str, list[float]] = {}
        try:
            self._threshold = float(os.environ.get("SKILLS_MATCH_THRESHOLD", DEFAULT_MATCH_THRESHOLD))
        except ValueError:
            self._threshold = DEFAULT_MATCH_THRESHOLD
        self.load()

    def load(self) -> None:
        skills: dict[str, Skill] = {}
        for path in sorted(self._dir.glob("*/SKILL.md")) if self._dir.is_dir() else []:
            m = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
            if not m:
                logger.warning("skill %s has no frontmatter; skipped", path)
                continue
            meta = yaml.safe_load(m.group(1)) or {}
            name = str(meta.get("name") or path.parent.name)
            skills[name] = Skill(
                name=name,
                description=" ".join(str(meta.get("description") or "").split()),
                requires_servers=tuple(str(s) for s in meta.get("requires_servers") or ()),
                body=m.group(2).strip(),
            )
        self._skills = skills
        self._vectors = {}
        logger.info("Loaded %d skill(s) from %s: %s", len(skills), self._dir, ", ".join(skills) or "-")

    def __bool__(self) -> bool:
        return bool(self._skills)

    def visible(self, policy: "AccessPolicy | None") -> list[Skill]:
        return [s for s in self._skills.values() if _visible(s, policy)]

    def get(self, name: str, policy: "AccessPolicy | None") -> Skill | None:
        s = self._skills.get((name or "").strip())
        return s if s is not None and _visible(s, policy) else None

    async def match(self, query: str, policy: "AccessPolicy | None") -> list[tuple[Skill, float]]:
        """Visible skills whose description is semantically close to `query`, best first."""
        cands = self.visible(policy)
        if not cands or self._embedder is None or not query.strip():
            return []
        try:
            missing = [s for s in cands if s.name not in self._vectors]
            if missing:
                vecs = await asyncio.to_thread(self._embedder.embed_many, [s.description for s in missing])
                self._vectors.update({s.name: v for s, v in zip(missing, vecs)})
            q = await asyncio.to_thread(self._embedder.embed_query, query)
        except Exception:
            logger.debug("skill matching failed", exc_info=True)
            return []
        scored = [(s, sum(a * b for a, b in zip(q, self._vectors[s.name]))) for s in cands]
        hits = [(s, round(score, 3)) for s, score in scored if score >= self._threshold]
        return sorted(hits, key=lambda x: (-x[1], x[0].name))
