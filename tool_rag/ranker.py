"""Ranking logic for Tool-RAG retrieval results.

Per spec section 7.6, final score = weighted combination of:
  semantic_score   — cosine similarity from vector search [0-1]
  keyword_boost    — BM25-like TF overlap in name/desc/tags [0-0.25]
  metadata_boost   — tool_type/permission/server match [0-0.15]
  policy_penalty   — filter mismatch [-1-0]

Tie-breaker: tool_id lexicographic.
"""

from __future__ import annotations

import re
from typing import Sequence

from gateway.tool_record import ToolRecord


class Ranker:
    """Computes final scores for tool retrieval candidates."""

    def __init__(
        self,
        semantic_weight: float = 1.0,
        keyword_weight: float = 0.25,
        metadata_weight: float = 0.15,
        policy_penalty: float = -1.0,
    ):
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        self._metadata_weight = metadata_weight
        self._policy_penalty = policy_penalty

    def compute_score(
        self,
        query: str,
        record: ToolRecord,
        semantic_score: float,
        allowed_servers: Sequence[str] | None = None,
        permission_scope: str | None = None,
        tool_type: str | None = None,
    ) -> float:
        """Compute a single combined score for one tool candidate."""
        kw = self._keyword_score(query, record)
        md = self._metadata_score(record, allowed_servers, permission_scope, tool_type)
        penalty = 0.0
        if allowed_servers and record.server_id not in allowed_servers:
            penalty += self._policy_penalty
        if permission_scope and record.permission_scope != permission_scope:
            penalty += self._policy_penalty * 0.5
        if tool_type and record.tool_type != tool_type:
            penalty += self._policy_penalty * 0.5
        score = (
            self._semantic_weight * semantic_score
            + self._keyword_weight * kw
            + self._metadata_weight * md
            + penalty
        )
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _keyword_score(query: str, record: ToolRecord) -> float:
        """Token overlap in name/desc/tags [0-1]."""
        q_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        if not q_tokens:
            return 0.0
        target = (
            record.tool_name.lower()
            + " "
            + record.description.lower()
            + " "
            + " ".join(record.tags).lower()
        )
        matches = sum(1 for t in q_tokens if t in target)
        return matches / len(q_tokens)

    @staticmethod
    def _metadata_score(
        record: ToolRecord,
        allowed_servers: Sequence[str] | None,
        permission_scope: str | None,
        tool_type: str | None,
    ) -> float:
        """Metadata alignment boost [0-1]."""
        score = 0.0
        if allowed_servers and record.server_id in allowed_servers:
            score += 0.5
        if permission_scope and record.permission_scope == permission_scope:
            score += 0.3
        if tool_type and record.tool_type == tool_type:
            score += 0.2
        return min(1.0, score)

    @staticmethod
    def reason(query: str, record: ToolRecord, score: float) -> str:
        """Short human-readable reason for the score."""
        if score >= 0.8:
            return f"Strong semantic match for '{query}'."
        if score >= 0.5:
            return f"Relevant to '{query}' (score={score:.2f})."
        if score >= 0.2:
            return f"Partial match for '{query}'."
        return f"Low relevance to '{query}'."
