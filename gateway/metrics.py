"""Per-tool call metrics (in memory, since gateway start).

Records every tool call twice where it applies: once as the client-facing call
(meta-tool or direct upstream name, incl. time spent as a background job) and once
per upstream call made by `_dispatch_tool` (run_tool / run_tools / delegate),
with the upstream result size before and after shaping (compaction, projection,
truncation). Exposed as JSON (GET /tool-rag/metrics/tools) and in Prometheus text
format (GET /tool-rag/metrics/prometheus).
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

_RECENT = 200


@dataclass
class _Stat:
    calls: int = 0
    errors: int = 0
    ms_sum: float = 0.0
    ms_max: float = 0.0
    raw_chars: int = 0
    sent_chars: int = 0
    jobs: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=_RECENT))


class Metrics:
    def __init__(self) -> None:
        self.started = time.time()
        self._tools: dict[tuple[str, str], _Stat] = defaultdict(_Stat)
        self._keys: dict[str, int] = defaultdict(int)

    def record(self, tool: str, kind: str, ok: bool, ms: float, key_id: str | None = None,
               raw_chars: int | None = None, sent_chars: int | None = None) -> None:
        s = self._tools[(kind, tool)]
        s.calls += 1
        s.errors += 0 if ok else 1
        s.ms_sum += ms
        s.ms_max = max(s.ms_max, ms)
        s.recent.append(ms)
        if raw_chars is not None:
            s.raw_chars += raw_chars
        if sent_chars is not None:
            s.sent_chars += sent_chars
        if key_id and kind == "call":
            self._keys[key_id] += 1

    def job_started(self, tool: str) -> None:
        self._tools[("call", tool)].jobs += 1

    def snapshot(self, include_keys: bool = False) -> dict:
        rows = []
        for (kind, tool), s in sorted(self._tools.items(), key=lambda kv: -kv[1].calls):
            lat = sorted(s.recent)
            rows.append({
                "tool": tool, "kind": kind, "calls": s.calls, "errors": s.errors,
                "avg_ms": round(s.ms_sum / s.calls) if s.calls else 0,
                "p50_ms": round(statistics.median(lat)) if lat else 0,
                "p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))]) if lat else 0,
                "max_ms": round(s.ms_max),
                **({"raw_chars": s.raw_chars, "sent_chars": s.sent_chars} if s.raw_chars else {}),
                **({"background_jobs": s.jobs} if s.jobs else {}),
            })
        out = {"since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)), "tools": rows}
        if include_keys:
            out["calls_by_key"] = dict(sorted(self._keys.items(), key=lambda kv: -kv[1]))
        return out

    def prometheus(self) -> str:
        def esc(v: str) -> str:
            return v.replace("\\", "\\\\").replace('"', '\\"')
        lines = [
            "# HELP gateway_tool_calls_total Tool calls by tool, kind (call|upstream) and status.",
            "# TYPE gateway_tool_calls_total counter",
        ]
        for (kind, tool), s in self._tools.items():
            l = f'tool="{esc(tool)}",kind="{kind}"'
            lines.append(f'gateway_tool_calls_total{{{l},status="ok"}} {s.calls - s.errors}')
            lines.append(f'gateway_tool_calls_total{{{l},status="error"}} {s.errors}')
        lines += ["# HELP gateway_tool_latency_seconds Tool call latency.", "# TYPE gateway_tool_latency_seconds summary"]
        for (kind, tool), s in self._tools.items():
            l = f'tool="{esc(tool)}",kind="{kind}"'
            lines.append(f"gateway_tool_latency_seconds_sum{{{l}}} {s.ms_sum / 1000:.3f}")
            lines.append(f"gateway_tool_latency_seconds_count{{{l}}} {s.calls}")
        lines += ["# HELP gateway_tool_result_chars_total Upstream result size before (raw) and after shaping (sent).",
                  "# TYPE gateway_tool_result_chars_total counter"]
        for (kind, tool), s in self._tools.items():
            if s.raw_chars:
                l = f'tool="{esc(tool)}"'
                lines.append(f'gateway_tool_result_chars_total{{{l},stage="raw"}} {s.raw_chars}')
                lines.append(f'gateway_tool_result_chars_total{{{l},stage="sent"}} {s.sent_chars}')
        lines += ["# HELP gateway_background_jobs_total Calls that continued as background jobs.",
                  "# TYPE gateway_background_jobs_total counter"]
        for (kind, tool), s in self._tools.items():
            if s.jobs:
                lines.append(f'gateway_background_jobs_total{{tool="{esc(tool)}"}} {s.jobs}')
        return "\n".join(lines) + "\n"
