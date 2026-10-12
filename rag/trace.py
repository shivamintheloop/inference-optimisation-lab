"""Minimal tracing: one JSON line per request, shaped like OpenTelemetry.

    with trace("query", config=CONFIG.snapshot()):
        with span("retrieve.vector", top_k=30) as s:
            hits = search(...)
            s.set("hits", len(hits))

A trace collects every span opened inside it and, when it ends, appends one
line to traces/<kind>-YYYY-MM-DD.jsonl. Field names (trace_id, span_id,
parent_id, attrs) follow OpenTelemetry, so moving to the OTel SDK later only
changes this file.

span() outside a trace does nothing, so library code can always call it.
Nesting uses contextvars, which also keeps concurrent asyncio tasks apart.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rag.config import CONFIG

_current_trace: ContextVar["Trace | None"] = ContextVar("current_trace", default=None)
_current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    start_ms: float                   # offset from the start of the trace
    duration_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_ms": round(self.start_ms, 2),
            "duration_ms": round(self.duration_ms or 0.0, 2),
            "status": self.status,
            "error": self.error,
            "attrs": self.attrs,
        }


class _NoopSpan:
    """Returned by span() when no trace is active."""

    def set(self, key: str, value: Any) -> None:
        pass


_NOOP = _NoopSpan()


def config_hash(config: dict | None) -> str | None:
    if config is None:
        return None
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:8]


class Trace:
    def __init__(self, kind: str, config: dict | None = None):
        self.trace_id = uuid.uuid4().hex
        self.kind = kind
        self.config = config
        self.start = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.duration_ms: float | None = None
        self.status = "ok"
        self.error: str | None = None
        self.attrs: dict[str, Any] = {}
        self.spans: list[Span] = []
        self._t0 = time.perf_counter()

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "start": self.start,
            "duration_ms": round(self.duration_ms or 0.0, 2),
            "status": self.status,
            "error": self.error,
            "config_hash": config_hash(self.config),
            "config": self.config,
            "attrs": self.attrs,
            "spans": [s.to_dict() for s in self.spans],
        }

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        day = self.start[:10]
        path = out_dir / f"{self.kind}-{day}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(self.to_dict(), default=str) + "\n")
        return path


@contextmanager
def trace(kind: str, config: dict | None = None,
          out_dir: Path | None = None) -> Iterator[Trace]:
    """Open a trace. It is written to disk when the block ends, even on error."""
    t = Trace(kind, config)
    token = _current_trace.set(t)
    try:
        yield t
    except BaseException as e:
        t.status = "error"
        t.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        t.duration_ms = t.now_ms()
        _current_trace.reset(token)
        t.write(out_dir or CONFIG.traces_dir)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span | _NoopSpan]:
    """Time one step inside the current trace."""
    t = _current_trace.get()
    if t is None:
        yield _NOOP
        return

    parent = _current_span.get()
    s = Span(
        name=name,
        span_id=uuid.uuid4().hex[:16],
        parent_id=parent.span_id if parent else None,
        start_ms=t.now_ms(),
        attrs=dict(attrs),
    )
    t.spans.append(s)
    token = _current_span.set(s)
    try:
        yield s
    except BaseException as e:
        s.status = "error"
        s.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        s.duration_ms = t.now_ms() - s.start_ms
        _current_span.reset(token)


def current_trace() -> Trace | None:
    return _current_trace.get()
