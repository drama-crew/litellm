"""Task-scoped OTLP traces for durable Causyn jobs.

Vendored into independently deployed services. Only safe scalar attributes are
accepted. A task has a stable trace/root identity across restarts; individual
attempts have random span IDs. The lifecycle root may be exported on completion.
No global OTel provider is replaced and exporting never runs on the GPU thread.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span, Tracer
    from opentelemetry.util.types import AttributeValue

_task: contextvars.ContextVar[str] = contextvars.ContextVar("causyn_task", default="")
_root: contextvars.ContextVar[bool] = contextvars.ContextVar("causyn_root", default=False)


@dataclass
class ExporterState:
    provider: TracerProvider | None = None
    warned: bool = False


_state = ExporterState()
_lock = threading.Lock()
_SAFE = re.compile(
    r"^(?:task_id|attempt|rank|frames|tokens|nfe|step|step_seconds|duration_seconds|queue_seconds|ark_task_id|native_request_id|compiler_[a-z_]+|graph_[a-z_]+|decode_[a-z_]+|gc_[a-z_]+|phase|outcome|error_type|model|mode|width|height)$"
)


def identity(task_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(("causyn.video.v1:" + task_id.removeprefix("h3_ir_")).encode()).digest()
    return int.from_bytes(digest[:16], "big") or 1, int.from_bytes(digest[16:24], "big") or 1


def _tracer() -> Tracer | None:
    if _state.provider is not None:
        return _state.provider.get_tracer("causyn.video.stages", "1")
    endpoint = os.getenv("CAUSYN_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint:
        return None
    with suppress(Exception):
        with _lock:
            if _state.provider is None:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
                from opentelemetry.sdk.trace.sampling import ALWAYS_ON

                class TaskIds(RandomIdGenerator):
                    def generate_trace_id(self) -> int:
                        return identity(_task.get())[0] if _task.get() else super().generate_trace_id()

                    def generate_span_id(self) -> int:
                        return identity(_task.get())[1] if _root.get() else super().generate_span_id()

                provider = TracerProvider(
                    resource=Resource.create(
                        {"service.name": os.getenv("CAUSYN_OTEL_SERVICE_NAME", __package__ or "causyn")}
                    ),
                    sampler=ALWAYS_ON,
                    id_generator=TaskIds(),
                )
                ingest_key = os.getenv("CAUSYN_OTEL_INGEST_KEY", "")
                key_file = os.getenv("CAUSYN_OTEL_INGEST_KEY_FILE")
                if key_file:
                    ingest_key = Path(key_file).read_text().strip()
                headers = {"X-Drama-Otel-Key": ingest_key} if ingest_key else {}
                provider.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=3),
                        max_queue_size=2048,
                        max_export_batch_size=128,
                        schedule_delay_millis=1000,
                        export_timeout_millis=4000,
                    )
                )
                _state.provider = provider
        return _state.provider.get_tracer("causyn.video.stages", "1")
    if not _state.warned:
        logging.getLogger(__name__).warning("Causyn stage exporter unavailable")
        _state.warned = True
    return None


def _attributes(values: Mapping[str, object]) -> Mapping[str, AttributeValue]:
    return {
        "causyn." + k: safe
        for k, v in values.items()
        if _SAFE.fullmatch(k)
        if (safe := _attribute_value(v)) is not None
    }


def _attribute_value(value: object) -> AttributeValue | None:
    if isinstance(value, (str, bool, int, float)):
        return None if isinstance(value, str) and len(value) > 160 else value
    if isinstance(value, (list, tuple)) and len(value) <= 32:
        numbers = tuple(x for x in value if isinstance(x, (int, float)))
        return numbers if len(numbers) == len(value) else None
    return None


@contextmanager
def task_scope(task_id: str) -> Iterator[None]:
    token = _task.set(task_id.removeprefix("h3_ir_") if isinstance(task_id, str) else "")
    try:
        yield
    finally:
        _task.reset(token)


def current_task() -> str:
    return _task.get()


def _start(name: str, start_ns: int, attrs: Mapping[str, object], *, root: bool = False) -> Span | None:
    with suppress(Exception):
        tracer = _tracer()
        if tracer is None or not _task.get():
            return None
        from opentelemetry import trace
        from opentelemetry.context import Context

        tid, sid = identity(_task.get())
        current = trace.get_current_span().get_span_context()
        if root:
            ctx = Context()
        elif current.is_valid and current.trace_id == tid:
            ctx = trace.set_span_in_context(trace.get_current_span())
        else:
            ctx = trace.set_span_in_context(
                trace.NonRecordingSpan(trace.SpanContext(tid, sid, is_remote=True, trace_flags=trace.TraceFlags(1)))
            )
        token = _root.set(root)
        try:
            return tracer.start_span(
                name,
                context=ctx,
                start_time=start_ns,
                attributes=_attributes({"task_id": _task.get(), **attrs}),
            )
        finally:
            _root.reset(token)
    return None


def interval(name: str, start_ns: int, end_ns: int, *, root: bool = False, **attrs: object) -> None:
    """Export an observed interval (e.g. durable queue boundaries), not a live timer."""
    if end_ns < start_ns:
        return
    span = _start(name, start_ns, attrs, root=root)
    if span is not None:
        with suppress(Exception):
            if attrs.get("outcome") == "failed":
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR))
            span.end(end_time=end_ns)


@contextmanager
def stage(name: str, **attrs: object) -> Iterator[None]:
    start_ns, tick = time.time_ns(), time.perf_counter_ns()
    span = _start(name, start_ns, attrs)
    token = None
    with suppress(Exception):
        if span is not None:
            from opentelemetry import context, trace

            token = context.attach(trace.set_span_in_context(span))
    try:
        yield
    except BaseException as error:
        if span is not None:
            with suppress(Exception):
                from opentelemetry.trace import Status, StatusCode

                span.set_attribute("causyn.error_type", type(error).__name__)
                span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        if token is not None:
            with suppress(Exception):
                from opentelemetry import context

                context.detach(token)
        if span is not None:
            with suppress(Exception):
                span.end(end_time=start_ns + time.perf_counter_ns() - tick)
