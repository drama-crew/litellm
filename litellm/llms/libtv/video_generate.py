"""Two-stage enqueue/poll primitives for the internal video-generation worker.

Mirrors ``litellm/llms/libtv/validated_transfer.py``'s fail-closed URL
allowlist and settings-parsing conventions, but the write path is
asynchronous (``POST`` enqueues, ``GET`` polls) instead of a blocking
``BRPOP`` round-trip: a video generation runs minutes rather than seconds,
and holding an HTTP connection open that long trips uvicorn/caddy/k8s idle
timeouts. See
``docs/superpowers/specs/2026-08-15-internal-video-worker-protocol.md``
sections 1-2 for the authoritative protocol this module implements.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

from .transfer import STATUS_TTL_SECONDS, WORKER_HEARTBEAT_WINDOW_SECONDS, result_key, status_key

# Task-type namespace for this worker task type. Status/result keys are
# generic (parameterized by task_id, reused as-is from transfer.py per spec
# §2.3 "无新键"); the stream/alive-zset/capacity-hash keys are per-task-type
# and transfer.py only has pre-formatted constants for its own two task
# types, so video_generate needs its own key-derivation helpers below.
TASK_TYPE_VIDEO_GENERATE = "video_generate"

STATUS_QUEUED = "queued"
STATUS_CLAIMED = "claimed"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_STATUS_TO_PUBLIC = {
    STATUS_QUEUED: "queued",
    STATUS_CLAIMED: "claimed",
    STATUS_DONE: "succeeded",
    STATUS_FAILED: "failed",
    STATUS_CANCELLED: "failed",
}

# Deadline/staging-upload defaults for the envelope built by enqueue_video_generate
# (plan Step 3): 15 minute generation budget, presign TTL comfortably >= deadline.
DEADLINE_SECONDS = 15 * 60
STAGING_UPLOAD_TTL_SECONDS = 1200

_REQUIRED_RESULT_FIELDS = ("staging_key", "bytes", "content_type", "duration_seconds")

_ALLOWED_TOP_LEVEL_KEYS = frozenset({"task_id", "model", "deadline_ts", "request", "staging_upload"})


class VideoGenerateError(Exception):
    """A classified video-generate failure.

    ``code`` is one of the four deterministic-rejection codes from spec
    §1.1 (``invalid_url`` | ``no_worker_available`` | ``no_capacity_available``
    | ``invalid_params``) -- all mapped to HTTP 422 by the endpoint layer --
    or any other code (e.g. ``misconfigured`` for an empty/malformed URL
    allowlist) for a fail-closed configuration error, mapped to HTTP 503.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def stream_key(task_type: str) -> str:
    return f"worker:tasks:{task_type}"


def alive_zset_key(task_type: str) -> str:
    return f"worker:alive:{task_type}"


def capacity_key(task_type: str) -> str:
    """Hash of worker_name -> free_slots, written by the heartbeat handler
    alongside the alive zset (spec §3.4). The protocol spec describes this
    key only in prose ("与 alive zset 并列的键"), unlike the alive zset which
    is spelled out as an exact literal -- this name mirrors the alive zset's
    prefix-swap convention (worker:alive:{type} -> worker:capacity:{type})
    and must be cross-checked against Task 3's actual heartbeat-handler key
    name before relying on it in production.
    """
    return f"worker:capacity:{task_type}"


def _parse_hosts(value: str, name: str) -> frozenset[str]:
    """Fail-closed host-list parsing, mirroring
    ``validated_transfer._parse_hosts``: empty or wildcard/path/userinfo-
    bearing entries are a configuration error (503), never a lenient
    fallback."""
    hosts = frozenset(host.strip().lower() for host in value.split(",") if host.strip())
    if not hosts or any("*" in host or "/" in host or "@" in host for host in hosts):
        raise VideoGenerateError("misconfigured", f"{name} must contain exact hosts")
    return hosts


def _parse_bool_env(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class VideoGenerateSettings:
    source_hosts: frozenset[str]
    target_hosts: frozenset[str]
    allow_http: bool = False

    @classmethod
    def from_environment(cls) -> "VideoGenerateSettings":
        # Deliberately dedicated env vars -- spec §3 forbids reusing
        # LIBTV_VALIDATED_TRANSFER_* or adding a LIBTV_PLATFORM_SOURCE_HOSTS-
        # style fallback; this allowlist is independent of every other libtv
        # transfer path.
        source_hosts = _parse_hosts(
            os.getenv("LIBTV_VIDEO_GENERATE_SOURCE_HOSTS", ""), "LIBTV_VIDEO_GENERATE_SOURCE_HOSTS"
        )
        target_hosts = _parse_hosts(
            os.getenv("LIBTV_VIDEO_GENERATE_TARGET_HOSTS", ""), "LIBTV_VIDEO_GENERATE_TARGET_HOSTS"
        )
        allow_http = _parse_bool_env(os.getenv("LIBTV_VIDEO_GENERATE_ALLOW_HTTP", "false"))
        return cls(source_hosts=source_hosts, target_hosts=target_hosts, allow_http=allow_http)


def _check_url(raw_url: Any, allowed_hosts: frozenset[str], settings: VideoGenerateSettings, role: str) -> None:
    """Fail-closed URL allowlist check. Ported from the exact shape of
    ``validated_transfer.ValidatedTransferSettings._validate_url`` (spec §5
    explicitly instructs reusing that shape): exact-hostname match,
    https-only unless allow_http, no userinfo/fragment, port restricted to
    443 (https) or 80 (http)."""
    if not isinstance(raw_url, str) or not raw_url:
        raise VideoGenerateError("invalid_url", f"invalid {role} URL")
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError as exc:
        raise VideoGenerateError("invalid_url", f"invalid {role} URL") from exc
    schemes = {"https"} | ({"http"} if settings.allow_http else set())
    if (
        parsed.scheme not in schemes
        or not parsed.hostname
        or parsed.hostname.lower() not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and port not in ({443} if parsed.scheme == "https" else {80}))
    ):
        raise VideoGenerateError("invalid_url", f"invalid {role} URL")


def _iter_reference_urls(payload: dict) -> Iterator[str]:
    """Yield reference-media URLs from the payload.

    Reads references from the spec-correct nested location
    (``payload["request"]["references"]``) AND, defensively, from a
    top-level ``references`` key if present. The top-level fallback is not
    part of the documented request shape (spec §1.1 nests references under
    ``request``); it exists so a stray top-level references list is still
    caught by the same fail-closed URL check instead of silently reaching
    generic extra-field rejection first (see the implementing task's final
    report for the deviation this resolves).
    """
    request = payload.get("request")
    sources = []
    if isinstance(request, dict):
        sources.append(request.get("references"))
    sources.append(payload.get("references"))
    for references in sources:
        if not isinstance(references, list):
            continue
        for item in references:
            if isinstance(item, dict) and isinstance(item.get("url"), str):
                yield item["url"]


def _validate_urls(payload: dict, settings: VideoGenerateSettings) -> None:
    for url in _iter_reference_urls(payload):
        _check_url(url, settings.source_hosts, settings, "reference")
    staging_upload = payload.get("staging_upload")
    if isinstance(staging_upload, dict) and "url" in staging_upload:
        _check_url(staging_upload.get("url"), settings.target_hosts, settings, "staging upload")


def _validate_shape(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise VideoGenerateError("invalid_params", "request body must be a JSON object")
    extra = set(payload) - _ALLOWED_TOP_LEVEL_KEYS
    if extra:
        raise VideoGenerateError("invalid_params", f"unrecognized field(s): {', '.join(sorted(extra))}")
    missing = _ALLOWED_TOP_LEVEL_KEYS - set(payload)
    if missing:
        raise VideoGenerateError("invalid_params", f"missing required field(s): {', '.join(sorted(missing))}")
    if not isinstance(payload.get("task_id"), str) or not payload["task_id"]:
        raise VideoGenerateError("invalid_params", "task_id must be a non-empty string")
    if not isinstance(payload.get("request"), dict):
        raise VideoGenerateError("invalid_params", "request must be an object")
    if not isinstance(payload.get("staging_upload"), dict):
        raise VideoGenerateError("invalid_params", "staging_upload must be an object")


def _decode(value: Any) -> Any:
    return value.decode() if isinstance(value, bytes) else value


async def _alive_workers(redis: Any, task_type: str) -> list[str]:
    now = time.time()
    members = await redis.zrangebyscore(alive_zset_key(task_type), now - WORKER_HEARTBEAT_WINDOW_SECONDS, "+inf")
    return [_decode(member) for member in (members or [])]


async def _check_capacity(redis: Any, task_type: str, alive: list[str]) -> None:
    raw_capacity = await redis.hgetall(capacity_key(task_type)) or {}
    capacity_by_worker: dict[str, int] = {}
    for raw_name, raw_slots in raw_capacity.items():
        name = _decode(raw_name)
        try:
            capacity_by_worker[name] = int(_decode(raw_slots))
        except (TypeError, ValueError):
            continue
    total_free = sum(capacity_by_worker.get(name, 0) for name in alive)
    if total_free <= 0:
        raise VideoGenerateError("no_capacity_available", "no free capacity among live workers")


async def enqueue_video_generate(
    payload: dict, *, redis_factory: Callable[[], Any], settings: VideoGenerateSettings
) -> str:
    """Validate, admission-check, and enqueue a video-generate task.

    ``redis_factory`` is a zero-arg callable (not an already-resolved
    client) and is deliberately invoked only *after* the redis-free checks
    below pass. Passing an already-resolved client would defeat the point:
    a caller like ``endpoints.py`` builds that client via an expression such
    as ``redis=_get_redis_client()``, and Python evaluates keyword-argument
    expressions before this function's body ever runs -- so if
    ``_get_redis_client()`` itself raises on misconfiguration (F5), it would
    fire *before* shape/URL validation regardless of how validation is
    ordered in here, breaking every redis-free-rejection test (e.g.
    ``test_extra_field_is_422_invalid_params``, which has no ``fake_redis``
    fixture on purpose). Deferring construction to a factory called from
    inside this function, after validation, keeps rejections genuinely
    redis-free.

    Returns the (caller-supplied) ``task_id`` regardless of whether this
    call actually enqueued a new stream entry or hit the NX dedupe path
    (spec §1.1: a duplicate submission still returns 202 with the same
    task_id, since that task is already progressing through the pipeline).
    """
    # Cheap, local, redis-free checks first -- shape before URLs. A non-dict
    # (or otherwise shape-invalid) payload must be a deterministic 422
    # invalid_params and must never reach _validate_urls -> _iter_reference_urls,
    # which immediately does payload.get("request") and raises an uncaught
    # AttributeError for a list/str/int/bool body (F1). That escapes as an
    # unhandled exception instead of a clean 4xx/5xx, and design §9.4 treats
    # any non-4xx outcome as a TransientTransferError worth retrying -- i.e. a
    # deterministically-bad input would be retried forever. Shape-first still
    # keeps purely-input-driven rejections redis-free, same as before (see
    # test_extra_field_is_422_invalid_params, which has no fake_redis fixture).
    _validate_shape(payload)
    _validate_urls(payload, settings)

    task_id = payload["task_id"]

    redis = redis_factory()

    alive = await _alive_workers(redis, TASK_TYPE_VIDEO_GENERATE)
    if not alive:
        raise VideoGenerateError("no_worker_available", "no live video_generate worker")
    await _check_capacity(redis, TASK_TYPE_VIDEO_GENERATE, alive)

    envelope = {
        "type": TASK_TYPE_VIDEO_GENERATE,
        "task_id": task_id,
        "deadline_ts": payload["deadline_ts"],
        "model": payload["model"],
        "request": payload["request"],
        "staging_upload": payload["staging_upload"],
    }

    # Write order is spec-mandated (§2.3.1) and safety-critical: SET status
    # NX must happen strictly before XADD. If reversed, a claim that races
    # in faster than the status write can land finds no status/task_id key
    # (routes/worker_runner.py:254-263), silently xacks, and the task
    # vanishes with the generation stuck RUNNING forever. NX is also the
    # *only* de-dup point in the whole pipeline -- the CAS layer explicitly
    # permits re-claiming from "claimed" (cas.py:83), so it never de-dupes.
    ok = await redis.set(status_key(task_id), STATUS_QUEUED, ex=STATUS_TTL_SECONDS, nx=True)
    if not ok:
        return task_id  # duplicate submission; task is already on the rails, see spec §1.1
    try:
        await redis.xadd(stream_key(TASK_TYPE_VIDEO_GENERATE), {"payload": json.dumps(envelope)})
    except Exception as exc:
        # The status key landed but the stream entry didn't (F2): a bare
        # 500 here leaves a status:queued key with no matching task_id, so a
        # platform retry's SET NX finds the key already present, returns
        # False, and short-circuits straight to a 202 that was never
        # actually enqueued -- the task then sits "queued" until the 24h
        # status TTL expires and it turns into a phantom 404 unknown_task.
        # Best-effort delete the status key so a retry's SET NX can succeed
        # for real; a failure to delete must not mask the original XADD
        # failure (mirrors validated_transfer.py:477-484's "ambiguous"
        # rollback pattern).
        try:
            await redis.delete(status_key(task_id))
        except Exception:
            pass
        raise VideoGenerateError(
            "enqueue_ambiguous", f"failed to enqueue task after status write: {exc}"
        ) from exc
    return task_id


def _valid_result(result: Any) -> bool:
    return isinstance(result, dict) and all(field in result for field in _REQUIRED_RESULT_FIELDS)


def _split_error(raw_error: Any) -> tuple[str, str]:
    """Split the worker's flat ``"<code>: <message>"`` error string (the
    shape ``ReportRequest.error`` is locked to, spec §2.2) into
    ``(code, message)``. Falls back to ``("unknown", raw_error)`` when no
    ``": "`` separator is present."""
    text = raw_error if isinstance(raw_error, str) else ""
    code, sep, message = text.partition(": ")
    if not sep:
        return "unknown", text
    return code, message


async def fetch_video_generate_status(task_id: str, *, redis: Any) -> dict:
    """Read back a video-generate task's current state and translate it into
    the platform-facing shape (spec §1.2's status-mapping table).

    Returns a plain (never-None) dict. An unknown task_id is represented by
    a same-shaped dict simply lacking the "status" key -- the endpoint layer
    decides HTTP 404 vs 200 by checking ``"status" in body``.
    """
    raw_status = _decode(await redis.get(status_key(task_id)))
    if raw_status is None:
        return {"ok": False, "task_id": task_id, "error": {"code": "unknown_task", "message": "unknown task_id"}}

    if raw_status == STATUS_CANCELLED:
        # Cancelled never reads result_key, even if one happens to be
        # present -- spec §1.2 is explicit that this status is terminal on
        # its own.
        return {
            "ok": False,
            "task_id": task_id,
            "status": "failed",
            "error": {"code": "cancelled", "message": "task was cancelled"},
        }

    if raw_status in (STATUS_QUEUED, STATUS_CLAIMED):
        return {"ok": True, "task_id": task_id, "status": _STATUS_TO_PUBLIC[raw_status]}

    if raw_status not in (STATUS_DONE, STATUS_FAILED):
        return {
            "ok": False,
            "task_id": task_id,
            "status": "failed",
            "error": {"code": "unexpected_status", "message": f"unexpected status: {raw_status}"},
        }

    # done/failed: result_key is a Redis LIST -- read via LINDEX 0, not GET.
    raw_result = _decode(await redis.lindex(result_key(task_id), 0))
    if raw_result is None:
        # status_key TTL is 24h but result_key TTL is only 1h -- this window
        # is real, not defensive padding (spec §1.2).
        return {
            "ok": False,
            "task_id": task_id,
            "status": "failed",
            "error": {"code": "result_expired", "message": "result is no longer available"},
        }

    try:
        parsed = json.loads(raw_result)
    except (TypeError, ValueError):
        return {
            "ok": False,
            "task_id": task_id,
            "status": "failed",
            "error": {"code": "malformed_result", "message": "result payload is not valid JSON"},
        }

    if raw_status == STATUS_DONE:
        if not isinstance(parsed, dict) or not parsed.get("ok") or not _valid_result(parsed.get("result")):
            return {
                "ok": False,
                "task_id": task_id,
                "status": "failed",
                "error": {"code": "malformed_result", "message": "worker result is missing required fields"},
            }
        return {"ok": True, "task_id": task_id, "status": "succeeded", "result": parsed["result"]}

    # STATUS_FAILED: translate the worker's flat error string into a
    # structured {code, message, kind} dict (spec §1.2's translation note).
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "task_id": task_id,
            "status": "failed",
            "error": {"code": "malformed_result", "message": "result payload is not an object"},
        }
    code, message = _split_error(parsed.get("error"))
    kind = parsed.get("error_kind") if isinstance(parsed.get("error_kind"), str) else "unknown"
    return {
        "ok": False,
        "task_id": task_id,
        "status": "failed",
        "error": {"code": code, "message": message, "kind": kind},
    }
