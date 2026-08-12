"""Strict, local execution for versioned validated media-transfer tasks.

This module deliberately does not reuse the legacy ``media_transfer`` code:
validated tasks must retain their byte limits, URL boundaries and decoded-image
checks even when the optional portable worker is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from .common import LibTVError
from .transfer import (
    STATUS_TTL_SECONDS,
    VALIDATED_WORKERS_ZSET,
    WORKER_HEARTBEAT_WINDOW_SECONDS,
    result_key,
    status_key,
)

VALIDATION_VERSION = "image-v1"
VALIDATED_TASK_TYPE = "validated_media_transfer"
_MODES = frozenset({"probe", "source_transfer", "transfer"})
_MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PartTarget(TypedDict):
    n: int
    url: str


class PartEtag(TypedDict):
    n: int
    etag: str


class ValidatedTransferError(LibTVError):
    """A classified validated-transfer failure.

    Validation errors are deterministic (HTTP 422 at the proxy edge); resource,
    ownership and transport errors are retryable (HTTP 503).
    """

    def __init__(self, message: str, *, validation: bool = True):
        super().__init__(status_code=422 if validation else 503, message=message)
        self.validation = validation


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidatedTransferError(f"{name} must be a positive integer")
    return value


def _parse_hosts(value: str, name: str) -> frozenset[str]:
    hosts = frozenset(host.strip().lower() for host in value.split(",") if host.strip())
    if not hosts or any("*" in host or "/" in host or "@" in host for host in hosts):
        raise ValidatedTransferError(f"{name} must contain exact hosts", validation=False)
    return hosts


def _env_positive(prefix: str, name: str, default: int) -> int:
    raw = os.getenv(prefix + name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidatedTransferError(f"{prefix}{name} must be a positive integer", validation=False) from exc
    try:
        return _positive_int(value, prefix + name)
    except ValidatedTransferError as exc:
        raise ValidatedTransferError(exc.message, validation=False) from exc


@dataclass(frozen=True)
class ValidatedTransferSettings:
    source_hosts: frozenset[str]
    target_hosts: frozenset[str]
    spool_dir: Path
    hard_cap: int
    quota_bytes: int
    max_dimensions: int
    max_pixels: int
    max_parts: int
    chunk_size: int
    put_retries: int
    local_concurrency: int = 2
    local_queue_limit: int = 4
    allow_http: bool = False
    orphan_max_age_seconds: int = 3600

    @classmethod
    def from_environment(cls) -> "ValidatedTransferSettings":
        prefix = "LIBTV_VALIDATED_TRANSFER_"
        # The old source-only setting remains a migration bridge for existing
        # Topaz callers; target hosts always require the new independent list.
        source_hosts = _parse_hosts(
            os.getenv(prefix + "SOURCE_HOSTS", "") or os.getenv("LIBTV_PLATFORM_SOURCE_HOSTS", ""),
            prefix + "SOURCE_HOSTS",
        )
        target_hosts = _parse_hosts(os.getenv(prefix + "TARGET_HOSTS", ""), prefix + "TARGET_HOSTS")
        allow_http = os.getenv(prefix + "ALLOW_HTTP", "false").strip().lower() == "true"
        if os.getenv("LITELLM_ENVIRONMENT", "production").lower() == "production" and allow_http:
            raise ValidatedTransferError("validated transfer HTTP is forbidden in production", validation=False)
        settings = cls(
            source_hosts=source_hosts,
            target_hosts=target_hosts,
            spool_dir=Path(os.getenv(prefix + "SPOOL_DIR", "/tmp/litellm-validated-transfer")),
            hard_cap=_env_positive(prefix, "HARD_CAP_BYTES", 64 * 1024 * 1024),
            quota_bytes=_env_positive(prefix, "QUOTA_BYTES", 64 * 1024 * 1024),
            max_dimensions=_env_positive(prefix, "MAX_DIMENSION", 16_384),
            max_pixels=_env_positive(prefix, "MAX_PIXELS", 100_000_000),
            max_parts=_env_positive(prefix, "MAX_PARTS", 128),
            chunk_size=_env_positive(prefix, "CHUNK_SIZE_BYTES", 1024 * 1024),
            put_retries=_env_positive(prefix, "PUT_RETRIES", 3),
            local_concurrency=_env_positive(prefix, "LOCAL_CONCURRENCY", 2),
            local_queue_limit=_env_positive(prefix, "LOCAL_QUEUE_LIMIT", 4),
            allow_http=allow_http,
            orphan_max_age_seconds=_env_positive(prefix, "SPOOL_ORPHAN_MAX_AGE_SECONDS", 3600),
        )
        if settings.quota_bytes > settings.hard_cap:
            raise ValidatedTransferError("quota may not exceed hard cap", validation=False)
        return settings

    def with_limits(self, **kwargs: Any) -> "ValidatedTransferSettings":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ValidatedTransferRequest:
    task_id: str | None
    mode: Literal["probe", "source_transfer", "transfer"]
    source_url: str
    source_bytes: int | None
    source_sha256: str | None
    target_parts: tuple[PartTarget, ...]
    target_part_size: int | None
    expected: Mapping[str, Any]
    hard_cap: int | None
    quota_bytes: int | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ValidatedTransferRequest":
        allowed = {"type", "task_id", "mode", "source", "target", "expected", "hard_cap", "quota_bytes", "deadline_ts", "created_ts"}
        if set(payload) - allowed or payload.get("type") != VALIDATED_TASK_TYPE:
            raise ValidatedTransferError("invalid validated transfer request")
        task_id = payload.get("task_id")
        mode = payload.get("mode")
        source = payload.get("source")
        if task_id is not None and (not isinstance(task_id, str) or not task_id):
            raise ValidatedTransferError("invalid validated transfer task ID")
        if mode not in _MODES or not isinstance(source, Mapping):
            raise ValidatedTransferError("invalid validated transfer request")
        source_url = source.get("url")
        source_bytes = source.get("bytes", source.get("size"))
        source_sha256 = source.get("sha256")
        if not isinstance(source_url, str) or not source_url:
            raise ValidatedTransferError("source URL is required")
        if source_bytes is not None:
            _positive_int(source_bytes, "source bytes")
        if source_sha256 is not None and (not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256.lower())):
            raise ValidatedTransferError("source SHA-256 must be lowercase hex")
        target = payload.get("target")
        parts: tuple[PartTarget, ...] = ()
        part_size: int | None = None
        if mode != "probe":
            if not isinstance(target, Mapping) or target.get("kind") != "presigned_parts" or not isinstance(target.get("parts"), list):
                raise ValidatedTransferError("transfer target parts are required")
            raw_parts = target["parts"]
            part_size = _positive_int(target.get("part_size"), "target part size")
            parts = tuple({"n": part.get("n"), "url": part.get("url")} for part in raw_parts if isinstance(part, Mapping))
            if len(parts) != len(raw_parts):
                raise ValidatedTransferError("invalid target part")
        expected = payload.get("expected", {})
        if not isinstance(expected, Mapping):
            raise ValidatedTransferError("expected metadata must be an object")
        hard_cap = payload.get("hard_cap")
        if hard_cap is not None:
            _positive_int(hard_cap, "hard cap")
        quota_bytes = payload.get("quota_bytes")
        if quota_bytes is not None:
            _positive_int(quota_bytes, "quota bytes")
        if mode == "transfer" and quota_bytes is None:
            raise ValidatedTransferError("quota bytes are required for transfer")
        return cls(task_id, mode, source_url, source_bytes, source_sha256.lower() if source_sha256 else None, parts, part_size, expected, hard_cap, quota_bytes)


OwnershipCheck = Callable[[], Awaitable[bool]]


class StrictValidatedTransferExecutor:
    """Streams one source into a bounded disk spool and validates decoded pixels."""

    def __init__(self, settings: ValidatedTransferSettings, *, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client
        self._slots = asyncio.BoundedSemaphore(settings.local_concurrency)
        self._waiting = 0

    def ready(self) -> bool:
        try:
            self.settings.spool_dir.mkdir(parents=True, exist_ok=True)
            return self.settings.spool_dir.is_dir() and os.access(self.settings.spool_dir, os.W_OK)
        except OSError:
            return False

    def cleanup_orphans(self) -> None:
        if not self.ready():
            return
        cutoff = time.time() - self.settings.orphan_max_age_seconds
        for spool in self.settings.spool_dir.glob("validated-transfer-*"):
            try:
                if spool.is_file() and spool.stat().st_mtime < cutoff:
                    spool.unlink()
            except OSError:
                continue

    def validate_request(self, request: ValidatedTransferRequest) -> None:
        self._validate_url(request.source_url, self.settings.source_hosts, "source")
        effective_cap = min(
            self.settings.hard_cap,
            self.settings.quota_bytes,
            request.hard_cap or self.settings.hard_cap,
            request.quota_bytes or self.settings.quota_bytes,
        )
        if request.source_bytes is not None and request.source_bytes > effective_cap:
            raise ValidatedTransferError("source exceeds byte limit")
        if request.mode != "probe":
            if not request.target_parts or len(request.target_parts) > self.settings.max_parts:
                raise ValidatedTransferError("invalid target part count")
            numbers = [part["n"] for part in request.target_parts]
            if numbers != list(range(1, len(request.target_parts) + 1)):
                raise ValidatedTransferError("target parts must be consecutive")
            for part in request.target_parts:
                if not isinstance(part["url"], str) or not isinstance(part["n"], int):
                    raise ValidatedTransferError("invalid target part")
                self._validate_url(part["url"], self.settings.target_hosts, "target")

    def _validate_url(self, raw_url: str, allowed_hosts: frozenset[str], role: str) -> None:
        try:
            parsed = urlsplit(raw_url)
            port = parsed.port
        except ValueError as exc:
            raise ValidatedTransferError(f"invalid {role} URL") from exc
        schemes = {"https"} | ({"http"} if self.settings.allow_http else set())
        if (
            parsed.scheme not in schemes
            or not parsed.hostname
            or parsed.hostname.lower() not in allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or (port is not None and port not in ({443} if parsed.scheme == "https" else {80}))
        ):
            raise ValidatedTransferError(f"invalid {role} URL")

    async def execute(self, request: ValidatedTransferRequest, *, ownership_check: OwnershipCheck | None = None) -> dict[str, Any]:
        if self._slots.locked() and self._waiting >= self.settings.local_queue_limit:
            raise ValidatedTransferError("validated transfer local queue is saturated", validation=False)
        self._waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self._waiting -= 1
        try:
            return await self._execute_reserved(request, ownership_check=ownership_check)
        finally:
            self._slots.release()

    async def _execute_reserved(self, request: ValidatedTransferRequest, *, ownership_check: OwnershipCheck | None = None) -> dict[str, Any]:
        self.validate_request(request)
        if not self.ready():
            raise ValidatedTransferError("validated transfer spool is unavailable", validation=False)
        self.cleanup_orphans()
        client = self.client or httpx.AsyncClient(follow_redirects=False, timeout=httpx.Timeout(120.0, connect=10.0))
        own_client = self.client is None
        spool: Path | None = None
        try:
            spool, size, digest = await self._download(client, request)
            metadata = self._validate_image(spool, size, digest, request)
            if request.mode != "probe":
                metadata["etags"] = await self._upload_parts(client, spool, size, request, ownership_check)
            return metadata
        finally:
            if spool is not None:
                spool.unlink(missing_ok=True)
            if own_client:
                await client.aclose()

    async def _download(self, client: httpx.AsyncClient, request: ValidatedTransferRequest) -> tuple[Path, int, str]:
        effective_cap = min(self.settings.hard_cap, self.settings.quota_bytes, request.hard_cap or self.settings.hard_cap, request.quota_bytes or self.settings.quota_bytes)
        self.settings.spool_dir.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix="validated-transfer-", dir=self.settings.spool_dir)
        spool = Path(filename)
        hasher = hashlib.sha256()
        total = 0
        declared_content_length: int | None = None
        try:
            try:
                response_context = client.stream("GET", request.source_url, follow_redirects=False)
                async with response_context as response:
                    if response.status_code != 200:
                        raise ValidatedTransferError(f"source HTTP {response.status_code}", validation=400 <= response.status_code < 500)
                    content_length = response.headers.get("content-length")
                    if content_length:
                        if not content_length.isdigit():
                            raise ValidatedTransferError("invalid source Content-Length")
                        declared_content_length = int(content_length)
                        if declared_content_length > effective_cap:
                            raise ValidatedTransferError("source exceeds byte limit")
                    with os.fdopen(descriptor, "wb") as output:
                        descriptor = -1
                        async for chunk in response.aiter_bytes(self.settings.chunk_size):
                            total += len(chunk)
                            if total > effective_cap:
                                raise ValidatedTransferError("source exceeds byte limit")
                            hasher.update(chunk)
                            output.write(chunk)
            except httpx.HTTPError as exc:
                raise ValidatedTransferError("source transport failed", validation=False) from exc
            if total == 0:
                raise ValidatedTransferError("source image is empty")
            if declared_content_length is not None and total != declared_content_length:
                raise ValidatedTransferError("source Content-Length mismatch")
            digest = hasher.hexdigest()
            if request.source_bytes is not None and total != request.source_bytes:
                raise ValidatedTransferError("source byte count mismatch")
            if request.source_sha256 is not None and digest != request.source_sha256:
                raise ValidatedTransferError("source SHA-256 mismatch")
            return spool, total, digest
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            spool.unlink(missing_ok=True)
            raise

    def _validate_image(self, spool: Path, size: int, digest: str, request: ValidatedTransferRequest) -> dict[str, Any]:
        original_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = self.settings.max_pixels
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(spool) as image:
                    image.verify()
                with Image.open(spool) as image:
                    image.load()
                    mime = _MIME_BY_FORMAT.get(image.format or "")
                    width, height = image.size
            if mime is None:
                raise ValidatedTransferError("image MIME is not accepted")
            if width <= 0 or height <= 0 or width > self.settings.max_dimensions or height > self.settings.max_dimensions:
                raise ValidatedTransferError("image dimensions exceed limit")
            if width * height > self.settings.max_pixels:
                raise ValidatedTransferError("image pixels exceed limit")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValidatedTransferError("invalid or unsafe image") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = original_limit
        expected = request.expected
        for key, value in {"bytes": size, "sha256": digest, "mime": mime, "width": width, "height": height}.items():
            if key in expected and expected[key] != value:
                raise ValidatedTransferError(f"expected {key} mismatch")
        return {"validation_version": VALIDATION_VERSION, "bytes": size, "mime": mime, "width": width, "height": height, "sha256": digest}

    async def _upload_parts(
        self,
        client: httpx.AsyncClient,
        spool: Path,
        size: int,
        request: ValidatedTransferRequest,
        ownership_check: OwnershipCheck | None,
    ) -> list[PartEtag]:
        assert request.target_part_size is not None
        expected_count = max(1, (size + request.target_part_size - 1) // request.target_part_size)
        if len(request.target_parts) != expected_count:
            raise ValidatedTransferError("target part count does not match source")
        etags: list[PartEtag] = []
        for index, part in enumerate(request.target_parts):
            await self._ensure_owned(ownership_check)
            start = index * request.target_part_size
            length = min(request.target_part_size, size - start)
            response: httpx.Response | None = None
            for attempt in range(self.settings.put_retries):
                async with self._file_stream(spool, start, length) as content:
                    try:
                        response = await client.put(part["url"], content=content, follow_redirects=False)
                    except httpx.HTTPError:
                        if attempt + 1 == self.settings.put_retries:
                            raise ValidatedTransferError("multipart PUT failed", validation=False)
                        continue
                if response.status_code in (200, 201, 204):
                    break
                if response.status_code < 500 or attempt + 1 == self.settings.put_retries:
                    raise ValidatedTransferError("multipart PUT failed", validation=False)
            assert response is not None
            etag = response.headers.get("etag", "")
            if not etag:
                raise ValidatedTransferError("multipart PUT returned no ETag", validation=False)
            await self._ensure_owned(ownership_check)
            etags.append({"n": part["n"], "etag": etag})
        return etags

    async def _ensure_owned(self, ownership_check: OwnershipCheck | None) -> None:
        if ownership_check is not None and not await ownership_check():
            raise ValidatedTransferError("validated transfer ownership was lost", validation=False)

    def _file_stream(self, spool: Path, start: int, length: int) -> "_SpoolSlice":
        """Return an async context manager yielding a bounded async byte iterator."""
        return _SpoolSlice(spool, start, length, self.settings.chunk_size)


class _SpoolSlice:
    def __init__(self, spool: Path, start: int, length: int, chunk_size: int):
        self.spool, self.start, self.remaining, self.chunk_size = spool, start, length, chunk_size
        self.file: Any = None

    async def __aenter__(self) -> "_SpoolSlice":
        self.file = self.spool.open("rb")
        self.file.seek(self.start)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.file is not None:
            self.file.close()

    def __aiter__(self) -> "_SpoolSlice":
        return self

    async def __anext__(self) -> bytes:
        if self.remaining <= 0:
            raise StopAsyncIteration
        chunk = self.file.read(min(self.chunk_size, self.remaining))
        if not chunk:
            raise ValidatedTransferError("spool was truncated", validation=False)
        self.remaining -= len(chunk)
        return chunk


class ValidatedTransferRouter:
    """Prefer a fresh portable worker and safely fall back to local execution."""

    def __init__(
        self,
        executor: StrictValidatedTransferExecutor,
        *,
        redis: Any | None,
        instance_id: str | None = None,
        wait_timeout: float = 60.0,
        heartbeat_window_seconds: float = WORKER_HEARTBEAT_WINDOW_SECONDS,
    ):
        self.executor = executor
        self.redis = redis
        self.instance_id = instance_id or os.getenv("LIBTV_VALIDATED_TRANSFER_INSTANCE_ID", "litellm")
        self.wait_timeout = wait_timeout
        self.heartbeat_window_seconds = heartbeat_window_seconds

    async def execute(self, request: ValidatedTransferRequest) -> dict[str, Any]:
        if self.redis is None or not await self._has_fresh_worker():
            return await self._execute_local(request, "local_no_worker")
        task_id = request.task_id or str(uuid.uuid4())
        payload = self._payload(request, task_id)
        try:
            await self.redis.set(status_key(task_id), "queued", ex=STATUS_TTL_SECONDS)
        except Exception:
            return await self._execute_local(request, "local_pre_enqueue_failure")
        try:
            await self.redis.xadd("worker:tasks:validated_media_transfer", {"payload": json.dumps(payload)})
        except Exception as exc:
            # The stream append may have reached Redis despite its client error.
            raise ValidatedTransferError("validated transfer enqueue is ambiguous", validation=False) from exc
        try:
            raw = await self.redis.brpop(result_key(task_id), timeout=self.wait_timeout)
        except Exception as exc:
            if await self._claim_fallback(task_id):
                return await self._execute_local(request, "local_worker_failure", ownership_check=self._ownership_check(task_id))
            raise ValidatedTransferError("validated transfer result wait is ambiguous", validation=False) from exc
        if raw is not None:
            result = self._parse_remote_result(raw)
            if result is not None:
                result["route"] = "remote"
                return result
        if await self._claim_fallback(task_id):
            return await self._execute_local(request, "local_worker_failure", ownership_check=self._ownership_check(task_id))
        raise ValidatedTransferError("validated transfer fallback ownership conflict", validation=False)

    async def _has_fresh_worker(self) -> bool:
        assert self.redis is not None
        try:
            return await self.redis.zcount(
                VALIDATED_WORKERS_ZSET, time.time() - self.heartbeat_window_seconds, "+inf"
            ) > 0
        except Exception:
            return False

    async def _execute_local(
        self, request: ValidatedTransferRequest, route: str, *, ownership_check: OwnershipCheck | None = None
    ) -> dict[str, Any]:
        result = await self.executor.execute(request, ownership_check=ownership_check)
        result["route"] = route
        return result

    def _payload(self, request: ValidatedTransferRequest, task_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": VALIDATED_TASK_TYPE,
            "task_id": task_id,
            "mode": request.mode,
            "source": {"url": request.source_url},
            "expected": dict(request.expected),
            "hard_cap": request.hard_cap or self.executor.settings.hard_cap,
            "quota_bytes": request.quota_bytes,
            "created_ts": time.time(),
            "deadline_ts": time.time() + self.wait_timeout,
        }
        if request.source_bytes is not None:
            payload["source"]["bytes"] = request.source_bytes
        if request.source_sha256 is not None:
            payload["source"]["sha256"] = request.source_sha256
        if request.mode != "probe":
            payload["target"] = {
                "kind": "presigned_parts", "parts": list(request.target_parts), "part_size": request.target_part_size
            }
        else:
            payload["target"] = {}
        return payload

    def _parse_remote_result(self, raw: Any) -> dict[str, Any] | None:
        try:
            _, body = raw
            if isinstance(body, bytes):
                body = body.decode("utf-8")
            envelope = json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidatedTransferError("validated worker result is malformed", validation=False) from exc
        if not isinstance(envelope, dict):
            raise ValidatedTransferError("validated worker result is malformed", validation=False)
        if not envelope.get("ok"):
            if envelope.get("error_kind") == "validation":
                raise ValidatedTransferError(str(envelope.get("error") or "worker validation rejected"))
            return None
        result = envelope.get("result")
        required = {"validation_version", "bytes", "mime", "width", "height", "sha256"}
        if not isinstance(result, dict) or not required.issubset(result) or result.get("validation_version") != VALIDATION_VERSION:
            raise ValidatedTransferError("validated worker result has invalid metadata", validation=False)
        if (
            not isinstance(result["bytes"], int)
            or isinstance(result["bytes"], bool)
            or result["bytes"] <= 0
            or not isinstance(result["mime"], str)
            or result["mime"] not in _MIME_BY_FORMAT.values()
            or not isinstance(result["width"], int)
            or isinstance(result["width"], bool)
            or result["width"] <= 0
            or not isinstance(result["height"], int)
            or isinstance(result["height"], bool)
            or result["height"] <= 0
            or not isinstance(result["sha256"], str)
            or not _SHA256_RE.fullmatch(result["sha256"])
        ):
            raise ValidatedTransferError("validated worker result has invalid metadata", validation=False)
        return dict(result)

    async def _claim_fallback(self, task_id: str) -> bool:
        assert self.redis is not None
        owner = f"fallback:{self.instance_id}"
        status = status_key(task_id)
        owner_storage = f"worker:task:owner:{task_id}"
        lease = f"worker:task:lease:{task_id}"
        try:
            # WATCH/MULTI is atomic in Redis and works with fakeredis, unlike
            # Lua on constrained test deployments. All three fencing keys are
            # changed in the one transaction.
            async with self.redis.pipeline(transaction=True) as pipe:
                await pipe.watch(status, owner_storage, lease)
                current = await pipe.get(status)
                current = current.decode() if isinstance(current, bytes) else current
                if current not in {"queued", "claimed"}:
                    await pipe.unwatch()
                    return False
                pipe.multi()
                pipe.set(status, "fallback_claimed", ex=STATUS_TTL_SECONDS)
                pipe.set(owner_storage, owner, ex=STATUS_TTL_SECONDS)
                pipe.delete(lease)
                await pipe.execute()
                return True
        except Exception:
            return False

    def _ownership_check(self, task_id: str) -> OwnershipCheck:
        async def owned() -> bool:
            if self.redis is None:
                return False
            try:
                status, owner = await self.redis.mget(
                    status_key(task_id), f"worker:task:owner:{task_id}"
                )
                status = status.decode() if isinstance(status, bytes) else status
                owner = owner.decode() if isinstance(owner, bytes) else owner
                return status == "fallback_claimed" and owner == f"fallback:{self.instance_id}"
            except Exception:
                return False

        return owned
