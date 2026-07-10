import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Awaitable, Callable, List, Optional, Protocol, Tuple, TypedDict

import httpx
import redis.exceptions as redis_exceptions

from .common import BRIDGE_PART_SIZE, LibTVError

# Generic worker-task protocol namespace (workers are general task executors;
# media transfer is one task type). Keep all key names here for cross-repo reference.
TASK_TYPE = "media_transfer"
TASKS_STREAM = f"worker:tasks:{TASK_TYPE}"
WORKERS_ZSET = f"worker:alive:{TASK_TYPE}"
STATUS_KEY_PREFIX = "worker:task:status:"
RESULT_KEY_PREFIX = "worker:task:result:"
STATUS_TTL_SECONDS = 24 * 3600
RESULT_TTL_SECONDS = 3600
DEFAULT_MIN_BYTES = 2 * 1024 * 1024
DEFAULT_WAIT_TIMEOUT = 60.0
MAX_WAIT_TIMEOUT = 300.0
# Assumed direct-upload rate on the constrained egress; the delegated wait is half
# the projected direct duration so falling back still beats never delegating.
ASSUMED_DIRECT_RATE_BYTES_PER_S = 0.5 * 1024 * 1024
DEFAULT_LATE_GRACE_TIMEOUT = 2.0
WARN_INTERVAL_SECONDS = 300.0
# A worker's heartbeat key/zset entry is refreshed every ~15s; anything older than
# 2x that interval is treated as dead so a stalled worker doesn't win task claims.
WORKER_HEARTBEAT_WINDOW_SECONDS = 30.0


logger = logging.getLogger(__name__)

_redis_clients: dict = {}  # (event loop, url) -> redis.asyncio client
_last_unavailable_warn_ts = 0.0


def _warn_unavailable(message: str, exc_info: bool = False) -> None:
    global _last_unavailable_warn_ts
    now = time.time()
    if now - _last_unavailable_warn_ts >= WARN_INTERVAL_SECONDS:
        _last_unavailable_warn_ts = now
        logger.warning(message, exc_info=exc_info)


def get_transfer_redis() -> Optional[Any]:
    """Lazy redis.asyncio client built from MEDIA_TRANSFER_REDIS_URL; None when the
    env var is unset. Cached per (event loop, url): a redis.asyncio client binds to
    the loop it was created under, so sharing one across loops raises RuntimeError
    and leaks connections. Entries for closed loops (and superseded urls on the
    current loop) are evicted, closing their clients best-effort."""
    url = os.getenv("MEDIA_TRANSFER_REDIS_URL")
    if not url:
        return None
    try:
        loop: Optional[Any] = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    key = (loop, url)
    client = _redis_clients.get(key)
    if client is None:
        import redis.asyncio as async_redis

        for stale_key in [k for k in _redis_clients if (k[0] is loop and k[1] != url) or (k[0] is not None and k[0].is_closed())]:
            stale = _redis_clients.pop(stale_key)
            if loop is not None and stale_key[0] is loop and not loop.is_closed():
                loop.create_task(stale.aclose())
        client = async_redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5,
            # No read timeout: BRPOP legitimately blocks up to wait_timeout (minutes);
            # a socket_timeout shorter than that would abort every delegated wait.
            health_check_interval=30,
        )
        _redis_clients[key] = client
    return client


def resolve_wait_timeout(size: int) -> float:
    env_value = os.getenv("MEDIA_TRANSFER_WAIT_TIMEOUT")
    if env_value:
        return float(env_value)
    return min(MAX_WAIT_TIMEOUT, max(DEFAULT_WAIT_TIMEOUT, size / ASSUMED_DIRECT_RATE_BYTES_PER_S * 0.5))


def status_key(task_id: str) -> str:
    return f"{STATUS_KEY_PREFIX}{task_id}"


def result_key(task_id: str) -> str:
    return f"{RESULT_KEY_PREFIX}{task_id}"


class PartTarget(TypedDict):
    n: int
    url: str


class PartEtag(TypedDict):
    n: int
    etag: str


class MediaTransferStrategy(Protocol):
    async def transfer(self, source_url: str, size: int, parts: List[PartTarget]) -> List[PartEtag]: ...


FetchFn = Callable[[str], Awaitable[bytes]]
PutFn = Callable[[str, bytes], Awaitable[Tuple[int, str]]]


async def _default_fetch(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, follow_redirects=True)
    if resp.status_code != 200:
        raise LibTVError(status_code=resp.status_code, message=f"libtv reference fetch HTTP {resp.status_code}")
    return resp.content


async def _default_put(url: str, data: bytes) -> Tuple[int, str]:
    async with httpx.AsyncClient() as client:
        resp = await client.put(url, content=data)
    return resp.status_code, resp.headers.get("etag", "")


class DirectTransfer:
    """In-process transfer: stream the source into memory, then PUT each
    presigned part. Equivalent to litellm's historical upload behavior."""

    def __init__(
        self,
        fetch: Optional[FetchFn] = None,
        put: Optional[PutFn] = None,
        part_size: int = BRIDGE_PART_SIZE,
    ):
        self._fetch = fetch or _default_fetch
        self._put = put or _default_put
        self.part_size = part_size

    async def transfer(self, source_url: str, size: int, parts: List[PartTarget]) -> List[PartEtag]:
        buffer = await self._fetch(source_url)
        etags: List[PartEtag] = []
        for i, part in enumerate(parts):
            chunk = buffer[i * self.part_size : (i + 1) * self.part_size]
            status, etag = await self._put(part["url"], chunk)
            if status not in (200, 204):
                raise LibTVError(status_code=status, message=f"libtv upload part {part.get('n')} failed")
            etags.append({"n": part.get("n", i + 1), "etag": etag})
        return etags


async def cas_status(redis: Any, task_id: str, allowed_current: set, new_value: str, ex: Optional[int] = None) -> bool:
    """Compare-and-set the status key for ``task_id``: only writes ``new_value``
    when the current value is one of ``allowed_current`` (``None`` may be a
    member to allow transitioning from a missing key). A stored value is matched
    by its base token before any ``:`` suffix, so an owner-tagged claim such as
    ``claimed:{worker_id}`` still counts as ``claimed``. Uses a WATCH/MULTI
    transaction rather than server-side Lua so it works against any Redis-
    compatible server (including fakeredis without the optional lupa/Lua
    scripting extra)."""
    key = status_key(task_id)
    async with redis.pipeline(transaction=True) as pipe:
        try:
            await pipe.watch(key)
            current = await pipe.get(key)  # immediate mode while watching
            base = current.split(":", 1)[0] if isinstance(current, str) else current
            if base not in allowed_current:
                await pipe.unwatch()
                return False
            pipe.multi()
            if ex is not None:
                pipe.set(key, new_value, ex=ex)
            else:
                pipe.set(key, new_value)
            await pipe.execute()
            return True
        except redis_exceptions.WatchError:
            return False


class DelegatedTransfer:
    """Hands the byte transfer off to a Redis-coordinated worker fleet, falling
    back to ``fallback`` (normally a DirectTransfer) whenever no worker is
    available, the worker reports failure, or it doesn't finish before
    ``wait_timeout``. All status transitions are compare-and-set so a requester
    fallback and a late worker completion can never both take effect."""

    def __init__(
        self,
        redis: Any,
        fallback: MediaTransferStrategy,
        wait_timeout: float = DEFAULT_WAIT_TIMEOUT,
        late_grace_timeout: float = DEFAULT_LATE_GRACE_TIMEOUT,
        heartbeat_window: float = WORKER_HEARTBEAT_WINDOW_SECONDS,
    ):
        self.redis = redis
        self.fallback = fallback
        self.wait_timeout = wait_timeout
        self.late_grace_timeout = late_grace_timeout
        self.heartbeat_window = heartbeat_window

    async def _has_active_worker(self) -> bool:
        now = time.time()
        count = await self.redis.zcount(WORKERS_ZSET, now - self.heartbeat_window, "+inf")
        return count > 0

    async def _cas_cancel(self, task_id: str) -> bool:
        return await cas_status(self.redis, task_id, {"queued", "claimed"}, "cancelled", ex=STATUS_TTL_SECONDS)

    def _parse_result(self, raw: Any) -> Optional[dict]:
        if raw is None:
            return None
        _, body = raw
        return json.loads(body)

    async def transfer(self, source_url: str, size: int, parts: List[PartTarget]) -> List[PartEtag]:
        task_id = str(uuid.uuid4())
        try:
            return await self._transfer_delegated(task_id, source_url, size, parts)
        except (redis_exceptions.RedisError, RuntimeError, OSError):
            # RuntimeError covers a redis.asyncio client used across event loops,
            # which surfaces outside the RedisError hierarchy.
            _warn_unavailable("media transfer redis unreachable; falling back to direct transfer", exc_info=True)
            try:
                # Best-effort cancel so a worker that did receive the task drops it.
                # If the cancel itself fails, a worker may still upload the parts in
                # parallel with the direct fallback; multipart PUTs to the same
                # uploadId/partNumber just overwrite each other, so the double
                # upload is harmless.
                await self._cas_cancel(task_id)
            except Exception:
                pass
            return await self.fallback.transfer(source_url, size, parts)

    async def _transfer_delegated(self, task_id: str, source_url: str, size: int, parts: List[PartTarget]) -> List[PartEtag]:
        if not await self._has_active_worker():
            return await self.fallback.transfer(source_url, size, parts)

        now = time.time()
        payload = {
            "type": TASK_TYPE,
            "task_id": task_id,
            "source": {"url": source_url, "size": size},
            "target": {"kind": "presigned_parts", "parts": parts, "part_size": BRIDGE_PART_SIZE},
            "deadline_ts": now + self.wait_timeout,
            "created_ts": now,
        }
        await self.redis.set(status_key(task_id), "queued", ex=STATUS_TTL_SECONDS)
        await self.redis.xadd(TASKS_STREAM, {"payload": json.dumps(payload)})

        result = self._parse_result(await self.redis.brpop(result_key(task_id), timeout=self.wait_timeout))
        if result is None:
            if not await self._cas_cancel(task_id):
                # A worker raced past the deadline and already CAS'd to done; give
                # it a short grace period to actually push the result it committed to.
                result = self._parse_result(
                    await self.redis.brpop(result_key(task_id), timeout=self.late_grace_timeout)
                )

        if result is not None and result.get("ok"):
            inner = result.get("result") or {}
            return [{"n": e["n"], "etag": e.get("etag", "")} for e in inner.get("etags", [])]
        return await self.fallback.transfer(source_url, size, parts)


def build_transfer_strategy(
    *,
    size: int,
    redis_client: Optional[Any] = None,
    fetch: Optional[FetchFn] = None,
    put: Optional[PutFn] = None,
    fallback: Optional[MediaTransferStrategy] = None,
) -> MediaTransferStrategy:
    direct = fallback or DirectTransfer(fetch=fetch, put=put)
    mode = os.getenv("MEDIA_TRANSFER_MODE", "direct").lower()
    if mode != "delegated":
        return direct
    if redis_client is None:
        redis_client = get_transfer_redis()
    if redis_client is None:
        _warn_unavailable(
            "MEDIA_TRANSFER_MODE=delegated but no redis client (set MEDIA_TRANSFER_REDIS_URL); "
            "using direct transfer"
        )
        return direct
    min_bytes = int(os.getenv("MEDIA_TRANSFER_MIN_BYTES", str(DEFAULT_MIN_BYTES)))
    if size < min_bytes:
        return direct
    return DelegatedTransfer(redis=redis_client, fallback=direct, wait_timeout=resolve_wait_timeout(size))
