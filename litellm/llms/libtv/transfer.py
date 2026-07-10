import asyncio
import json
import os
import time
import uuid
from typing import Any, Awaitable, Callable, List, Optional, Protocol, Tuple, TypedDict

import httpx
import redis.exceptions as redis_exceptions

from .common import BRIDGE_PART_SIZE, LibTVError

TASKS_STREAM = "media:transfer:tasks"
WORKERS_ZSET = "media:transfer:workers"
STATUS_TTL_SECONDS = 24 * 3600
RESULT_TTL_SECONDS = 3600
DEFAULT_MIN_BYTES = 2 * 1024 * 1024
DEFAULT_WAIT_TIMEOUT = 60.0
DEFAULT_LATE_GRACE_TIMEOUT = 2.0
# A worker's heartbeat key/zset entry is refreshed every ~15s; anything older than
# 2x that interval is treated as dead so a stalled worker doesn't win task claims.
WORKER_HEARTBEAT_WINDOW_SECONDS = 30.0


def status_key(task_id: str) -> str:
    return f"media:transfer:status:{task_id}"


def result_key(task_id: str) -> str:
    return f"media:transfer:result:{task_id}"


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
    member to allow transitioning from a missing key). Uses a WATCH/MULTI
    transaction rather than server-side Lua so it works against any Redis-
    compatible server (including fakeredis without the optional lupa/Lua
    scripting extra)."""
    key = status_key(task_id)
    async with redis.pipeline(transaction=True) as pipe:
        try:
            await pipe.watch(key)
            current = await redis.get(key)
            if current not in allowed_current:
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
        if not await self._has_active_worker():
            return await self.fallback.transfer(source_url, size, parts)

        task_id = str(uuid.uuid4())
        now = time.time()
        payload = {
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
            return [{"n": e["n"], "etag": e.get("etag", "")} for e in result.get("etags", [])]
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
    if mode != "delegated" or redis_client is None:
        return direct
    min_bytes = int(os.getenv("MEDIA_TRANSFER_MIN_BYTES", str(DEFAULT_MIN_BYTES)))
    if size < min_bytes:
        return direct
    wait_timeout = float(os.getenv("MEDIA_TRANSFER_WAIT_TIMEOUT", str(DEFAULT_WAIT_TIMEOUT)))
    return DelegatedTransfer(redis=redis_client, fallback=direct, wait_timeout=wait_timeout)
