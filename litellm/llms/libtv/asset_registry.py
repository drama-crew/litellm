"""Rate limiting and dedupe primitives for libtv third_asset registration.

Why this module exists (2026-09-01 causyn.cn incident): ten seedance-2.5 video
shots carrying 97 reference images -- only 30 of them distinct -- were submitted
inside 11.5 seconds. `LibTVClient.aresolve_compliant_image_refs` registers each
reference with its own `POST /api/third_asset/create`, with no dedupe, no reuse
across requests and no pacing, so ~97 registrations hit libtv inside 12s against
a documented cap of 15/minute. All ten submits came back

    code=10026 当前素材检测提交较频繁，请控制在每分钟 15 个以内

Nothing here decides WHAT to register; it only bounds how often, and lets the
caller collapse repeats. The cross-request cache itself lives in
`persistence.LibTVPersistence` next to the upload cache it mirrors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Coroutine, Deque, Hashable, Iterable, Optional, Sequence, TypeVar

from litellm.llms.libtv.common import LibTVError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Hashable)

WINDOW_SECONDS = 60.0

# Upstream states 15/minute. Sitting exactly on a vendor's stated cap loses to
# clock skew, their counting window's edges, and any other consumer of the same
# account, so the default keeps 20% headroom.
DEFAULT_CREATE_RPM = 12

# openhands `video_provider.DEFAULT_VIDEO_SUBMIT_READ_TIMEOUT_SECONDS` is 600s.
# Queueing longer than that converts a survivable wait into a client-side
# ReadTimeout, which the platform records as 生成超时 -- strictly worse than the
# throttle we were trying to absorb.
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 420.0

DEFAULT_ASSET_CACHE_TTL_SECONDS = 86400.0

# Atomic sliding window. Returns "0" when the caller may proceed, otherwise the
# number of seconds until the oldest entry leaves the window. Kept as a script
# (not a pipeline) so two uvicorn workers can never both read `len < limit`.
_ACQUIRE_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local used = redis.call('ZCARD', key)
if used < limit then
  redis.call('ZADD', key, now, member)
  redis.call('EXPIRE', key, math.ceil(window * 2))
  return '0'
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
if not oldest or #oldest < 2 then
  return '0'
end
return tostring(tonumber(oldest[2]) + window - now)
"""


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _non_negative_float(raw: str | None, default: float) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def create_rate_per_minute() -> int:
    return _positive_int(os.getenv("LIBTV_THIRD_ASSET_CREATE_RPM"), DEFAULT_CREATE_RPM)


def acquire_timeout_seconds() -> float:
    raw = os.getenv("LIBTV_THIRD_ASSET_ACQUIRE_TIMEOUT_SECONDS")
    value = _non_negative_float(raw, DEFAULT_ACQUIRE_TIMEOUT_SECONDS)
    return value if value > 0 else DEFAULT_ACQUIRE_TIMEOUT_SECONDS


def asset_cache_ttl_seconds() -> float:
    """TTL for a cached third_asset registration; 0 disables reuse entirely."""
    return _non_negative_float(os.getenv("LIBTV_ASSET_CACHE_TTL_SECONDS"), DEFAULT_ASSET_CACHE_TTL_SECONDS)


def dedupe_preserving_order(values: Sequence[T]) -> tuple[list[T], dict[T, int]]:
    """Distinct values in first-seen order, plus value -> index into that list.

    Callers register the unique list and then fan the results back out to their
    original positions, so a shot that names the same character sheet twice pays
    for one registration instead of two.
    """
    unique: list[T] = []
    index_for: dict[T, int] = {}
    for value in values:
        if value not in index_for:
            index_for[value] = len(unique)
            unique.append(value)
    return unique, index_for


class ThirdAssetRateLimiter:
    """Per-account sliding-window gate in front of `third_asset/create`.

    Redis-backed so it holds across uvicorn workers, with an in-process window
    as the fallback. The fallback is not a nicety: a limiter that raised when
    Redis blinked would take generation down for a reason unrelated to the
    user's request, which is a worse failure than the throttle it guards.
    """

    def __init__(
        self,
        redis_client: Any = None,
        *,
        rate_per_minute: int | None = None,
        acquire_timeout: float | None = None,
        window_seconds: float = WINDOW_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.0, 0.5),
    ) -> None:
        self._redis = redis_client
        self._rate = rate_per_minute if rate_per_minute is not None else create_rate_per_minute()
        self._timeout = acquire_timeout if acquire_timeout is not None else acquire_timeout_seconds()
        self._window = window_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._jitter = jitter
        self._local: dict[str, Deque[float]] = defaultdict(deque)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._redis_usable = redis_client is not None

    async def acquire(self, account_key: str) -> None:
        """Block until this account may issue one more create, or raise 429.

        429 rather than 502 so the caller can distinguish "we queued you out"
        from "this input can never work" -- the distinction the 2026-09-01
        incident lacked, where every failure collapsed into one generic message
        and the agent retried straight back into the wall.
        """
        deadline = self._monotonic() + self._timeout
        while True:
            wait_for = await self._try_acquire(account_key)
            if wait_for <= 0:
                return
            # Jitter keeps a burst of sibling shots from re-colliding in lockstep.
            wait_for = min(wait_for + self._jitter(), self._window)
            if self._monotonic() + wait_for > deadline:
                raise LibTVError(
                    status_code=429,
                    message=(
                        "libtv third_asset/create rate limit: waited "
                        f"{self._timeout:.0f}s for a slot on this account "
                        f"(limit {self._rate}/min) and gave up"
                    ),
                )
            await self._sleep(wait_for)

    async def _try_acquire(self, account_key: str) -> float:
        if self._redis_usable:
            try:
                return await self._try_acquire_redis(account_key)
            except Exception:  # noqa: BLE001  # any redis fault degrades to the local window, never to a failed generation
                # Latch off rather than retrying per asset: a dead Redis would
                # otherwise add one failing round trip to every registration.
                self._redis_usable = False
                logger.warning(
                    "libtv third_asset limiter: redis unavailable, falling back to the in-process window",
                    exc_info=True,
                )
        return await self._try_acquire_local(account_key)

    async def _try_acquire_redis(self, account_key: str) -> float:
        raw = await self._redis.eval(
            _ACQUIRE_LUA,
            1,
            f"libtv:third-asset-create:{account_key}",
            self._monotonic(),
            self._window,
            self._rate,
            uuid.uuid4().hex,
        )
        if isinstance(raw, bytes):
            raw = raw.decode()
        return float(raw)

    async def _try_acquire_local(self, account_key: str) -> float:
        async with self._locks[account_key]:
            now = self._monotonic()
            bucket = self._local[account_key]
            while bucket and bucket[0] <= now - self._window:
                bucket.popleft()
            if len(bucket) < self._rate:
                bucket.append(now)
                return 0.0
            return bucket[0] + self._window - now


# One limiter per event loop, NOT per LibTVClient: a client is constructed per
# request, so a per-client window would be no window at all -- exactly the hole
# the 2026-09-01 burst went through.
_limiters: dict[Any, ThirdAssetRateLimiter] = {}


def _resolve_limiter_redis() -> Any:
    """Redis for the shared window, or None to fall back in-process.

    The env chain ends at MEDIA_TRANSFER_REDIS_URL (get_transfer_redis's own
    default). In causyn production all three names already point at the same
    drama-redis, so this needs no new environment variable to work.
    """
    try:
        from litellm.llms.libtv.transfer import get_transfer_redis

        return get_transfer_redis(
            os.getenv("LIBTV_THIRD_ASSET_REDIS_URL") or os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL")
        )
    except Exception:  # noqa: BLE001  # no redis just means the in-process window; never fail registration over it
        logger.warning("libtv third_asset limiter: redis client construction failed", exc_info=True)
        return None


def get_third_asset_limiter(redis_client: Any = None) -> ThirdAssetRateLimiter:
    try:
        loop: Optional[Any] = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    for stale in [k for k in _limiters if k is not None and k.is_closed()]:
        _limiters.pop(stale, None)
    limiter = _limiters.get(loop)
    if limiter is None:
        limiter = ThirdAssetRateLimiter(
            redis_client if redis_client is not None else _resolve_limiter_redis()
        )
        _limiters[loop] = limiter
    return limiter


def reset_third_asset_limiters() -> None:
    """Drop cached limiters. For tests -- production builds one per loop and keeps it."""
    _limiters.clear()


def build_rate_limiter(redis_client: Any = None) -> ThirdAssetRateLimiter:
    return ThirdAssetRateLimiter(redis_client)


def iter_unique(values: Iterable[T]) -> list[T]:
    unique, _ = dedupe_preserving_order(list(values))
    return unique


__all__ = [
    "DEFAULT_ACQUIRE_TIMEOUT_SECONDS",
    "DEFAULT_ASSET_CACHE_TTL_SECONDS",
    "DEFAULT_CREATE_RPM",
    "ThirdAssetRateLimiter",
    "acquire_timeout_seconds",
    "asset_cache_ttl_seconds",
    "build_rate_limiter",
    "create_rate_per_minute",
    "get_third_asset_limiter",
    "reset_third_asset_limiters",
    "dedupe_preserving_order",
    "iter_unique",
]
