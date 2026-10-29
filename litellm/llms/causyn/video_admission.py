from __future__ import annotations

import os
from typing import Protocol

from litellm.llms.causyn.video_priority import priority_route
from litellm.llms.libtv.transfer import STATUS_TTL_SECONDS, status_key

ACTIVE = "causyn:video:admitted"
MAX_ADMITTED = 8
PRODUCTION_MAX_ADMITTED_ENV = "CAUSYN_H3_MAX_ADMITTED"
TEST_POOL = "test"
TEST_POOL_MAX_ADMITTED_ENV = "CAUSYN_TEST_POOL_MAX_ADMITTED"
TEST_POOL_MAX_ADMITTED_DEFAULT = 2


def active_key(pool: str | None = None) -> str:
    """Active-set key: production keeps the original global key, the test pool has its own."""
    return ACTIVE if pool is None else f"{ACTIVE}:{pool}"


def max_admitted(pool: str | None = None) -> int:
    """Bound queued plus claimed work independently for each worker pool."""
    env = PRODUCTION_MAX_ADMITTED_ENV if pool is None else TEST_POOL_MAX_ADMITTED_ENV
    default = MAX_ADMITTED if pool is None else TEST_POOL_MAX_ADMITTED_DEFAULT
    try:
        value = int(os.getenv(env, ""))
    except ValueError:
        return default
    return value if value >= 1 else default


class RedisPort(Protocol):
    async def eval(self, script: str, numkeys: int, *values: str | float) -> object: ...


ADMIT = """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
for _, id in ipairs(redis.call('ZRANGE', KEYS[4], 0, -1)) do
    local status = redis.call('GET', ARGV[5] .. id)
    if not status or (status ~= 'queued' and status ~= 'claimed') then
        redis.call('ZREM', KEYS[4], id)
    end
end
if redis.call('ZCARD', KEYS[4]) >= tonumber(ARGV[6]) then return -1 end
redis.call('SET', KEYS[1], 'queued', 'EX', ARGV[4])
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[4])
redis.call('XADD', KEYS[3], '*', 'payload', ARGV[3])
redis.call('ZADD', KEYS[4], ARGV[7], ARGV[1])
return 1
"""

PRIORITY_ADMIT = (
    """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
local kind = redis.call('TYPE', KEYS[5]).ok
if kind ~= 'none' and kind ~= 'zset' then return -1 end
local stream_kind = redis.call('TYPE', KEYS[3]).ok
if stream_kind ~= 'none' and stream_kind ~= 'stream' then return -1 end
local function admit()
"""
    + ADMIT
    + """
end
local result = admit()
if result == 1 then
    redis.call('ZADD', KEYS[5], ARGV[8], KEYS[3])
end
return result
"""
)


async def admit_video(
    redis: RedisPort,
    *,
    task_id: str,
    metadata: str,
    envelope: str,
    deadline: float,
    stream: str = "worker:tasks:video_generate",
    pool: str | None = None,
    request: object = None,
) -> bool:
    """Admit one causyn video task.

    ``stream`` must be the stream for the task type in the envelope. Ref2VA has
    its own stream and its own worker, so hardcoding the shared one here put
    ref2va work on a queue whose worker cannot run it -- the envelope said
    ``video_generate_ref2va`` while the XADD went to ``video_generate``
    (observed in production 2026-09-15). The default keeps every non-ref2va
    caller unchanged.

    ``pool`` selects the active set and its cap (``None`` = production, unchanged). The Lua script sweeps
    terminal tasks out of whichever set it is given, so a test task is released from the test set only.
    """
    route = priority_route(stream, model="causyn-1.1", request=request, pool=pool)
    keys = (
        status_key(task_id),
        f"worker:task:metadata:{task_id}",
        route.stream,
        active_key(pool),
    )
    values = (
        task_id,
        metadata,
        envelope,
        STATUS_TTL_SECONDS,
        status_key(""),
        max_admitted(pool),
        deadline,
    )
    result = (
        await redis.eval(PRIORITY_ADMIT, 5, *keys, route.catalog_key, *values, route.priority)
        if route.catalog_key is not None
        else await redis.eval(ADMIT, 4, *keys, *values)
    )
    return result != -1
