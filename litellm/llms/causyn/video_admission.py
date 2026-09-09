from __future__ import annotations

from typing import Protocol

from litellm.llms.libtv.transfer import STATUS_TTL_SECONDS, status_key

ACTIVE = "causyn:video:admitted"
MAX_ADMITTED = 8


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


async def admit_video(redis: RedisPort, *, task_id: str, metadata: str, envelope: str, deadline: float) -> bool:
    result = await redis.eval(
        ADMIT,
        4,
        status_key(task_id),
        f"worker:task:metadata:{task_id}",
        "worker:tasks:video_generate",
        ACTIVE,
        task_id,
        metadata,
        envelope,
        STATUS_TTL_SECONDS,
        status_key(""),
        MAX_ADMITTED,
        deadline,
    )
    return result != -1
