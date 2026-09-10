from __future__ import annotations

import asyncio
import time
import uuid
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue

from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError, RewriteResult

TTL = 7 * 86400
LEASE = 240
PENDING = "causyn:context-ir:pending"
READY = "causyn:context-ir:ready"
PREFIX = "h3_ir_"


class RedisPort(Protocol):
    async def get(self, name: str) -> bytes | str | None: ...
    async def eval(self, script: str, numkeys: int, *values: str | float) -> object: ...
    async def zrangebyscore(
        self, name: str, minimum: float | str, maximum: float | str, *, start: int, num: int
    ) -> list[bytes | str]: ...
    async def zrevrange(self, name: str, start: int, end: int) -> list[bytes | str]: ...


class BillingIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    api_key: str | None = None
    team_id: str | None = None
    user_id: str | None = None
    organization_id: str | None = None


class Notification(BaseModel):
    model_config = ConfigDict(frozen=True)
    body: dict[str, JsonValue]
    attempts: int = 0


class ContextIRTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    owner: str
    request: ContextIRRequest
    created_at: int
    updated_at: int
    trace_created_ns: int | None = None
    rewrite_completed_ns: int | None = None
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    result: RewriteResult | None = None
    error: str | None = None
    attempts: int = 0
    billing: BillingIdentity
    price: float
    reservation: dict[str, JsonValue] | None = None
    video_payload: dict[str, JsonValue] | None = None
    settled: bool = False
    listed: bool = True
    notifications: tuple[Notification, ...] = ()

    def public(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "model": self.request.model,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration": self.request.duration,
            "ratio": self.request.effective_ratio,
            "task_type": "h3_context_ir",
            "modality": "text",
            "usage": self.result.usage.model_dump(mode="json", exclude={"cost"})
            if self.result and self.status == "succeeded"
            else {},
            **({"content": {"prompt": self.result.prompt}} if self.result and self.status == "succeeded" else {}),
            **({"error": {"code": "1000", "message": self.error}} if self.error else {}),
        }


def task_key(task_id: str) -> str:
    return f"causyn:context-ir:task:{task_id}"


def owner_key(owner: str) -> str:
    return f"causyn:context-ir:owner:{owner}"


def text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


class ContextIRStore:
    def __init__(self, redis: RedisPort, *, max_pending: int = 128) -> None:
        self.redis = redis
        self.max_pending = max_pending

    async def create(self, task: ContextIRTask) -> ContextIRTask:
        result = await self.redis.eval(
            "if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end "
            "if redis.call('ZCARD', KEYS[2]) + redis.call('ZCARD', KEYS[4]) >= tonumber(ARGV[6]) then return -1 end; "
            "redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2]); "
            "redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4]); "
            "if ARGV[5] == '1' then redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', tonumber(ARGV[3])-tonumber(ARGV[2])); "
            "redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4]); redis.call('EXPIRE', KEYS[3], ARGV[2]) end; "
            "return 1",
            4,
            task_key(task.id),
            PENDING,
            owner_key(task.owner),
            READY,
            task.model_dump_json(),
            TTL,
            task.created_at,
            task.id,
            "1" if task.listed else "0",
            self.max_pending,
        )
        if result == -1:
            raise RewriteError("Context IR queue is full; retry later", 429)
        stored = await self.get(task.id)
        if stored is None:
            raise RewriteError("Context IR task persistence failed", 503)
        if stored.owner != task.owner or stored.request != task.request or stored.price != task.price:
            raise RewriteError("Idempotency key was used with a different request", 409)
        return stored

    async def get(self, task_id: str) -> ContextIRTask | None:
        raw = await self.redis.get(task_key(task_id))
        return ContextIRTask.model_validate_json(raw) if raw is not None else None

    async def due(self, queue: str = PENDING, *, limit: int = 4) -> tuple[str, ...]:
        return tuple(
            text(item) for item in await self.redis.zrangebyscore(queue, "-inf", time.time(), start=0, num=limit)
        )

    async def claim(self, task_id: str, token: str) -> bool:
        result = await self.redis.eval(
            "if redis.call('EXISTS', KEYS[1]) == 0 then "
            "redis.call('ZREM', KEYS[3], ARGV[3]); redis.call('ZREM', KEYS[4], ARGV[3]); return 0 end; "
            "if not redis.call('SET', KEYS[2], ARGV[1], 'NX', 'EX', ARGV[2]) then return 0 end; "
            "local queue = redis.call('ZSCORE', KEYS[4], ARGV[3]) and KEYS[4] or KEYS[3]; "
            "redis.call('ZADD', queue, ARGV[4], ARGV[3]); return 1",
            4,
            task_key(task_id),
            task_key(task_id) + ":lease",
            PENDING,
            READY,
            token,
            LEASE,
            task_id,
            time.time() + LEASE,
        )
        return result == 1

    async def save(self, task: ContextIRTask, token: str, *, done: bool = False) -> None:
        result = await self.redis.eval(
            "if redis.call('GET', KEYS[2]) ~= ARGV[1] or redis.call('EXISTS', KEYS[1]) == 0 then return 0 end; "
            "redis.call('SET', KEYS[1], ARGV[2], 'KEEPTTL'); "
            "local due = redis.call('ZSCORE', KEYS[3], ARGV[4]) or redis.call('ZSCORE', KEYS[4], ARGV[4]); "
            "redis.call('ZREM', KEYS[3], ARGV[4]); redis.call('ZREM', KEYS[4], ARGV[4]); "
            "if ARGV[3] == '1' then redis.call('DEL', KEYS[2]); "
            "else redis.call('ZADD', KEYS[tonumber(ARGV[5])], due or ARGV[6], ARGV[4]) end; return 1",
            4,
            task_key(task.id),
            task_key(task.id) + ":lease",
            PENDING,
            READY,
            token,
            task.model_dump_json(),
            "1" if done else "0",
            task.id,
            4 if task.result is not None or task.status in {"succeeded", "failed", "cancelled"} else 3,
            time.time() + LEASE,
        )
        if result != 1:
            raise RewriteError("Context IR task lease was lost", 503)

    async def retry(self, task_id: str, token: str, delay: float) -> None:
        await self.redis.eval(
            "if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end; "
            "local queue = redis.call('ZSCORE', KEYS[3], ARGV[3]) and KEYS[3] or KEYS[2]; "
            "redis.call('DEL', KEYS[1]); redis.call('ZADD', queue, ARGV[2], ARGV[3]); return 1",
            3,
            task_key(task_id) + ":lease",
            PENDING,
            READY,
            token,
            time.time() + delay,
            task_id,
        )

    async def list_tasks(self, owner: str) -> tuple[ContextIRTask, ...]:
        ids = await self.redis.zrevrange(owner_key(owner), 0, -1)
        rows = tuple(await asyncio.gather(*(self.get(text(item)) for item in ids)))
        return tuple(row for row in rows if row is not None and row.owner == owner and row.listed)

    async def delete(self, task: ContextIRTask, token: str) -> None:
        result = await self.redis.eval(
            "if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end; "
            "redis.call('DEL', KEYS[1], KEYS[2]); redis.call('ZREM', KEYS[3], ARGV[2]); "
            "redis.call('ZREM', KEYS[4], ARGV[2]); redis.call('ZREM', KEYS[5], ARGV[2]); return 1",
            5,
            task_key(task.id),
            task_key(task.id) + ":lease",
            PENDING,
            owner_key(task.owner),
            READY,
            token,
            task.id,
        )
        if result != 1:
            raise RewriteError("Context IR task is busy", 409)


def new_task_id(owner: str) -> str:
    return PREFIX + uuid.uuid4().hex
