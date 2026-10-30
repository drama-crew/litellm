from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, SerializerFunctionWrapHandler, TypeAdapter, model_serializer

from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError, RewriteResult

TTL = 7 * 86400
LEASE = 240
PENDING = "causyn:context-ir:pending"
READY = "causyn:context-ir:ready"
PREFIX = "h3_ir_"
PENDING_AGING_ENV = "CAUSYN_H3_PENDING_AGING_ENABLED"
PENDING_AGING_SECONDS = 60
PENDING_SCAN_LIMIT = 128
_PENDING_AGED_DUE = """
local now, age, count = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local rows = {}
for i = 2, #KEYS do
    local id = ARGV[i + 2]
    local due = tonumber(redis.call('ZSCORE', KEYS[1], id))
    if due and due <= now then
        local raw = redis.pcall('GET', KEYS[i])
        local created = type(raw) == 'string' and tonumber(raw) or nil
        if not created or created ~= created or created <= 0 or created > now then
            return redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, count)
        end
        rows[#rows + 1] = {id=id, due=due, created=created, aged=now-created >= age}
    end
end
table.sort(rows, function(a, b)
    if a.aged ~= b.aged then return a.aged end
    if a.aged and a.created ~= b.created then return a.created < b.created end
    if a.due ~= b.due then return a.due < b.due end
    return a.id < b.id
end)
local selected = {}
for i = 1, math.min(count, #rows) do selected[i] = rows[i].id end
return selected
"""


def pending_aging_enabled() -> bool:
    return os.getenv(PENDING_AGING_ENV, "false").strip().lower() in ("true", "1", "yes", "on")


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
    # Redacted upstream status/message; stored for diagnosis, never in public().
    error_detail: str | None = None
    # Start of the first rewrite attempt (unix seconds); the retry window runs from here.
    rewrite_started_at: float | None = None
    attempts: int = 0
    # Attempts whose answer the structural validators rejected. Counted apart from `attempts` so provider
    # 429s/outages do not eat the (small) budget for re-sampling a bad answer. Default 0 keeps old blobs valid.
    validation_failures: int = 0
    # 投递（进 GPU 队列）的重试计数，与改写的 attempts 分开。两个阶段的失败语义
    # 不同：改写失败有 3 次上限，投递遇到的是下游背压，上界由视频 deadline 决定。
    # 共用一个计数器会让退避永远停在同一档（改写阶段结束后它就不再增长）。
    # 默认 0 使旧的持久化 blob 保持兼容。
    deliver_attempts: int = 0
    billing: BillingIdentity
    price: float
    reservation: dict[str, JsonValue] | None = None
    metering_binding_json: str | None = None
    financial_actual: float | None = None
    financial_facts_json: str | None = None
    financial_event_json: str | None = None
    video_payload: dict[str, JsonValue] | None = None
    settled: bool = False
    listed: bool = True
    notifications: tuple[Notification, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_untouched_validation_failures(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Rollback safety: the model is extra="forbid", so an older image would reject a blob carrying the new key.
        Keep blobs byte-compatible with the previous schema unless a validation retry actually happened."""
        data: dict[str, object] = handler(self)
        if data.get("validation_failures") == 0:
            data.pop("validation_failures", None)
        return data

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


def created_key(task_id: str) -> str:
    return f"{task_key(task_id)}:created_at"


def aging_keys(task_id: str) -> tuple[str, ...]:
    return (created_key(task_id),) if pending_aging_enabled() else ()


def owner_key(owner: str) -> str:
    return f"causyn:context-ir:owner:{owner}"


def text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


class ContextIRStore:
    def __init__(self, redis: RedisPort, *, max_pending: int = 128) -> None:
        self.redis = redis
        self.max_pending = max_pending

    async def create(self, task: ContextIRTask) -> ContextIRTask:
        extra_keys = aging_keys(task.id)
        result = await self.redis.eval(
            "if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end "
            # 只计 PENDING（改写队列）。READY 里是"改写已完成、等 GPU"的任务；
            # 把它算进同一个预算，等于让渲染积压反压到改写入口——渲染慢到堆满预算
            # 时，所有新 IR 请求都被 429，包括根本不碰 GPU 的纯文本改写。
            # READY 的深度另有界：视频准入闸(MAX_ADMITTED)与任务 deadline 会让它
            # 自行排空，不需要靠拒绝上游来限制。
            "if redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[6]) then return -1 end; "
            "redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2]); "
            "redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4]); "
            "if ARGV[5] == '1' then redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', tonumber(ARGV[3])-tonumber(ARGV[2])); "
            "redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4]); redis.call('EXPIRE', KEYS[3], ARGV[2]) end; "
            + ("redis.call('SET', KEYS[5], ARGV[3], 'EX', ARGV[2]); " if extra_keys else "")
            + "return 1",
            4 + len(extra_keys),
            task_key(task.id),
            PENDING,
            owner_key(task.owner),
            READY,
            *extra_keys,
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
        if queue != PENDING or not pending_aging_enabled():
            return tuple(
                text(item) for item in await self.redis.zrangebyscore(queue, "-inf", time.time(), start=0, num=limit)
            )
        count = max(0, min(limit, PENDING_SCAN_LIMIT))
        if count == 0:
            return ()
        now = time.time()
        candidates = tuple(
            text(item) for item in await self.redis.zrangebyscore(queue, "-inf", now, start=0, num=PENDING_SCAN_LIMIT)
        )
        if not candidates:
            return ()
        selected = await self.redis.eval(
            _PENDING_AGED_DUE,
            1 + len(candidates),
            queue,
            *(created_key(task_id) for task_id in candidates),
            now,
            PENDING_AGING_SECONDS,
            count,
            *candidates,
        )
        return tuple(text(item) for item in TypeAdapter(tuple[bytes | str, ...]).validate_python(selected))

    async def claim(self, task_id: str, token: str) -> bool:
        extra_keys = aging_keys(task_id)
        result = await self.redis.eval(
            "if redis.call('EXISTS', KEYS[1]) == 0 then "
            "redis.call('ZREM', KEYS[3], ARGV[3]); redis.call('ZREM', KEYS[4], ARGV[3]); "
            + ("redis.call('DEL', KEYS[5]); " if extra_keys else "")
            + "return 0 end; "
            "if not redis.call('SET', KEYS[2], ARGV[1], 'NX', 'EX', ARGV[2]) then return 0 end; "
            "local queue = redis.call('ZSCORE', KEYS[4], ARGV[3]) and KEYS[4] or KEYS[3]; "
            "redis.call('ZADD', queue, ARGV[4], ARGV[3]); return 1",
            4 + len(extra_keys),
            task_key(task_id),
            task_key(task_id) + ":lease",
            PENDING,
            READY,
            *extra_keys,
            token,
            LEASE,
            task_id,
            time.time() + LEASE,
        )
        return result == 1

    async def save(self, task: ContextIRTask, token: str, *, done: bool = False) -> None:
        extra_keys = aging_keys(task.id)
        result = await self.redis.eval(
            "if redis.call('GET', KEYS[2]) ~= ARGV[1] or redis.call('EXISTS', KEYS[1]) == 0 then return 0 end; "
            "redis.call('SET', KEYS[1], ARGV[2], 'KEEPTTL'); "
            "local due = redis.call('ZSCORE', KEYS[3], ARGV[4]) or redis.call('ZSCORE', KEYS[4], ARGV[4]); "
            "redis.call('ZREM', KEYS[3], ARGV[4]); redis.call('ZREM', KEYS[4], ARGV[4]); "
            "if ARGV[3] == '1' then redis.call('DEL', KEYS[2]); "
            "else redis.call('ZADD', KEYS[tonumber(ARGV[5])], due or ARGV[6], ARGV[4]) end; "
            + (
                "if ARGV[3] == '1' or ARGV[5] ~= '3' then redis.call('DEL', KEYS[5]); "
                "else local ttl = redis.call('PTTL', KEYS[1]); "
                "if ttl > 0 then redis.call('SET', KEYS[5], ARGV[7], 'NX', 'PX', ttl) end end; "
                if extra_keys
                else ""
            )
            + "return 1",
            4 + len(extra_keys),
            task_key(task.id),
            task_key(task.id) + ":lease",
            PENDING,
            READY,
            *extra_keys,
            token,
            task.model_dump_json(),
            "1" if done else "0",
            task.id,
            4 if task.result is not None or task.status in {"succeeded", "failed", "cancelled"} else 3,
            time.time() + LEASE,
            *((task.created_at,) if extra_keys else ()),
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
        extra_keys = aging_keys(task.id)
        result = await self.redis.eval(
            "if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end; "
            "redis.call('DEL', KEYS[1], KEYS[2]); redis.call('ZREM', KEYS[3], ARGV[2]); "
            "redis.call('ZREM', KEYS[4], ARGV[2]); redis.call('ZREM', KEYS[5], ARGV[2]); "
            + ("redis.call('DEL', KEYS[6]); " if extra_keys else "")
            + "return 1",
            5 + len(extra_keys),
            task_key(task.id),
            task_key(task.id) + ":lease",
            PENDING,
            owner_key(task.owner),
            READY,
            *extra_keys,
            token,
            task.id,
        )
        if result != 1:
            raise RewriteError("Context IR task is busy", 409)


def new_task_id(owner: str) -> str:
    return PREFIX + uuid.uuid4().hex
