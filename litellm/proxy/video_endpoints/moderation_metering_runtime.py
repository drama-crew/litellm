from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, JsonValue, TypeAdapter

import litellm
from litellm.proxy.video_endpoints.moderation_metering import (
    BillingBinding,
    MeteringStore,
    PhaseEvent,
    request_id,
)
from litellm.proxy.video_endpoints.openapi_log_capture import raw_method
from litellm.types.videos.main import CharacterObject, VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore


@dataclass(frozen=True)
class Scope:
    binding: BillingBinding
    phase: str
    store: MeteringStore


CONTEXT: ContextVar[Scope | None] = ContextVar("moderation_metering_scope", default=None)
SCOPE_KEY = "_moderation_metering_scope"
EVENT_KEY = "_moderation_metering_event"
CALL_TYPES = frozenset(
    ("avideo_generation", "avideo_remix", "avideo_edit", "avideo_extension", "avideo_status", "avideo_create_character")
)


@dataclass(frozen=True)
class RedisAdapter:
    evaluate: Callable[..., Awaitable[object]]

    async def eval(self, script: str, numkeys: int, *args: str) -> object:
        value = await self.evaluate(script, numkeys, *args)
        return value.decode() if isinstance(value, bytes) else value


def configured() -> bool:
    return os.getenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "false").lower() == "true"


async def counter_mode(counter_key: str) -> bool:
    from litellm.proxy import proxy_server

    redis_cache = proxy_server.spend_counter_cache.redis_cache
    if redis_cache is None:
        if configured():
            raise ValueError("protected budget writer requires Redis")
        return False
    try:
        mode = await redis_cache.async_get_cache(key="protected:mode")
    except Exception:
        if configured() or getattr(redis_cache, "protected_budget_mode", False) is True:
            raise
        return False
    if mode == "protected-v1":
        redis_cache.protected_budget_mode = True
    if getattr(redis_cache, "protected_budget_mode", False) is True and not configured():
        raise ValueError("protected budget writer configuration mismatch; maintenance fencing required")
    return configured()


def store() -> MeteringStore:
    from litellm.proxy import proxy_server

    redis_cache = proxy_server.spend_counter_cache.redis_cache
    if proxy_server.prisma_client is None or redis_cache is None:
        raise ValueError("moderation metering requires durable SQL and Redis")
    redis = redis_cache.init_async_client()
    namespace = redis_cache.check_and_fix_namespace("")
    return MeteringStore.from_client(
        proxy_server.prisma_client, RedisAdapter(raw_method(redis, "eval")), namespace=namespace
    )


class ReservationEntry(BaseModel):
    counter_key: str
    reserved_cost: Decimal


class Reservation(BaseModel):
    reservation_id: str | None = None
    entries: tuple[ReservationEntry, ...] = ()


async def prepare(binding: BillingBinding, reservation: object) -> Scope:
    authority = store()
    frozen = Reservation.model_validate(reservation or {})
    await authority.prepare(
        binding,
        {entry.counter_key: entry.reserved_cost for entry in frozen.entries},
        reservation_id=frozen.reservation_id,
    )
    return Scope(binding, binding.expected_phases[0], authority)


async def continuation(intent_id: str, phase: str) -> Scope:
    authority = store()
    binding = await authority.binding(intent_id)
    if phase not in binding.expected_phases:
        raise ValueError("moderation billing phase mismatch")
    return Scope(binding, phase, authority)


def attach(logging_obj: Logging, call_type: str) -> None:
    scope = CONTEXT.get()
    if scope is not None and call_type in CALL_TYPES:
        logging_obj.model_call_details[SCOPE_KEY] = scope


def owns(value: object) -> bool:
    return isinstance(value, Scope)


async def handoff(logging_obj: Logging, result: object, start_time: datetime, end_time: datetime) -> None:
    try:
        await _handoff(logging_obj, result, start_time, end_time)
    except Exception:
        if isinstance(result, (VideoObject, CharacterObject)):
            result._hidden_params["_moderation_metering_pending"] = True
        logging.getLogger(__name__).warning("moderation metering result retains unknown financial phase")


async def _handoff(logging_obj: Logging, result: object, start_time: datetime, end_time: datetime) -> None:
    scope = logging_obj.model_call_details.get(SCOPE_KEY)
    if not isinstance(scope, Scope) or not isinstance(result, (VideoObject, CharacterObject)):
        return
    if (
        scope.phase == "completion"
        and isinstance(result, VideoObject)
        and result.status not in ("completed", "failed", "cancelled")
    ):
        return
    logging_obj._process_hidden_params_and_response_cost(result, start_time, end_time, emit=False)
    raw = logging_obj.model_call_details.get("response_cost")
    decoded = decode_video_id_with_provider(result.id)
    provider = TypeAdapter(str).validate_python(
        decoded.get("custom_llm_provider") or logging_obj.custom_llm_provider or ""
    )
    params = TypeAdapter(dict[str, object]).validate_python(logging_obj.model_call_details.get("litellm_params") or {})
    model_info = TypeAdapter(dict[str, JsonValue]).validate_python(params.get("model_info") or {})
    deployment = TypeAdapter(str).validate_python(decoded.get("model_id") or model_info.get("id") or "")
    event = PhaseEvent(
        binding=scope.binding,
        request_id=request_id(scope.binding, scope.phase),
        phase=scope.phase,
        provider=provider,
        deployment_id=deployment,
        native_id=result.id,
        provider_task_id=TypeAdapter(str).validate_python(decoded.get("video_id") or result.id),
        amount=TypeAdapter(Decimal | None).validate_python(raw),
        finalized=raw is not None,
    )
    result._hidden_params[EVENT_KEY] = event.model_dump(mode="json")
    try:
        async with asyncio.timeout(5):
            await scope.store.persist(event)
    except Exception:
        result._hidden_params["_moderation_metering_pending"] = True
        logging.getLogger(__name__).warning(
            "moderation metering native result awaits durable financial recovery", exc_info=True
        )


def private_event(result: object) -> dict[str, JsonValue] | None:
    if not isinstance(result, (VideoObject, CharacterObject)):
        return None
    value = result._hidden_params.get(EVENT_KEY)
    if value is None:
        return None
    return TypeAdapter(dict[str, JsonValue]).validate_python(PhaseEvent.model_validate(value).model_dump(mode="json"))


async def check_budget(counter_key: str) -> None:
    if await counter_mode(counter_key):
        await store().assert_admission((counter_key,))


async def guarded_increment(counter_key: str, amount: float) -> float | None:
    if not await counter_mode(counter_key):
        return None
    authority = store()
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore

    protected = ProtectedBudgetStore(
        authority.db, authority.transactions, authority.redis, namespace=authority.namespace
    )
    states = await protected.assert_admission((counter_key,))
    if not states:
        return None
    if projected(counter_key):
        return await protected_value(counter_key)
    raise ValueError("protected counter mutation requires durable operation or reservation identity")


async def protect_invalidation(counter_key: str) -> bool:
    if not await counter_mode(counter_key):
        return False
    authority = store()
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
        await authority.db.query_raw(
            'SELECT counter_key FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=$1', counter_key
        )
    )
    if not rows:
        return False
    await authority.quarantine((counter_key,))
    logging.getLogger(__name__).error("moderation protected spend counter requires reconciliation")
    return True


async def consume(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            async with asyncio.timeout(15):
                await store().run_once()
        except Exception:
            logging.getLogger(__name__).warning("moderation metering durable retry pending", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass


@dataclass(frozen=True)
class Projection:
    counter_keys: frozenset[str]


PROJECTION: ContextVar[Projection | None] = ContextVar("protected_budget_projection", default=None)


def projected(counter_key: str) -> bool:
    current = PROJECTION.get()
    return current is not None and counter_key in current.counter_keys


async def protected_authority(counter_key: str) -> ProtectedBudgetStore | None:
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore

    if not await counter_mode(counter_key):
        return None
    authority = store()
    protected = ProtectedBudgetStore(
        authority.db, authority.transactions, authority.redis, namespace=authority.namespace
    )
    if not await protected.assert_admission((counter_key,)):
        return None
    return protected


async def protected_value(counter_key: str) -> float | None:
    authority = await protected_authority(counter_key)
    if authority is None:
        return None
    value = await authority.redis.eval('return redis.call("GET",KEYS[1])', 1, authority.namespace + counter_key)
    return TypeAdapter(float).validate_python(value)


async def reserve_registered(
    counter_key: str, reservation_id: str, amount: float, valid_until: datetime
) -> float | None:
    authority = await protected_authority(counter_key)
    if authority is None:
        return None
    operation_id = "reserve:" + reservation_id + ":" + counter_key
    previous = await authority.operation(operation_id)
    if previous and previous.payload.changes[0].reservation is not None:
        valid_until = previous.payload.changes[0].reservation.valid_until
    await authority.mutate(
        operation_id,
        (counter_key,),
        kind="reserve",
        amount=Decimal(str(amount)),
        reservation_id=reservation_id,
        valid_until=valid_until,
    )
    return await protected_value(counter_key)


async def adjust_registered(
    counter_key: str, reservation_id: str, amount: float, *, resize: bool, release: bool
) -> bool:
    authority = await protected_authority(counter_key)
    if authority is None:
        return False
    if projected(counter_key):
        return True
    kind = "resize" if resize else "release" if release else "debit"
    identity = kind + ":" + reservation_id + ":" + counter_key + (":" + str(amount) if resize else "")
    await authority.mutate(
        identity, (counter_key,), kind=kind, amount=Decimal(str(amount)), reservation_id=reservation_id
    )
    return True


class CounterKey(BaseModel):
    counter_key: str


async def settle_legacy(
    operation_id: str,
    *,
    token: str | None,
    user_id: str | None,
    team_id: str | None,
    org_id: str | None,
    end_user_id: str | None,
    tags: tuple[str, ...],
    amount: float | None,
    reservation_id: str | None,
) -> Projection:
    from litellm.proxy.proxy_server import litellm_proxy_budget_name
    from litellm.proxy.spend_tracking.protected_budget import BudgetIdentity, ProtectedBudgetStore

    targets = (
        *((BudgetIdentity(kind="key", identity=token),) if token else ()),
        *((BudgetIdentity(kind="user", identity=user_id),) if user_id else ()),
        *(
            (BudgetIdentity(kind="user", identity=litellm_proxy_budget_name),)
            if litellm.max_budget > 0 and litellm_proxy_budget_name
            else ()
        ),
        *((BudgetIdentity(kind="team", identity=team_id),) if team_id else ()),
        *((BudgetIdentity(kind="team_member", identity=user_id, team_id=team_id),) if user_id and team_id else ()),
        *((BudgetIdentity(kind="org", identity=org_id),) if org_id else ()),
        *((BudgetIdentity(kind="end_user", identity=end_user_id),) if end_user_id else ()),
        *(BudgetIdentity(kind="tag", identity=tag) for tag in tags),
    )
    if not targets or not await counter_mode(targets[0].counter_key):
        return Projection(frozenset())
    authority = store()
    protected = ProtectedBudgetStore(
        authority.db, authority.transactions, authority.redis, namespace=authority.namespace
    )
    groups = await asyncio.gather(
        *(
            authority.db.query_raw(
                'SELECT counter_key FROM "LiteLLM_ModerationMeteringCounter" '
                "WHERE target->>'kind'=$1 AND target->>'identity'=$2 AND COALESCE(target->>'team_id','')=$3 "
                'UNION SELECT counter_key FROM "LiteLLM_BudgetBirth" '
                "WHERE target->>'kind'=$1 AND target->>'identity'=$2 AND COALESCE(target->>'team_id','')=$3",
                target.kind,
                target.identity,
                target.team_id or "",
            )
            for target in targets
        )
    )
    counter_keys = tuple(
        sorted(
            {row.counter_key for group in groups for row in TypeAdapter(tuple[CounterKey, ...]).validate_python(group)}
        )
    )
    if not counter_keys:
        return Projection(frozenset())
    if not operation_id.strip():
        raise ValueError("stable server operation identity required")
    if amount is None:
        raise ValueError("protected actual fee remains unknown")
    await protected.mutate(
        "legacy:" + operation_id, counter_keys, kind="debit", amount=Decimal(str(amount)), reservation_id=reservation_id
    )
    return Projection(frozenset(counter_keys))


async def reset_registered(counter_key: str, boundary: datetime, reset_at: datetime | None) -> bool:
    from datetime import timezone

    authority = await protected_authority(counter_key)
    if authority is None:
        return False
    normalized = boundary.replace(tzinfo=timezone.utc) if boundary.tzinfo is None else boundary
    next_reset = reset_at.replace(tzinfo=timezone.utc) if reset_at is not None and reset_at.tzinfo is None else reset_at
    operation_id = "reset:" + counter_key + ":" + normalized.isoformat()
    previous = await authority.operation(operation_id)
    if previous:
        next_reset = previous.payload.changes[0].reset_at
    await authority.reset(operation_id, counter_key, boundary=normalized, reset_at=next_reset)
    return True


async def reset_linked_registered(counter_key: str, budget_id: str | None) -> bool:
    authority = await protected_authority(counter_key)
    if authority is None:
        return False
    if not budget_id:
        raise ValueError("protected budget link has no period identity")
    rows = TypeAdapter(tuple[dict[str, datetime | None], ...]).validate_python(
        await authority.db.query_raw(
            'SELECT budget_reset_at FROM "LiteLLM_BudgetTable" WHERE budget_id=$1',
            budget_id,
        )
    )
    if len(rows) != 1 or rows[0]["budget_reset_at"] is None:
        raise ValueError("protected linked budget period missing")
    return await reset_registered(counter_key, rows[0]["budget_reset_at"], None)


def projection_tags(value: object) -> tuple[str, ...]:
    adapter = TypeAdapter(tuple[str, ...])
    tags = adapter.validate_json(value) if isinstance(value, str) else adapter.validate_python(value or ())
    return tuple(sorted(set(tags)))
