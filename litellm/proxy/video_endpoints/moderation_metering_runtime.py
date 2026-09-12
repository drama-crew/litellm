from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.proxy.video_endpoints import moderation_metering_cache as cache
from litellm.proxy.video_endpoints.moderation_metering import (
    BillingBinding,
    CounterFailure,
    MeteringStore,
    PhaseEvent,
    base_counter_keys,
    request_id,
)
from litellm.proxy.video_endpoints.openapi_log_capture import raw_method
from litellm.types.videos.main import CharacterObject, VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider

if TYPE_CHECKING:
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging


@dataclass(frozen=True)
class Scope:
    binding: BillingBinding
    phase: str
    store: MeteringStore


CONTEXT: ContextVar[Scope | None] = ContextVar('moderation_metering_scope', default=None)
SCOPE_KEY = '_moderation_metering_scope'
EVENT_KEY = '_moderation_metering_event'
CALL_TYPES = frozenset(('avideo_generation', 'avideo_remix', 'avideo_edit', 'avideo_extension', 'avideo_status', 'avideo_create_character'))


@dataclass(frozen=True)
class RedisAdapter:
    evaluate: Callable[..., Awaitable[object]]

    async def eval(self, script: str, numkeys: int, *args: str) -> object:
        return await self.evaluate(script, numkeys, *args)


def configured() -> bool:
    return bool(os.getenv('DRAMA_MODERATION_PLATFORM_URL'))


def store() -> MeteringStore:
    from litellm.proxy import proxy_server

    redis_cache = proxy_server.spend_counter_cache.redis_cache
    if proxy_server.prisma_client is None or redis_cache is None:
        raise ValueError('moderation metering requires durable SQL and Redis')
    redis = redis_cache.init_async_client()
    namespace = redis_cache.check_and_fix_namespace('')
    return MeteringStore.from_client(
        proxy_server.prisma_client, RedisAdapter(raw_method(redis, 'eval')), namespace=namespace
    )


class ReservationEntry(BaseModel):
    counter_key: str
    reserved_cost: Decimal


class Reservation(BaseModel):
    entries: tuple[ReservationEntry, ...] = ()


async def prepare(binding: BillingBinding, reservation: object) -> Scope:
    authority = store()
    entries = Reservation.model_validate(reservation or {}).entries
    await authority.prepare(binding, {entry.counter_key: entry.reserved_cost for entry in entries})
    return Scope(binding, binding.expected_phases[0], authority)


async def continuation(intent_id: str, phase: str) -> Scope:
    authority = store()
    binding = await authority.binding(intent_id)
    if phase not in binding.expected_phases:
        raise ValueError('moderation billing phase mismatch')
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
            result._hidden_params['_moderation_metering_pending'] = True
        logging.getLogger(__name__).warning('moderation metering result retains unknown financial phase')


async def _handoff(logging_obj: Logging, result: object, start_time: datetime, end_time: datetime) -> None:
    scope = logging_obj.model_call_details.get(SCOPE_KEY)
    if not isinstance(scope, Scope) or not isinstance(result, (VideoObject, CharacterObject)):
        return
    if scope.phase == 'completion' and isinstance(result, VideoObject) and result.status not in ('completed', 'failed', 'cancelled'):
        return
    logging_obj._process_hidden_params_and_response_cost(result, start_time, end_time, emit=False)
    raw = logging_obj.model_call_details.get('response_cost')
    decoded = decode_video_id_with_provider(result.id)
    provider = TypeAdapter(str).validate_python(decoded.get('custom_llm_provider') or logging_obj.custom_llm_provider or '')
    params = TypeAdapter(dict[str, object]).validate_python(logging_obj.model_call_details.get('litellm_params') or {})
    model_info = TypeAdapter(dict[str, JsonValue]).validate_python(params.get('model_info') or {})
    deployment = TypeAdapter(str).validate_python(decoded.get('model_id') or model_info.get('id') or '')
    event = PhaseEvent(
        binding=scope.binding, request_id=request_id(scope.binding, scope.phase), phase=scope.phase,
        provider=provider, deployment_id=deployment, native_id=result.id,
        provider_task_id=TypeAdapter(str).validate_python(decoded.get('video_id') or result.id),
        amount=TypeAdapter(Decimal | None).validate_python(raw), finalized=raw is not None,
    )
    result._hidden_params[EVENT_KEY] = event.model_dump(mode='json')
    try:
        async with asyncio.timeout(5):
            await scope.store.persist(event)
    except Exception:
        result._hidden_params['_moderation_metering_pending'] = True
        logging.getLogger(__name__).warning('moderation metering native result awaits durable financial recovery')


def private_event(result: object) -> dict[str, JsonValue] | None:
    if not isinstance(result, (VideoObject, CharacterObject)):
        return None
    value = result._hidden_params.get(EVENT_KEY)
    if value is None:
        return None
    return TypeAdapter(dict[str, JsonValue]).validate_python(PhaseEvent.model_validate(value).model_dump(mode='json'))


async def check_budget(counter_key: str) -> None:
    if configured():
        await store().assert_admission((counter_key,))


async def guarded_increment(counter_key: str, amount: float) -> float | None:
    if not configured():
        return None
    authority = store()
    await authority.assert_admission((counter_key,))
    result = TypeAdapter(tuple[str, ...]).validate_python(await authority.redis.eval(
        cache.INCREMENT, 2, *cache.keys(counter_key, authority.namespace), str(amount)
    ))
    if result[0] == 'legacy':
        return None
    if result[0] != 'ok':
        await authority.quarantine((counter_key,))
        raise CounterFailure((counter_key,), result[0])
    return float(result[1])


async def protect_invalidation(counter_key: str) -> bool:
    if not configured():
        return False
    authority = store()
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(await authority.db.query_raw(
        'SELECT counter_key FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=$1', counter_key
    ))
    if not rows:
        return False
    await authority.quarantine((counter_key,))
    logging.getLogger(__name__).error('moderation protected spend counter requires reconciliation')
    return True


async def consume(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            async with asyncio.timeout(15):
                await store().run_once()
        except Exception:
            logging.getLogger(__name__).warning('moderation metering durable retry pending')
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass
