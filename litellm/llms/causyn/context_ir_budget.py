from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, JsonValue

from litellm.llms.causyn.h3_prompt import RewriteError


class ReservationEntry(BaseModel):
    reservation_id: str | None = None
    counter_key: str
    reserved_cost: float | None = None
    applied_adjustment: float = 0.0


class Reservation(BaseModel):
    reservation_id: str | None = None
    reserved_cost: float = Field(ge=0)
    entries: tuple[ReservationEntry, ...]
    finalized: bool = False


REFUND_SCRIPT = """
if redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
local current = tonumber(redis.call('GET', KEYS[1]))
local delta = tonumber(ARGV[1])
if current and current + delta >= -0.000000000001 then
    redis.call('INCRBYFLOAT', KEYS[1], delta)
end
redis.call('SET', KEYS[2], '1', 'EX', 691200)
return 1
"""


def require_durable_budget(reservation: dict[str, JsonValue] | None) -> None:
    from litellm.proxy.proxy_server import spend_counter_cache

    if reservation and spend_counter_cache.redis_cache is None:
        raise RewriteError("Context IR budget reservations require a shared Redis spend cache", 503)


async def settle_reservation(task_id: str, reservation: dict[str, JsonValue] | None, actual: float) -> None:
    from litellm.proxy.proxy_server import spend_counter_cache

    if not reservation:
        return
    parsed = Reservation.model_validate(reservation)
    if parsed.finalized:
        return
    require_durable_budget(reservation)
    cache = spend_counter_cache.redis_cache
    if cache is None:
        raise RewriteError("Context IR budget cache unavailable", 503)
    script = cache.async_register_script(REFUND_SCRIPT)
    for entry in parsed.entries:
        from decimal import Decimal

        from litellm.proxy.video_endpoints.moderation_metering_runtime import protected_authority

        authority = await protected_authority(entry.counter_key)
        if authority is not None:
            reservation_id = entry.reservation_id or parsed.reservation_id
            if not reservation_id:
                raise RewriteError("Protected Context IR reservation identity missing", 503)
            await authority.mutate(
                "context-ir:" + task_id + ":" + entry.counter_key,
                (entry.counter_key,),
                kind="reconcile",
                amount=Decimal(str(actual)),
                reservation_id=reservation_id,
            )
            continue
        reserved = entry.reserved_cost if entry.reserved_cost is not None else parsed.reserved_cost
        delta = actual - reserved - entry.applied_adjustment
        if delta == 0:
            continue
        marker = hashlib.sha256((task_id + "\0" + entry.counter_key).encode()).hexdigest()
        await script(keys=[entry.counter_key, f"causyn:context-ir:budget:{marker}"], args=[delta])
        spend_counter_cache.in_memory_cache.delete_cache(key=entry.counter_key)


async def release_unaccepted_reservation(reservation: dict[str, JsonValue] | None) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import (
        release_budget_reservation,  # pyright: ignore[reportUnknownVariableType]  # legacy reservation API uses an unparameterized dict
    )

    await release_budget_reservation(reservation)
