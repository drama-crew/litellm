from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Awaitable
from datetime import datetime
from typing import Literal, TypeVar

from fastapi import Request
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, BillingWindow

ENTRY_KEY = "moderation_metering_admission"
T = TypeVar("T")


class Admission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    intent_id: str
    request_digest: str
    actor_user_id: str
    model: str
    generation_id: str | None = None
    phase: str = "submit"


class WindowRow(BaseModel):
    kind: Literal["key", "team"]
    identity: str
    duration: str
    period_start: datetime | None
    reset_at: datetime


def attest(request: Request, value: object, *, phase: str = "submit") -> None:
    proof = Admission.model_validate(value)
    request.scope[ENTRY_KEY] = proof.model_copy(update={"phase": phase})


async def scope_for(request: Request, auth: UserAPIKeyAuth, route: str) -> runtime.Scope | None:
    proof = request.scope.get(ENTRY_KEY)
    if not isinstance(proof, Admission):
        return None
    if proof.phase != "submit":
        scope = await runtime.continuation(proof.intent_id, proof.phase)
        if (scope.binding.request_digest, scope.binding.actor_user_id, scope.binding.generation_id) != (
            proof.request_digest,
            proof.actor_user_id,
            proof.generation_id,
        ):
            raise ValueError("metering continuation actor binding mismatch")
        return scope
    identity = auth.api_key or auth.token
    if not identity or not auth.user_id or not auth.team_id:
        raise ValueError("server authenticated billing identity required")
    fingerprint = identity if re.fullmatch(r"[a-f0-9]{64}", identity) else hashlib.sha256(identity.encode()).hexdigest()
    authority = runtime.store()
    reservation = runtime.Reservation.model_validate(auth.budget_reservation or {})
    windows = TypeAdapter(tuple[WindowRow, ...]).validate_python(
        await authority.db.query_raw(
            "SELECT 'key' AS kind,t.token AS identity,w.value->>'budget_duration' AS duration,"
            "c.period_start,(w.value->>'reset_at')::timestamptz AS reset_at "
            'FROM "LiteLLM_VerificationToken" t '
            "CROSS JOIN LATERAL jsonb_array_elements(t.budget_limits::jsonb) w(value) "
            'LEFT JOIN "LiteLLM_ModerationMeteringCounter" c ON '
            "c.counter_key='spend:key:'||t.token||':window:'||(w.value->>'budget_duration') "
            "WHERE t.token=$1 "
            "UNION ALL SELECT 'team',t.team_id,w.value->>'budget_duration',"
            "c.period_start,(w.value->>'reset_at')::timestamptz "
            'FROM "LiteLLM_TeamTable" t '
            "CROSS JOIN LATERAL jsonb_array_elements(t.budget_limits::jsonb) w(value) "
            'LEFT JOIN "LiteLLM_ModerationMeteringCounter" c ON '
            "c.counter_key='spend:team:'||t.team_id||':window:'||(w.value->>'budget_duration') "
            "WHERE t.team_id=$2",
            fingerprint,
            auth.team_id,
        )
    )
    from litellm.proxy.proxy_server import litellm_proxy_budget_name
    from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start

    binding = BillingBinding(
        intent_id=proof.intent_id,
        generation_id=proof.generation_id,
        request_digest=proof.request_digest,
        fingerprint=fingerprint,
        user_id=auth.user_id,
        actor_user_id=proof.actor_user_id,
        team_id=auth.team_id,
        organization_id=auth.org_id,
        global_user_id=litellm_proxy_budget_name if litellm.max_budget > 0 else None,
        model=proof.model,
        end_user_id=auth.end_user_id,
        tag_ids=tuple(
            sorted(
                set(
                    runtime.projection_tags(
                        TypeAdapter(dict[str, object])
                        .validate_python(getattr(auth, "metadata", None) or {})
                        .get("tags")
                    )
                )
                | {
                    entry.counter_key.removeprefix("spend:tag:")
                    for entry in reservation.entries
                    if entry.counter_key.startswith("spend:tag:")
                }
            )
        ),
        windows=tuple(
            BillingWindow(
                kind=w.kind,
                identity=w.identity,
                duration=w.duration,
                window_start=w.period_start
                or TypeAdapter(datetime).validate_python(
                    get_budget_window_start({"budget_duration": w.duration, "reset_at": w.reset_at})
                ),
                reset_at=w.reset_at,
            )
            for w in windows
        ),
        expected_phases=("completion",)
        if route == "context_ir"
        else ("submit",)
        if route == "avideo_create_character"
        else ("submit", "completion"),
    )
    await authority.prepare(
        binding,
        {e.counter_key: e.reserved_cost for e in reservation.entries},
        reservation_id=reservation.reservation_id,
    )
    return runtime.Scope(
        binding,
        binding.expected_phases[0],
        authority,
        await authority.phase(binding.intent_id, binding.expected_phases[0]),
    )


async def execute(
    request: Request,
    auth: UserAPIKeyAuth,
    route: str,
    call: Awaitable[T],
    request_data: dict[str, object] | None = None,
) -> T:
    from litellm.proxy.video_endpoints import moderation_bridge

    try:
        scope = await scope_for(request, auth, route)
    except BaseException:
        if inspect.iscoroutine(call):
            call.close()
        raise
    if scope is not None and request_data is not None:
        request_data[runtime.SCOPE_KEY] = runtime.Ownership.PROTECTED
    token = runtime.CONTEXT.set(scope)
    accepted = False
    try:
        result = await call
        accepted = True
        await moderation_bridge.capture(request, result)
        return result
    except Exception as exc:
        from litellm.proxy.video_endpoints.moderation_execution import failure_outcome

        if scope is not None and not accepted and failure_outcome(exc) in {"not_sent", "rejected"}:
            from litellm.proxy.spend_tracking import budget_reservation
            from litellm.proxy.video_endpoints.openapi_log_capture import raw_method

            await raw_method(budget_reservation, "release_budget_reservation")(
                budget_reservation=auth.budget_reservation
            )
        raise
    finally:
        runtime.CONTEXT.reset(token)
