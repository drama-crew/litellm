import time
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import jwt
import pytest
from fastapi import FastAPI

from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.proxy.video_endpoints.moderation_metering import (
    BillingBinding,
    PhaseEvent,
    SettlementEnvelope,
)
from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider

SECRET = "synthetic-settlement-secret-long-enough-32-bytes"
NOW = datetime.now(timezone.utc)
NATIVE = encode_video_id_with_provider("task-1", "libtv", "dep-1")


def ticket():
    return jwt.encode(
        {
            "intent_id": "intent",
            "request_digest": "a" * 64,
            "input_digest": "b" * 64,
            "model": "libtv-video",
            "policy_version": "v1",
            "policy_digest": "c" * 64,
            "purpose": "settlement",
            "aud": "moderation-fork",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        SECRET,
        algorithm="HS256",
    )


def binding(phases=("submit", "completion")):
    return BillingBinding(
        intent_id="intent",
        request_digest="a" * 64,
        fingerprint="key",
        user_id="user",
        actor_user_id="actor",
        team_id="team",
        model="libtv-video",
        expected_phases=phases,
    )


def submit_event(bound, native_id=NATIVE):
    return PhaseEvent(
        binding=bound,
        request_id="public-video:intent:submit",
        phase="submit",
        provider="libtv" if native_id else "",
        deployment_id="dep-1" if native_id else "",
        native_id=native_id,
        provider_task_id="task-1" if native_id else "",
        amount=Decimal(0) if native_id else None,
        finalized=bool(native_id),
        facts=BillingFacts(started_at=NOW, ended_at=NOW, route="avideo_generation") if native_id else None,
    )


def placeholder(bound):
    return PhaseEvent(
        binding=bound,
        request_id="public-video:intent:completion",
        phase="completion",
        provider="",
        deployment_id="",
        native_id="",
        provider_task_id="",
    )


def finalized_completion(bound, amount):
    from litellm.proxy.video_endpoints.moderation_metering_projection import actual_debit

    raw = Decimal(amount)
    return PhaseEvent(
        binding=bound,
        request_id="public-video:intent:completion",
        phase="completion",
        provider="libtv",
        deployment_id="dep-1",
        native_id=NATIVE,
        provider_task_id="task-1",
        amount=actual_debit(raw),
        finalized=True,
        facts=BillingFacts(started_at=NOW, ended_at=NOW, route="avideo_status", raw_cost_credit=raw),
    )


class Meter:
    def __init__(self, bound, phases):
        self.bound = bound
        self.phases = dict(phases)
        self.persisted = []

    async def binding(self, intent_id):
        return self.bound

    async def manual_settlement_reason(self, intent_id):
        return None

    async def submission_failure(self, intent_id):
        return None

    async def phase(self, intent_id, phase):
        return self.phases.get(phase)

    async def persist(self, event):
        self.persisted.append(event)
        self.phases[event.phase] = event

    async def settlement(self, bound):
        receipts = tuple(self.phases[p] for p in bound.expected_phases if p in self.phases)
        return SettlementEnvelope(
            binding=bound,
            receipts=receipts,
            complete=len(receipts) == len(bound.expected_phases) and all(r.finalized for r in receipts),
            total_actual=None,
        )


async def settle(monkeypatch, meter, close=None):
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setattr(runtime, "store", lambda: meter)

    async def platform(request, method, path, payload):
        return dict(binding=None, native_id="native", events={}, recovery_binding={"intent_id": "intent"})

    monkeypatch.setattr(execution.bridge, "platform", platform)
    app = FastAPI()
    app.include_router(execution.router)
    body = {"ticket": ticket(), **({"close_completion": close} if close else {})}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fork") as client:
        return await client.post(
            "/internal/moderation/settlement", headers={"Authorization": "Bearer " + SECRET}, json=body
        )


def libtv_meter(completion=None):
    bound = binding()
    return Meter(bound, {"submit": submit_event(bound), "completion": completion or placeholder(bound)})


@pytest.mark.asyncio
async def test_undelivered_zeroes_placeholder_under_submit_identity(monkeypatch):
    meter = libtv_meter()
    response = await settle(monkeypatch, meter, "undelivered")
    assert response.status_code == 200
    (event,) = meter.persisted
    assert (event.phase, event.provider, event.deployment_id, event.native_id, event.provider_task_id) == (
        "completion",
        "libtv",
        "dep-1",
        NATIVE,
        "task-1",
    )
    assert event.request_id == "public-video:intent:completion"
    assert event.amount == Decimal(0) and event.finalized is True
    assert event.facts.route == "avideo_status" and event.facts.raw_cost_credit == Decimal(0)
    assert response.json()["complete"] is True


@pytest.mark.asyncio
async def test_undelivered_repeat_is_idempotent(monkeypatch):
    meter = libtv_meter()
    await settle(monkeypatch, meter, "undelivered")
    first = meter.phases["completion"]
    await settle(monkeypatch, meter, "undelivered")
    assert len(meter.persisted) == 1
    assert meter.phases["completion"] == first


@pytest.mark.asyncio
async def test_undelivered_never_overwrites_finalized_completion(monkeypatch):
    bound = binding()
    real = finalized_completion(bound, "25")
    meter = libtv_meter(real)
    response = await settle(monkeypatch, meter, "undelivered")
    assert response.status_code == 200
    assert meter.persisted == []
    assert meter.phases["completion"] == real


@pytest.mark.asyncio
async def test_ignored_when_submit_has_no_native_id(monkeypatch):
    bound = binding()
    meter = Meter(bound, {"submit": submit_event(bound, ""), "completion": placeholder(bound)})
    response = await settle(monkeypatch, meter, "undelivered")
    assert response.status_code == 200
    assert meter.persisted == []


@pytest.mark.asyncio
async def test_ignored_for_completion_only_binding(monkeypatch):
    bound = binding(("completion",))
    meter = Meter(bound, {"completion": placeholder(bound)})
    response = await settle(monkeypatch, meter, "undelivered")
    assert response.status_code == 200
    assert meter.persisted == []


@pytest.mark.asyncio
async def test_unknown_close_value_is_rejected(monkeypatch):
    response = await settle(monkeypatch, libtv_meter(), "bogus")
    assert response.status_code == 422


def patch_status(monkeypatch, status):
    from litellm.proxy.video_endpoints import endpoints

    calls = []

    async def video_status(video_id, request, fastapi_response, user_api_key_dict):
        calls.append((video_id, request.scope))
        return VideoObject(id=video_id, object="video", status=status)

    monkeypatch.setattr(endpoints, "video_status", video_status)
    return calls


@pytest.mark.asyncio
async def test_delivered_reads_completion_status_once_under_completion_scope(monkeypatch):
    calls = patch_status(monkeypatch, "in_progress")
    meter = libtv_meter()
    response = await settle(monkeypatch, meter, "delivered")
    assert response.status_code == 200
    assert len(calls) == 1
    video_id, scope = calls[0]
    assert video_id == NATIVE
    admission = scope["moderation_metering_admission"]
    assert (admission.intent_id, admission.phase, admission.actor_user_id) == ("intent", "completion", "actor")
    (call_id,) = [value for name, value in scope["headers"] if name == b"x-litellm-call-id"]
    # a poll id, never the phase request id (that one belongs to the phase's financial projection row)
    assert call_id.startswith(b"public-video:intent:completion:poll:")
    assert meter.persisted == []


@pytest.mark.asyncio
async def test_delivered_skipped_when_completion_already_finalized(monkeypatch):
    bound = binding()
    calls = patch_status(monkeypatch, "completed")
    meter = libtv_meter(finalized_completion(bound, "25"))
    await settle(monkeypatch, meter, "delivered")
    assert calls == []


@pytest.mark.asyncio
async def test_undelivered_payload_is_deterministic(monkeypatch):
    first, second = libtv_meter(), libtv_meter()
    await settle(monkeypatch, first, "undelivered")
    await settle(monkeypatch, second, "undelivered")
    assert first.persisted[0].model_dump_json() == second.persisted[0].model_dump_json()
    assert first.persisted[0].facts.started_at == first.phases["submit"].facts.started_at


@pytest.mark.asyncio
async def test_undelivered_lost_race_is_a_noop(monkeypatch):
    meter = libtv_meter()
    real = finalized_completion(meter.bound, "25")

    async def persist(event):
        meter.phases["completion"] = real
        raise ValueError("moderation billing payload replay conflict")

    meter.persist = persist
    response = await settle(monkeypatch, meter, "undelivered")
    assert response.status_code == 200
    assert meter.phases["completion"] == real


@pytest.mark.asyncio
async def test_undelivered_conflict_without_finalized_completion_still_raises(monkeypatch):
    meter = libtv_meter()

    async def persist(event):
        raise ValueError("boom")

    meter.persist = persist
    with pytest.raises(ValueError):
        await settle(monkeypatch, meter, "undelivered")


@pytest.mark.asyncio
async def test_no_close_when_parked_for_manual_settlement(monkeypatch):
    meter = libtv_meter()

    async def manual(intent_id):
        return "key_identity_mismatch: x"

    meter.manual_settlement_reason = manual
    response = await settle(monkeypatch, meter, "undelivered")
    assert response.status_code == 409
    assert meter.persisted == []


@pytest.mark.asyncio
async def test_delivered_read_failure_returns_envelope(monkeypatch):
    from litellm.proxy.video_endpoints import endpoints

    async def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(endpoints, "video_status", boom)
    meter = libtv_meter()
    response = await settle(monkeypatch, meter, "delivered")
    assert response.status_code == 200
    assert response.json()["complete"] is False


def libtv_scope(authority):
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    bound = binding()
    facts = BillingFacts(
        started_at=NOW,
        ended_at=NOW,
        route="avideo_generation",
        duration_seconds=Decimal(5),
        resolution="720p",
        pricing=(("output_cost_per_second_720p", Decimal("2")),),
        raw_cost_credit=Decimal(0),
    )
    submit = submit_event(bound).model_copy(update={"facts": facts})
    return runtime.Scope(bound, "completion", authority, placeholder(bound), submit)


@pytest.mark.asyncio
async def test_delivered_libtv_completed_persists_provider_correct_amount(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.llms.libtv import billing_outbox
    from litellm.llms.libtv.handler import LibTVLLM
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    authority = AsyncMock()
    scope = libtv_scope(authority)
    sent = []
    monkeypatch.setattr(billing_outbox, "enqueue_causyn_billing", AsyncMock(side_effect=lambda r, e: sent.append(e)))
    monkeypatch.setattr("litellm.llms.causyn.handler._default_redis_factory", lambda: None)
    vo = VideoObject(id=NATIVE, object="video", status="completed")
    token = runtime.CONTEXT.set(scope)
    try:
        await LibTVLLM()._bill_protected_video(vo, "task-1", scope)
        logging_obj = MagicMock()
        setattr(logging_obj, runtime.SCOPE_KEY, scope)
        setattr(logging_obj, runtime.CALL_TYPE_KEY, "avideo_status")
        logging_obj.model_call_details = {}
        await runtime._handoff(logging_obj, vo, NOW, NOW)
    finally:
        runtime.CONTEXT.reset(token)
    (event,) = [c.args[0] for c in authority.persist.await_args_list]
    assert event.phase == "completion" and event.finalized is True
    assert event.native_id == NATIVE and event.amount > 0
    assert event.facts.raw_cost_credit == Decimal("10")  # 5s x rate 2
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_libtv_failed_completion_persists_zero():
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    authority = AsyncMock()
    scope = libtv_scope(authority)
    logging_obj = MagicMock()
    setattr(logging_obj, runtime.SCOPE_KEY, scope)
    setattr(logging_obj, runtime.CALL_TYPE_KEY, "avideo_status")
    logging_obj.model_call_details = {}
    logging_obj.custom_llm_provider = "libtv"
    await runtime._handoff(logging_obj, VideoObject(id=NATIVE, object="video", status="failed"), NOW, NOW)
    (event,) = [c.args[0] for c in authority.persist.await_args_list]
    assert event.amount == Decimal(0) and event.finalized is True


# --- real MeteringStore (needs the throwaway PostgreSQL/Redis from test-env.sh) ---
import os  # noqa: E402

from tests.test_litellm.proxy.video_endpoints.test_moderation_metering import (  # noqa: E402
    prepare_meter,
)
from tests.test_litellm.proxy.video_endpoints.test_moderation_metering import store  # noqa: E402,F401

needs_db = pytest.mark.skipif(
    not os.getenv("MODERATION_METERING_POSTGRES_URL") or not os.getenv("MODERATION_METERING_REDIS_URL"),
    reason="isolated PostgreSQL and Redis required",
)


def db_binding():
    return BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="video",
        expected_phases=("submit", "completion"),
    )


@needs_db
@pytest.mark.asyncio
async def test_real_store_undelivered_closes_placeholder_and_keeps_real_charge(store):  # noqa: F811
    meter = store[0]
    bound = db_binding()
    await prepare_meter(meter, bound, {})
    await meter.persist(submit_event(bound))
    await execution.close_completion(None, meter, bound, "undelivered")
    closed = await meter.phase("intent", "completion")
    assert (closed.amount, closed.finalized, closed.native_id) == (Decimal(0), True, NATIVE)
    await execution.close_completion(None, meter, bound, "undelivered")
    assert await meter.phase("intent", "completion") == closed


@needs_db
@pytest.mark.asyncio
async def test_real_store_finalized_charge_is_never_zeroed(store):  # noqa: F811
    meter = store[0]
    bound = db_binding()
    await prepare_meter(meter, bound, {})
    await meter.persist(submit_event(bound))
    real = finalized_completion(bound, "25")
    await meter.persist(real)
    await execution.close_completion(None, meter, bound, "undelivered")
    assert await meter.phase("intent", "completion") == real
