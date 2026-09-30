"""Undelivered close against the real MeteringStore, from every state the completion row can be in."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import litellm
import pytest

from litellm.llms.custom_llm import ProviderTaskNotFound
from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent
from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts, actual_debit
from litellm.types.videos.utils import encode_video_id_with_provider
from tests.test_litellm.proxy.video_endpoints.test_moderation_metering import prepare_meter
from tests.test_litellm.proxy.video_endpoints.test_moderation_metering import store  # noqa: F401

needs_db = pytest.mark.skipif(
    not os.getenv("MODERATION_METERING_POSTGRES_URL") or not os.getenv("MODERATION_METERING_REDIS_URL"),
    reason="isolated PostgreSQL and Redis required",
)
NATIVE = encode_video_id_with_provider("task-1", "causyn", "dep-1")
LONG_AGO = datetime.now(timezone.utc) - timedelta(hours=30)


def bound():
    return BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        actor_user_id="actor",
        model="video",
        expected_phases=("submit", "completion"),
    )


def submit_event(b):
    return PhaseEvent(
        binding=b,
        request_id="public-video:intent:submit",
        phase="submit",
        provider="causyn",
        deployment_id="dep-1",
        native_id=NATIVE,
        provider_task_id="task-1",
        amount=Decimal(0),
        finalized=True,
        facts=BillingFacts(started_at=LONG_AGO, ended_at=LONG_AGO, route="avideo_generation"),
    )


def running_completion(b):
    later = LONG_AGO + timedelta(minutes=5)
    return submit_event(b).model_copy(
        update={
            "request_id": "public-video:intent:completion",
            "phase": "completion",
            "amount": None,
            "finalized": False,
            "facts": BillingFacts(started_at=later, ended_at=later, route="avideo_status"),
        }
    )


def finalized_completion(b):
    return running_completion(b).model_copy(
        update={
            "amount": actual_debit(Decimal(25)),
            "finalized": True,
            "facts": BillingFacts(
                started_at=LONG_AGO, ended_at=LONG_AGO, route="avideo_status", raw_cost_credit=Decimal(25)
            ),
        }
    )


async def via_settlement(meter, b):
    await execution.close_completion(None, meter, b, "undelivered")


async def via_dead_upstream(meter, b):
    error = litellm.NotFoundError(message="gone", model="m", llm_provider="causyn")
    error.__cause__ = ProviderTaskNotFound("causyn video was not found")
    await execution._close_dead_upstream(meter, b.intent_id, NATIVE, error)


PATHS = pytest.mark.parametrize("close", [via_settlement, via_dead_upstream])


async def arranged(meter, completion=None):
    b = bound()
    await prepare_meter(meter, b, {})
    await meter.persist(submit_event(b))
    if completion is not None:
        await meter.persist(completion(b))
    return b


@needs_db
@PATHS
@pytest.mark.asyncio
async def test_placeholder_closes_at_zero(store, close):  # noqa: F811
    meter = store[0]
    b = await arranged(meter)
    await close(meter, b)
    done = await meter.phase("intent", "completion")
    assert (done.amount, done.finalized, done.native_id) == (Decimal(0), True, NATIVE)


@needs_db
@PATHS
@pytest.mark.asyncio
async def test_unfinalized_running_event_with_facts_closes_at_zero_same_identity(store, close):  # noqa: F811
    meter = store[0]
    b = await arranged(meter, running_completion)
    before = await meter.phase("intent", "completion")
    assert before.finalized is False and before.facts is not None
    await close(meter, b)
    done = await meter.phase("intent", "completion")
    assert (done.amount, done.finalized) == (Decimal(0), True)
    assert (done.provider, done.deployment_id, done.native_id, done.provider_task_id) == (
        before.provider,
        before.deployment_id,
        before.native_id,
        before.provider_task_id,
    )
    assert done.facts.raw_cost_credit == Decimal(0)
    assert done.facts.started_at == before.facts.started_at
    settled = await meter.settlement(b)
    assert settled.receipts is not None


@needs_db
@PATHS
@pytest.mark.asyncio
async def test_finalized_completion_is_a_noop(store, close):  # noqa: F811
    meter = store[0]
    b = await arranged(meter, finalized_completion)
    real = await meter.phase("intent", "completion")
    await close(meter, b)
    assert await meter.phase("intent", "completion") == real


@needs_db
@PATHS
@pytest.mark.asyncio
@pytest.mark.parametrize("completion", [None, running_completion])
async def test_repeat_close_is_idempotent(store, close, completion):  # noqa: F811
    meter = store[0]
    b = await arranged(meter, completion)
    await close(meter, b)
    first = await meter.phase("intent", "completion")
    await close(meter, b)
    assert await meter.phase("intent", "completion") == first


@needs_db
@pytest.mark.asyncio
async def test_real_charge_after_undelivered_close_is_permanent_not_retried(store, monkeypatch):  # noqa: F811
    from litellm.llms.libtv.billing_outbox import CausynBillingEvent, PermanentBillingFailure
    from litellm.proxy.video_endpoints import moderation_metering_outbox as outbox
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    meter = store[0]
    b = await arranged(meter)
    await via_settlement(meter, b)
    charge = finalized_completion(b)
    monkeypatch.setattr(runtime, "store", lambda: meter)
    event = CausynBillingEvent(
        provider_task_id="task-1",
        response_cost=25,
        team_id="team",
        user_id="user",
        api_key="key",
        provider="causyn",
        deployment_id="dep-1",
        metering_event_json=charge.model_dump_json(),
    )
    with pytest.raises(PermanentBillingFailure, match="already finalized"):
        await outbox.settle_outbox(event)
