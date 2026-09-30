import os
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.proxy.video_endpoints.moderation_metering import BillingBinding
from tests.test_litellm.proxy.video_endpoints import test_moderation_settlement_close_completion as cc
from tests.test_litellm.proxy.video_endpoints.test_moderation_metering import (  # noqa: F401
    prepare_meter,
    store,
)

needs_db = pytest.mark.skipif(
    not os.getenv("MODERATION_METERING_POSTGRES_URL") or not os.getenv("MODERATION_METERING_REDIS_URL"),
    reason="isolated PostgreSQL and Redis required",
)


async def statuses(meter):
    rows = await meter.db.query_raw(
        'SELECT phase,status,failure_reason FROM "LiteLLM_ModerationMeteringPhase" WHERE intent_id=$1 ORDER BY phase',
        "intent",
    )
    return {r["phase"]: (r["status"], r["failure_reason"]) for r in rows}


@needs_db
@pytest.mark.asyncio
async def test_void_unsent_voids_all_placeholders(store):  # noqa: F811
    meter, bound = store[0], cc.db_binding()
    await prepare_meter(meter, bound, {})
    await meter.void_unsent("intent")
    assert await statuses(meter) == {"submit": ("void", "not_sent"), "completion": ("void", "not_sent")}
    envelope = await meter.settlement(bound)
    assert envelope.receipts == () and envelope.complete is False
    assert await meter.run_once() is False


@needs_db
@pytest.mark.asyncio
async def test_void_unsent_is_noop_when_any_phase_has_native_id(store):  # noqa: F811
    meter, bound = store[0], cc.db_binding()
    await prepare_meter(meter, bound, {})
    await meter.persist(cc.submit_event(bound))
    await meter.void_unsent("intent")
    assert (await statuses(meter))["completion"][0] == "unknown"
    assert (await statuses(meter))["submit"][0] == "pending"


@needs_db
@pytest.mark.asyncio
async def test_void_placeholder_can_still_be_settled_by_a_later_submission(store):  # noqa: F811
    meter, bound = store[0], cc.db_binding()
    await prepare_meter(meter, bound, {})
    await meter.void_unsent("intent")
    await meter.persist(cc.submit_event(bound))
    assert (await statuses(meter))["submit"] == ("pending", None)


@needs_db
@pytest.mark.asyncio
async def test_settlement_complete_ignores_void_phase(store):  # noqa: F811
    meter, bound = store[0], cc.db_binding()
    await prepare_meter(meter, bound, {})
    await meter.persist(cc.submit_event(bound))
    await meter.db.execute_raw(
        "UPDATE \"LiteLLM_ModerationMeteringPhase\" SET status='void',failure_reason='not_sent' WHERE phase='completion'"
    )
    await meter.db.execute_raw(
        "UPDATE \"LiteLLM_ModerationMeteringPhase\" SET status='settled',receipt=payload WHERE phase='submit'"
    )
    envelope = await meter.settlement(bound)
    assert envelope.complete is True and len(envelope.receipts) == 1


@pytest.mark.asyncio
async def test_execute_not_sent_voids_placeholders(monkeypatch):
    from fastapi import FastAPI, HTTPException, Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.spend_tracking import budget_reservation
    from litellm.proxy.video_endpoints import moderation_metering_entry as entry
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    authority = AsyncMock()
    bound = cc.binding()
    monkeypatch.setattr(entry, "scope_for", AsyncMock(return_value=runtime.Scope(bound, "submit", authority)))
    monkeypatch.setattr(budget_reservation, "release_budget_reservation", AsyncMock())
    request = Request({"type": "http", "method": "POST", "path": "/v1/videos", "headers": [], "app": FastAPI()})
    auth = UserAPIKeyAuth(api_key="k", user_id="u", team_id="t", budget_reservation={"reservation_id": "r"})

    async def refused():
        raise HTTPException(429, "refused")

    with pytest.raises(HTTPException):
        await entry.execute(request, auth, "avideo_generation", refused())
    authority.void_unsent.assert_awaited_once_with("intent")

    async def ambiguous():
        raise TimeoutError("unknown")

    authority.void_unsent.reset_mock()
    with pytest.raises(TimeoutError):
        await entry.execute(request, auth, "avideo_generation", ambiguous())
    authority.void_unsent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("code,voided", [("provider_not_submitted", True), ("provider_submission_ambiguous", False)])
async def test_settlement_provider_not_submitted_voids(monkeypatch, code, voided):
    meter = cc.Meter(cc.binding(), {"submit": cc.submit_event(cc.binding(), ""), "completion": cc.placeholder(cc.binding())})
    meter.void_unsent = AsyncMock()

    async def failure(intent_id):
        return (code, 503, "m", "a")

    meter.submission_failure = failure
    response = await cc.settle(monkeypatch, meter)
    assert response.status_code == 409
    assert meter.void_unsent.await_count == (1 if voided else 0)
