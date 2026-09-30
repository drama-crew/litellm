"""A regular spend-log row under a phase request id must not wedge that phase's financial projection."""

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.proxy.video_endpoints.moderation_metering_projection import (
    BillingFacts,
    FinancialProjection,
    project,
)
from tests.test_litellm.proxy.video_endpoints.test_moderation_close_undelivered_real_store import (
    arranged,
    via_settlement,
)
from tests.test_litellm.proxy.video_endpoints.test_moderation_metering import (
    create_financial_projection_tables,
)
from tests.test_litellm.proxy.video_endpoints.test_moderation_metering import store  # noqa: F401

needs_db = pytest.mark.skipif(
    not os.getenv("MODERATION_METERING_POSTGRES_URL") or not os.getenv("MODERATION_METERING_REDIS_URL"),
    reason="isolated PostgreSQL and Redis required",
)
STAMP = datetime(2026, 9, 22, 22, 39, 9, tzinfo=timezone.utc)
RID = "public-video:intent:completion"


def projection(amount="0"):
    return FinancialProjection(
        request_id=RID,
        fingerprint="key",
        user_id="user",
        team_id="team",
        provider="causyn",
        deployment_id="dep",
        provider_task_id="task",
        model="causyn-1-1",
        amount=Decimal(amount),
        facts=BillingFacts(started_at=STAMP, ended_at=STAMP, route="avideo_status", raw_cost_credit=Decimal(amount)),
        intent_id="intent",
    )


async def regular_row(db, spend=0.0, metadata="{}", counted=True):
    """What LiteLLM's regular logging leaves behind for a status call, daily rows included."""
    await db.execute_raw(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id,call_type,api_key,spend,"startTime","endTime",model,"user",team_id,metadata) '
        "VALUES ($1,'',$2,$3,$4::timestamptz,$4::timestamptz,'causyn-1-1','user','team',$5::jsonb)",
        RID,
        "key",
        spend,
        STAMP,
        metadata,
    )
    if counted:
        for table, dim, ident in (("LiteLLM_DailyUserSpend", "user_id", "user"), ("LiteLLM_DailyTeamSpend", "team_id", "team")):
            await db.execute_raw(
                f'INSERT INTO "{table}" (id,{dim},date,api_key,model,model_group,custom_llm_provider,mcp_namespaced_tool_name,endpoint,spend,api_requests,successful_requests,prompt_tokens,completion_tokens) '
                "VALUES ($1,$2,'2026-09-22','key','causyn-1.1','causyn-1.1','causyn','','avideo_status',0,1,0,0,0)",
                "regular-" + table,
                ident,
            )


async def run_project(db, value):
    async with db.tx() as tx:
        await project(tx, value, protected=frozenset(), debit_unprotected=True)


async def spend_state(db):
    return {
        "log": (await db.query_raw('SELECT spend,call_type,metadata FROM "LiteLLM_SpendLogs"')),
        "team": (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"'))[0]["spend"],
        "daily": await db.query_raw(
            'SELECT spend,api_requests FROM "LiteLLM_DailyTeamSpend" ORDER BY endpoint,model'
        ),
    }


@needs_db
@pytest.mark.asyncio
async def test_a_zero_spend_regular_row_is_adopted_without_double_counting_spend(store):  # noqa: F811
    _, db, _, _ = store
    await create_financial_projection_tables(db)
    await regular_row(db)
    await run_project(db, projection("20"))
    logs = await db.query_raw('SELECT spend,call_type,model_id,metadata FROM "LiteLLM_SpendLogs"')
    assert len(logs) == 1
    assert (logs[0]["spend"], logs[0]["call_type"], logs[0]["model_id"]) == (20.0, "avideo_status", "dep")
    metadata = logs[0]["metadata"] if isinstance(logs[0]["metadata"], dict) else json.loads(logs[0]["metadata"])
    assert metadata["moderation_intent_id"] == "intent" and "billing_facts" in metadata
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 20.0}]
    daily = await db.query_raw('SELECT model,spend,api_requests FROM "LiteLLM_DailyTeamSpend" ORDER BY model')
    # the spend is counted exactly once, and the call is still one request per (regular row + adopted projection)
    assert sum(row["spend"] for row in daily) == 20.0
    assert sum(row["api_requests"] for row in daily) == 1


@needs_db
@pytest.mark.asyncio
async def test_b_a_row_that_already_holds_spend_still_raises_and_rolls_back(store):  # noqa: F811
    _, db, _, _ = store
    await create_financial_projection_tables(db)
    await regular_row(db, spend=5.0, counted=False)
    with pytest.raises(ValueError, match="already exists"):
        await run_project(db, projection("20"))
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 0.0}]
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_SpendLogs"')) == [{"spend": 5.0}]


@needs_db
@pytest.mark.asyncio
async def test_b_a_zero_spend_row_carrying_another_projection_still_raises(store):  # noqa: F811
    _, db, _, _ = store
    await create_financial_projection_tables(db)
    other = json.dumps({"moderation_intent_id": "someone-else", "billing_facts": {}})
    await regular_row(db, metadata=other, counted=False)
    with pytest.raises(ValueError, match="already exists"):
        await run_project(db, projection("0"))
    # same projection, but the earlier one was zero and this one carries an amount
    await db.execute_raw('DELETE FROM "LiteLLM_SpendLogs"')
    await run_project(db, projection("0"))
    with pytest.raises(ValueError, match="already exists"):
        await run_project(db, projection("20"))
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 0.0}]


@needs_db
@pytest.mark.asyncio
async def test_c_rerunning_the_same_zero_projection_is_idempotent(store):  # noqa: F811
    _, db, _, _ = store
    await create_financial_projection_tables(db)
    await run_project(db, projection("0"))
    once = await spend_state(db)
    await run_project(db, projection("0"))
    await run_project(db, projection("0"))
    assert await spend_state(db) == once
    assert once["daily"][0]["api_requests"] == 1


def test_d_status_poll_call_id_never_equals_the_phase_request_id():
    import asyncio

    seen = []

    async def go():
        from litellm.proxy.video_endpoints import moderation_metering_entry as entry

        async def video_status(native_id, request, response, auth):
            seen.append(dict(request.scope["headers"])[b"x-litellm-call-id"].decode())
            return "ok"

        import litellm.proxy.video_endpoints.endpoints as endpoints

        original = endpoints.video_status
        endpoints.video_status = video_status
        original_attest = entry.attest
        entry.attest = lambda *args, **kwargs: None
        try:
            request = SimpleNamespace(headers={}, scope={"headers": []}, app=None)
            monkey_request = execution.execution_request
            execution.execution_request = lambda *a, **k: SimpleNamespace(scope={"headers": []})
            try:
                for _ in range(2):
                    await execution.completion_status_read(
                        request, None, native_id="n", intent_id="intent", metering={}, read_ticket=None, context=None
                    )
            finally:
                execution.execution_request = monkey_request
        finally:
            endpoints.video_status = original
            entry.attest = original_attest

    asyncio.run(go())
    assert len(seen) == 2 and seen[0] != seen[1]
    for call_id in seen:
        assert call_id != RID and call_id.startswith(RID + ":poll:")


@needs_db
@pytest.mark.asyncio
async def test_e_run_once_settles_a_closed_undelivered_completion_over_a_regular_row(store):  # noqa: F811
    meter, db, _, _ = store
    await create_financial_projection_tables(db)
    b = await arranged(meter)
    await via_settlement(meter, b)
    await regular_row(db)
    for _ in range(4):
        await meter.run_once()
    logs = await db.query_raw('SELECT request_id,spend,call_type FROM "LiteLLM_SpendLogs" ORDER BY request_id')
    assert [(r["request_id"], r["spend"], r["call_type"]) for r in logs] == [
        (RID, 0.0, "avideo_status"),
        ("public-video:intent:submit", 0.0, "avideo_generation"),
    ]
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 0.0}]
    assert (await meter.settlement(b)).complete


@pytest.mark.asyncio
async def test_recovery_consumer_logs_the_failure_type_message_and_traceback(caplog):
    from litellm.proxy.video_endpoints.moderation_metering_runtime import RecoveryConsumer

    class Broken:
        async def recover_pending(self, last):
            return last

        async def run_once(self):
            raise ValueError("financial projection request identity already exists outside its receipt")

    consumer = RecoveryConsumer(Broken(), interval=0.01)
    with caplog.at_level(logging.WARNING):
        await consumer.start()
        import asyncio

        await asyncio.sleep(0.15)
        await consumer.stop()
    records = [r for r in caplog.records if "durable recovery pending" in r.getMessage()]
    assert records and "ValueError" in records[0].getMessage() and "already exists" in records[0].getMessage()
    assert records[0].exc_info and records[0].exc_info[0] is ValueError
    assert len(records) == 1  # rate limited across repeated identical ticks
