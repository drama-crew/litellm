import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest

import litellm.llms.libtv.billing_outbox as billing_outbox
from litellm.llms.libtv.billing_outbox import (
    BILLING_STREAM_KEY,
    CAUSYN_BILLING_MARKER_PREFIX,
    CAUSYN_BILLING_STREAM_KEY,
    CausynBillingEvent,
    ImageBillingEvent,
    LibTVBillingReconciler,
    _event_time,
    _ENQUEUE_SCRIPT,
    enqueue_causyn_billing,
)
from litellm.llms.libtv.receipts import LibTVReceiptStore, _TRANSITION_SCRIPT, request_fingerprint


class RecordingRedis:
    def __init__(self):
        self.eval_calls = []

    async def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        return ["ok", args[3]]


class EnqueueRedis:
    def __init__(self, outcomes=("enqueued",)):
        self.outcomes = iter(outcomes)
        self.calls = []

    async def eval(self, script, numkeys, *args):
        self.calls.append((script, numkeys, args))
        return [next(self.outcomes), args[3]]


@pytest.mark.asyncio
async def test_terminal_receipt_transition_appends_one_billing_event():
    redis = RecordingRedis()
    store = LibTVReceiptStore(redis)
    receipt = type(
        "Receipt",
        (),
        {
            "team_id": "team-1",
            "model": "topaz-image-upscaler",
            "request_id": "request-1",
            "fingerprint": request_fingerprint({"source_sha256": "a" * 64}, "topaz-image-upscaler"),
            "submission_state": "submitting",
            "deployment_id": "dep-1",
            "provider_task_id": None,
            "resume_token": None,
            "provider_code": None,
            "message": None,
        },
    )()
    event = ImageBillingEvent(
        deployment_id="dep-1",
        provider_task_id="task-1",
        response_cost=1.25,
        team_id="team-1",
    )

    await store.transition(receipt, "receipt-key", "submitted", billing_event=event)

    assert len(redis.eval_calls) == 1
    script, numkeys, args = redis.eval_calls[0]
    assert numkeys == 5
    assert args[3] == BILLING_STREAM_KEY
    assert args[4] == f"{BILLING_STREAM_KEY}:delivered:{event.event_id}"
    assert json.loads(args[-1])["provider_task_id"] == "task-1"
    assert script.count("XADD") == 1


def test_terminal_transition_script_deduplicates_billing_event_on_repeated_poll():
    assert "current['billing_event_id']" in _TRANSITION_SCRIPT


def test_terminal_transition_enqueues_before_recording_delivery_marker():
    assert _TRANSITION_SCRIPT.index("redis.call('XADD'") < _TRANSITION_SCRIPT.index("redis.call('SET', KEYS[3]")


def test_billing_event_id_and_downstream_request_id_are_stable_across_retries():
    first = ImageBillingEvent(deployment_id="dep-1", provider_task_id="task-1", response_cost=1.25)
    retry = ImageBillingEvent(deployment_id="dep-1", provider_task_id="task-1", response_cost=1.25)

    assert first.event_id == retry.event_id
    assert first.request_id == retry.request_id


@pytest.mark.asyncio
async def test_causyn_enqueue_is_atomic_and_repeated_terminal_poll_is_idempotent():
    event = CausynBillingEvent(provider_task_id="task-1", response_cost=1.25)
    redis = EnqueueRedis(("enqueued", "existing"))

    assert await enqueue_causyn_billing(redis, event)
    assert await enqueue_causyn_billing(redis, event)

    assert len(redis.calls) == 2
    script, numkeys, args = redis.calls[0]
    assert numkeys == 2
    assert args[:2] == (
        CAUSYN_BILLING_STREAM_KEY,
        f"{CAUSYN_BILLING_MARKER_PREFIX}task-1",
    )
    assert script == _ENQUEUE_SCRIPT
    assert script.index("XADD") < script.index("SET")
    assert event.request_id == "causyn:task-1"
    assert CausynBillingEvent.from_dict(__import__("json").loads(args[2])).event_id == event.event_id


@pytest.mark.asyncio
async def test_billing_event_persists_narrow_upscale_attribution_in_spend_logs():
    event = ImageBillingEvent(
        deployment_id="dep-1",
        provider_task_id="task-1",
        response_cost=1.25,
        team_id="team-1",
        user_id="billing-user-1",
        organization_id="org-1",
        api_key="key-1",
        scale=4,
        project_id="project-1",
        artifact_id="artifact-1",
        attribution_user_id="owner-1",
    )
    transaction = FakeTransaction(inserted=True)
    reconciler = LibTVBillingReconciler(FakeStreamRedis([]), FakePrisma(transaction))

    await reconciler._reconcile_event(event)

    metadata = json.loads(transaction.sql[0][1][7])
    assert metadata == {
        "libtv_billing_key": "libtv-image:dep-1:task-1",
        "scale": 4,
        "project_id": "project-1",
        "artifact_id": "artifact-1",
        "user_id": "owner-1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "expected_call_type", "expected_time"),
    (
        (
            ImageBillingEvent(
                deployment_id="dep-1",
                provider_task_id="task-1",
                response_cost=1.25,
                occurred_at="2026-08-27T04:00:00+08:00",
            ),
            "image_upscale",
            datetime(2026, 8, 26, 20, 0),
        ),
        (
            CausynBillingEvent(
                provider_task_id="task-1",
                response_cost=1.25,
                occurred_at="2026-08-27T04:00:00-05:00",
            ),
            "video_generation",
            datetime(2026, 8, 27, 9, 0),
        ),
    ),
)
async def test_billing_event_uses_utc_timestamp_cast_for_image_and_causyn(event, expected_call_type, expected_time):
    transaction = FakeTransaction(inserted=True)
    reconciler = LibTVBillingReconciler(FakeStreamRedis([]), FakePrisma(transaction))

    await reconciler._reconcile_event(event)

    query, args = transaction.sql[0]
    assert query.count("$5::timestamp") == 2
    assert "$8::jsonb" in query
    assert args[1] == expected_call_type
    assert args[4] == expected_time


def test_event_time_assumes_utc_for_naive_iso_timestamp():
    assert _event_time("2026-08-27T04:00:00") == datetime(2026, 8, 27, 4, 0)


def test_event_time_uses_deterministic_utc_now_for_invalid_iso_timestamp(monkeypatch):
    class FrozenDatetime:
        @classmethod
        def fromisoformat(cls, value):
            raise ValueError(value)

        @classmethod
        def now(cls, tz):
            assert tz is timezone.utc
            return datetime(2026, 8, 27, 12, 34, 56, 789, tzinfo=tz)

    monkeypatch.setattr(billing_outbox, "datetime", FrozenDatetime)

    assert _event_time("not-an-iso-timestamp") == datetime(2026, 8, 27, 12, 34, 56, 789)


class FakeTransaction:
    def __init__(self, inserted=True):
        self.inserted = inserted
        self.sql = []

    async def execute_raw(self, query, *args):
        self.sql.append((query, args))
        return 1 if self.inserted and len(self.sql) == 1 else 0


class FakeDB:
    def __init__(self, transaction):
        self.transaction = transaction

    def tx(self, **kwargs):
        transaction = self.transaction

        class Context:
            async def __aenter__(self):
                return transaction

            async def __aexit__(self, *exc):
                return False

        return Context()


class FakePrisma:
    def __init__(self, transaction):
        self.db = FakeDB(transaction)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("LIBTV_BILLING_POSTGRES_URL"),
    reason="Set LIBTV_BILLING_POSTGRES_URL to an isolated PostgreSQL 17 database",
)
@pytest.mark.parametrize("task_type", ["video_generation", "h3_context_ir"])
async def test_billing_event_executes_against_isolated_postgres(task_type):
    from prisma import Prisma

    database_url = os.environ["LIBTV_BILLING_POSTGRES_URL"]
    if os.getenv("LIBTV_BILLING_POSTGRES_ISOLATED") != "1":
        pytest.fail("Set LIBTV_BILLING_POSTGRES_ISOLATED=1 to confirm this is an isolated database")

    schema_name = f"libtv_billing_test_{uuid.uuid4().hex}"
    url_parts = urlsplit(database_url)
    query = [
        (key, value)
        for key, value in parse_qsl(url_parts.query, keep_blank_values=True)
        if key not in {"schema", "connection_limit"}
    ] + [("schema", schema_name), ("connection_limit", "1")]
    schema_url = urlunsplit(url_parts._replace(query=urlencode(query)))
    control_db = Prisma(datasource={"url": database_url})
    schema_db = Prisma(datasource={"url": schema_url})
    schema_created = False
    image_event = ImageBillingEvent(
        deployment_id=f"pg-{schema_name}",
        provider_task_id="image-task",
        response_cost=1.25,
        team_id=f"billing-outbox-test-{schema_name}",
        occurred_at="2026-08-27T04:00:00+08:00",
    )
    causyn_cost = 4.0 if task_type == "h3_context_ir" else 2.5
    causyn_event = CausynBillingEvent(
        task_type=task_type,
        model="causyn-h3-context-ir" if task_type == "h3_context_ir" else "causyn-1.0",
        provider_task_id=f"pg-{schema_name}",
        response_cost=causyn_cost,
        team_id=f"billing-outbox-test-{schema_name}",
        occurred_at="2026-08-27T04:00:00-05:00",
    )
    team_id = f"billing-outbox-test-{schema_name}"
    try:
        await control_db.connect()
        await control_db.execute_raw(f'CREATE SCHEMA "{schema_name}"')
        schema_created = True
        await schema_db.connect()
        current_schema_rows = await schema_db.query_raw("SELECT current_schema() AS current_schema")
        assert current_schema_rows == [{"current_schema": schema_name}]
        await schema_db.execute_raw(
            'CREATE TABLE "LiteLLM_SpendLogs" ('
            "request_id TEXT PRIMARY KEY, call_type TEXT NOT NULL, api_key TEXT NOT NULL, "
            "spend DOUBLE PRECISION NOT NULL, total_tokens INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL, "
            'completion_tokens INTEGER NOT NULL, "startTime" TIMESTAMP WITHOUT TIME ZONE NOT NULL, '
            '"endTime" TIMESTAMP WITHOUT TIME ZONE NOT NULL, model TEXT NOT NULL, "user" TEXT NOT NULL, '
            "metadata JSONB NOT NULL, team_id TEXT, organization_id TEXT)"
        )
        await schema_db.execute_raw(
            'CREATE TABLE "LiteLLM_TeamTable" (team_id TEXT PRIMARY KEY, spend DOUBLE PRECISION NOT NULL DEFAULT 0)'
        )
        await schema_db.execute_raw('INSERT INTO "LiteLLM_TeamTable" (team_id, spend) VALUES ($1, 0)', team_id)
        reconciler = LibTVBillingReconciler(FakeStreamRedis([]), schema_db)
        await reconciler._reconcile_event(image_event)
        await reconciler._reconcile_event(image_event)
        await reconciler._reconcile_event(causyn_event)
        await reconciler._reconcile_event(causyn_event)

        rows = await schema_db.query_raw(
            "SELECT call_type, spend, "
            'to_char("startTime", \'YYYY-MM-DD"T"HH24:MI:SS.US\') AS start_time, '
            'to_char("endTime", \'YYYY-MM-DD"T"HH24:MI:SS.US\') AS end_time '
            'FROM "LiteLLM_SpendLogs" ORDER BY call_type'
        )
        assert rows == sorted(
            [
                {
                    "call_type": "image_upscale",
                    "spend": 1.25,
                    "start_time": "2026-08-26T20:00:00.000000",
                    "end_time": "2026-08-26T20:00:00.000000",
                },
                {
                    "call_type": task_type,
                    "spend": causyn_cost,
                    "start_time": "2026-08-27T09:00:00.000000",
                    "end_time": "2026-08-27T09:00:00.000000",
                },
            ],
            key=lambda row: row["call_type"],
        )
        team_rows = await schema_db.query_raw('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = $1', team_id)
        assert team_rows == [{"spend": 1.25 + causyn_cost}]
    finally:
        if schema_db.is_connected():
            await schema_db.disconnect()
        if control_db.is_connected():
            if schema_created:
                await control_db.execute_raw(f'DROP SCHEMA "{schema_name}" CASCADE')
            await control_db.disconnect()


class FakeStreamRedis:
    def __init__(self, events, stream_key=BILLING_STREAM_KEY):
        self.events = list(events)
        self.stream_key = stream_key
        self.acked = []

    async def xgroup_create(self, *args, **kwargs):
        return True

    async def xreadgroup(self, *args, **kwargs):
        if not self.events:
            return []
        event = self.events.pop(0)
        return [(self.stream_key, [(event[0], event[1])])]

    async def xack(self, stream, group, event_id):
        self.acked.append((stream, group, event_id))


@pytest.mark.asyncio
async def test_replay_after_db_commit_before_ack_is_idempotent():
    event = ImageBillingEvent(
        deployment_id="dep-1",
        provider_task_id="task-1",
        response_cost=2.5,
        team_id="team-1",
        api_key="hashed-key",
    )
    fields = {"payload": json.dumps(event.to_dict())}
    redis = FakeStreamRedis([("1-0", fields), ("1-0", fields)])
    transaction = FakeTransaction(inserted=True)
    reconciler = LibTVBillingReconciler(redis, FakePrisma(transaction), consumer="test")

    await reconciler.reconcile_once()
    await reconciler.reconcile_once()

    assert len(redis.acked) == 2
    assert sum('INSERT INTO "LiteLLM_SpendLogs"' in query for query, _ in transaction.sql) == 2
    assert sum('UPDATE "LiteLLM_VerificationToken"' in query for query, _ in transaction.sql) == 1


@pytest.mark.asyncio
async def test_causyn_event_replays_after_db_failure_without_double_spend():
    event = CausynBillingEvent(
        provider_task_id="task-1",
        response_cost=2.5,
        team_id="team-1",
        api_key="hashed-key",
    )
    fields = {"payload": json.dumps(event.to_dict())}
    redis = FakeStreamRedis([("1-0", fields), ("1-0", fields)], stream_key=CAUSYN_BILLING_STREAM_KEY)
    transaction = FailOnceTransaction()
    reconciler = LibTVBillingReconciler(
        redis,
        FakePrisma(transaction),
        stream_key=CAUSYN_BILLING_STREAM_KEY,
        consumer_group="test-group",
        consumer="test",
        batch_size=1,
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        await reconciler.reconcile_once()
    assert redis.acked == []

    await reconciler.reconcile_once()

    assert redis.acked == [(CAUSYN_BILLING_STREAM_KEY, "test-group", "1-0")]
    assert transaction.attempts >= 2
    assert sum('INSERT INTO "LiteLLM_SpendLogs"' in query for query, _ in transaction.sql) == 1
    assert sum('UPDATE "LiteLLM_TeamTable"' in query for query, _ in transaction.sql) == 1


class FailOnceTransaction(FakeTransaction):
    def __init__(self):
        super().__init__(inserted=True)
        self.failed = False
        self.attempts = 0

    async def execute_raw(self, query, *args):
        self.attempts += 1
        if not self.failed:
            self.failed = True
            raise RuntimeError("database unavailable")
        return await super().execute_raw(query, *args)


@pytest.mark.asyncio
async def test_reconciler_runs_without_http_requests_and_stops_cleanly():
    event = ImageBillingEvent(deployment_id="dep-1", provider_task_id="task-1", response_cost=1.0)
    redis = FakeStreamRedis([("1-0", {"payload": json.dumps(event.to_dict())})])
    transaction = FakeTransaction(inserted=True)
    reconciler = LibTVBillingReconciler(redis, FakePrisma(transaction), poll_interval=0.001, consumer="test")

    await reconciler.start()
    await asyncio.sleep(0.01)
    await reconciler.stop()

    assert redis.acked


def test_custom_libtv_image_poll_is_not_a_generic_spend_call():
    from litellm.proxy.spend_tracking.spend_tracking_utils import is_libtv_image_billing_call

    assert is_libtv_image_billing_call({"call_type": "image_upscale_finalize", "custom_llm_provider": "libtv"})
