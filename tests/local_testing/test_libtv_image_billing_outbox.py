import asyncio
import json
from datetime import datetime

import pytest

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
async def test_billing_event_uses_utc_timestamp_cast_for_image_and_causyn(
    event, expected_call_type, expected_time
):
    transaction = FakeTransaction(inserted=True)
    reconciler = LibTVBillingReconciler(FakeStreamRedis([]), FakePrisma(transaction))

    await reconciler._reconcile_event(event)

    query, args = transaction.sql[0]
    assert query.count("$5::timestamp") == 2
    assert args[1] == expected_call_type
    assert args[4] == expected_time


def test_event_time_assumes_utc_for_naive_iso_timestamp():
    assert _event_time("2026-08-27T04:00:00") == datetime(2026, 8, 27, 4, 0)


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
