import asyncio
import json

import pytest

from litellm.llms.libtv.billing_outbox import (
    BILLING_STREAM_KEY,
    ImageBillingEvent,
    LibTVBillingReconciler,
)
from litellm.llms.libtv.receipts import LibTVReceiptStore, _TRANSITION_SCRIPT, request_fingerprint


class RecordingRedis:
    def __init__(self):
        self.eval_calls = []

    async def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        return ["ok", args[3]]


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
    assert numkeys == 4
    assert args[3] == BILLING_STREAM_KEY
    assert json.loads(args[8])["provider_task_id"] == "task-1"
    assert script.count("XADD") == 1


def test_terminal_transition_script_deduplicates_billing_event_on_repeated_poll():
    assert "current['billing_event_id']" in _TRANSITION_SCRIPT


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
    def __init__(self, events):
        self.events = list(events)
        self.acked = []

    async def xgroup_create(self, *args, **kwargs):
        return True

    async def xreadgroup(self, *args, **kwargs):
        if not self.events:
            return []
        event = self.events.pop(0)
        return [(BILLING_STREAM_KEY, [(event[0], event[1])])]

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
