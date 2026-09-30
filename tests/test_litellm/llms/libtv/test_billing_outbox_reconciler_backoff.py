import json
import logging
from decimal import Decimal
from types import SimpleNamespace

import pytest

from litellm.llms.libtv import billing_outbox
from litellm.llms.libtv.billing_outbox import (
    CausynBillingEvent,
    LibTVBillingReconciler,
    PermanentBillingFailure,
)
from litellm.proxy.video_endpoints.moderation_metering import KeyIdentityMismatch


class Redis:
    def __init__(self, claimed=(), fresh=()):
        self.claimed = list(claimed)
        self.fresh = list(fresh)
        self.claim_kwargs = None
        self.acked = []
        self.added = []

    async def xgroup_create(self, *a, **k):
        return True

    async def xautoclaim(self, *args, **kwargs):
        self.claim_kwargs = kwargs
        return ("0-0", self.claimed, [])

    async def xreadgroup(self, *args, **kwargs):
        stream = next(iter(kwargs["streams"]))
        return [(stream, self.fresh)] if self.fresh else []

    async def xack(self, stream, group, event_id):
        self.acked.append(event_id)

    async def xadd(self, stream, fields):
        self.added.append((stream, fields))


def good_fields(task="task-1"):
    event = CausynBillingEvent(provider_task_id=task, response_cost=1.0, team_id="t", api_key="k")
    return {"payload": json.dumps(event.to_dict())}


def make(redis, monkeypatch, effect=None):
    worker = LibTVBillingReconciler(redis, SimpleNamespace(), stream_key="causyn:billing:outbox", consumer="c")

    async def reconcile(event):
        if effect is not None:
            raise effect

    monkeypatch.setattr(worker, "_reconcile_event", reconcile)
    return worker


@pytest.mark.asyncio
async def test_reclaims_only_entries_idle_at_least_sixty_seconds(monkeypatch):
    monkeypatch.delenv("LITELLM_BILLING_OUTBOX_RETRY_IDLE_MS", raising=False)
    redis = Redis()
    await make(redis, monkeypatch).reconcile_once()
    assert redis.claim_kwargs["min_idle_time"] == 60000


@pytest.mark.asyncio
async def test_retry_idle_is_configurable_by_env(monkeypatch):
    monkeypatch.setenv("LITELLM_BILLING_OUTBOX_RETRY_IDLE_MS", "1234")
    redis = Redis()
    await make(redis, monkeypatch).reconcile_once()
    assert redis.claim_kwargs["min_idle_time"] == 1234


@pytest.mark.asyncio
async def test_success_is_acked_in_the_same_iteration(monkeypatch):
    redis = Redis(fresh=[("1-0", good_fields())])
    assert await make(redis, monkeypatch).reconcile_once() == 1
    assert redis.acked == ["1-0"] and redis.added == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [KeyIdentityMismatch("gone"), PermanentBillingFailure("parked")])
async def test_permanent_failure_is_dead_lettered_then_acked(monkeypatch, error):
    fields = good_fields()
    redis = Redis(claimed=[("1-0", fields)])
    await make(redis, monkeypatch, error).reconcile_once()
    ((stream, entry),) = redis.added
    assert stream == "causyn:billing:outbox:dead"
    assert entry["payload"] == fields["payload"] and entry["source_event_id"] == "1-0"
    assert type(error).__name__ in entry["reason"] and str(error) in entry["reason"]
    assert redis.acked == ["1-0"]


@pytest.mark.asyncio
async def test_transient_failure_is_retried_and_logged_with_type_and_message(monkeypatch, caplog):
    redis = Redis(fresh=[("1-0", good_fields())])
    caplog.set_level(logging.WARNING)
    await make(redis, monkeypatch, RuntimeError("x" * 500)).reconcile_once()
    assert redis.acked == [] and redis.added == []
    line = next(r.getMessage() for r in caplog.records if "1-0" in r.getMessage())
    assert "RuntimeError: xxx" in line and "x" * 301 not in line


@pytest.mark.asyncio
async def test_parse_failure_goes_to_dead_stream(monkeypatch):
    redis = Redis(claimed=[("2-0", {"payload": "{not json"})])
    await make(redis, monkeypatch).reconcile_once()
    ((stream, entry),) = redis.added
    assert stream == "causyn:billing:outbox:dead" and "unparseable" in entry["reason"]
    assert redis.acked == ["2-0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["needs_manual_settlement", "waived", "void"])
async def test_settle_outbox_raises_permanent_for_parked_phase(monkeypatch, status):
    from litellm.proxy.video_endpoints import moderation_metering_outbox as outbox
    from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent

    binding = BillingBinding(
        intent_id="i",
        request_digest="a" * 64,
        fingerprint="k",
        user_id="u",
        team_id="t",
        model="m",
        expected_phases=("completion",),
    )
    phase = PhaseEvent(
        binding=binding,
        request_id="public-video:i:completion",
        phase="completion",
        provider="causyn",
        deployment_id="d",
        native_id="n",
        provider_task_id="p",
        amount=Decimal(0),
        finalized=True,
    )

    class Authority:
        async def persist(self, p):
            pass

        async def run_once(self):
            return False

        async def settlement(self, b):
            return SimpleNamespace(receipts=())

        async def phase_status(self, request_id):
            return status

    monkeypatch.setattr(outbox.runtime, "store", lambda: Authority())
    event = CausynBillingEvent(
        provider_task_id="p",
        response_cost=0,
        team_id="t",
        api_key="k",
        user_id="u",
        provider="causyn",
        deployment_id="d",
        metering_event_json=phase.model_dump_json(),
    )
    with pytest.raises(PermanentBillingFailure):
        await outbox.settle_outbox(event)
