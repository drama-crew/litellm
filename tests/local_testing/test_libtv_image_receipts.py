import asyncio

import pytest

fakeredis = pytest.importorskip("fakeredis")
from fakeredis import aioredis as fakeredis_aioredis
from redis.exceptions import RedisError

from litellm.llms.libtv.image_upscale import (
    ImageUpscaleSubmitter,
    IdempotencyFingerprintMismatch,
    ProviderRejected,
    ProviderTransportError,
)
from litellm.llms.libtv.receipts import LibTVReceiptStore, request_fingerprint


def _payload(request_id: str = "request-1", style: str = "Standard V2") -> dict[str, object]:
    return {
        "request_id": request_id,
        "team_id": "team-1",
        "model": "topaz-image-upscaler",
        "source_sha256": "a" * 64,
        "source_url": "https://assets.example/input.png",
        "style": style,
        "scale": 2,
    }


def _store():
    return LibTVReceiptStore(fakeredis_aioredis.FakeRedis(decode_responses=True))


@pytest.mark.asyncio
async def test_concurrent_submitters_create_once_and_share_receipt():
    store = _store()

    class Provider:
        def __init__(self):
            self.calls = 0

        async def create(self, payload):
            self.calls += 1
            await asyncio.sleep(0.01)
            return {"task_id": "task-1"}

    provider = Provider()
    submitter = ImageUpscaleSubmitter(
        ("primary", provider), receipt_store=store, team_id="team-1", model="topaz-image-upscaler"
    )
    receipts = await asyncio.gather(submitter.submit(_payload()), submitter.submit(_payload()))

    assert provider.calls == 1
    assert receipts[0] == receipts[1]
    assert receipts[0].submission_state == "submitted"
    assert receipts[0].provider_task_id == "task-1"


@pytest.mark.asyncio
async def test_same_request_with_different_fingerprint_returns_conflict():
    store = _store()

    class Provider:
        async def create(self, payload):
            return {"task_id": "task-1"}

    submitter = ImageUpscaleSubmitter(
        ("primary", Provider()), receipt_store=store, team_id="team-1", model="topaz-image-upscaler"
    )
    await submitter.submit(_payload())

    with pytest.raises(IdempotencyFingerprintMismatch) as error:
        await submitter.submit(_payload(style="CGI"))
    assert error.value.status_code == 409
    assert error.value.code == "idempotency_fingerprint_mismatch"


@pytest.mark.asyncio
async def test_explicit_rejection_allows_secondary_claim():
    store = _store()
    calls = []

    class Rejected:
        async def create(self, payload):
            calls.append("primary")
            raise ProviderRejected("capacity", provider_code="1200000136")

    class Accepted:
        async def create(self, payload):
            calls.append("secondary")
            return {"task_id": "task-2"}

    receipt = await ImageUpscaleSubmitter(
        ("primary", Rejected()),
        ("secondary", Accepted()),
        receipt_store=store,
        team_id="team-1",
        model="topaz-image-upscaler",
    ).submit(_payload())

    assert calls == ["primary", "secondary"]
    assert receipt.submission_state == "submitted"
    assert receipt.deployment_id == "secondary"


@pytest.mark.asyncio
async def test_unknown_transport_locks_pool_for_secondary_submitter():
    store = _store()
    primary_calls = []
    secondary_calls = []

    class Unknown:
        async def create(self, payload):
            primary_calls.append(1)
            raise ProviderTransportError("connection lost", crossed_create_boundary=True)

    class Secondary:
        async def create(self, payload):
            secondary_calls.append(1)
            return {"task_id": "task-2"}

    first = await ImageUpscaleSubmitter(
        ("primary", Unknown()),
        ("secondary", Secondary()),
        receipt_store=store,
        team_id="team-1",
        model="topaz-image-upscaler",
    ).submit(_payload())
    second = await ImageUpscaleSubmitter(
        ("primary", Unknown()),
        ("secondary", Secondary()),
        receipt_store=store,
        team_id="team-1",
        model="topaz-image-upscaler",
    ).submit(_payload())

    assert primary_calls == [1]
    assert secondary_calls == []
    assert first == second
    assert second.submission_state == "unknown"


@pytest.mark.asyncio
async def test_missing_receipt_after_pending_is_unknown_and_not_resubmittable():
    redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    store = LibTVReceiptStore(redis)
    claim = await store.claim(
        "team-1",
        "topaz-image-upscaler",
        "request-1",
        request_fingerprint(_payload(), "topaz-image-upscaler"),
        "primary",
    )
    await redis.delete(claim.receipt_key)

    calls = []

    class Provider:
        async def create(self, payload):
            calls.append(1)
            return {"task_id": "task-1"}

    receipt = await ImageUpscaleSubmitter(
        ("primary", Provider()), receipt_store=store, team_id="team-1", model="topaz-image-upscaler"
    ).submit(_payload())

    assert calls == []
    assert receipt.submission_state == "unknown"
    assert receipt.message == "receipt missing after pending claim"


@pytest.mark.asyncio
async def test_missing_receipt_still_rejects_a_different_fingerprint():
    redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    store = LibTVReceiptStore(redis)
    claim = await store.claim("team-1", "topaz-image-upscaler", "request-1", "f" * 64, "primary")
    await redis.delete(claim.receipt_key)

    with pytest.raises(IdempotencyFingerprintMismatch):
        await ImageUpscaleSubmitter(
            ("primary", object()), receipt_store=store, team_id="team-1", model="topaz-image-upscaler"
        ).submit(_payload())


@pytest.mark.asyncio
async def test_receipt_store_redis_error_fails_closed_before_provider_create():
    class BrokenStore:
        async def claim(self, *args, **kwargs):
            raise RedisError("redis unavailable")

    calls = []

    class Provider:
        async def create(self, payload):
            calls.append(1)
            return {"task_id": "task-1"}

    receipt = await ImageUpscaleSubmitter(
        ("primary", Provider()), receipt_store=BrokenStore(), team_id="team-1", model="topaz-image-upscaler"
    ).submit(_payload())

    assert calls == []
    assert receipt.submission_state == "not_submitted"
    assert receipt.message == "receipt store unavailable"


@pytest.mark.asyncio
async def test_receipt_store_readiness_requires_durable_aof_and_write_read():
    class Redis:
        def __init__(self, config):
            self.config = config
            self.values = {}

        async def ping(self):
            return True

        async def config_get(self, *names):
            return self.config

        async def set(self, key, value):
            self.values[key] = value
            return True

        async def get(self, key):
            return self.values.get(key)

    assert await LibTVReceiptStore(Redis({"appendonly": "yes", "appendfsync": "always"})).readiness()
    assert not await LibTVReceiptStore(Redis({"appendonly": "yes", "appendfsync": "everysec"})).readiness()
