import asyncio
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from litellm.llms.libtv.image_upscale import (
    ImageUpscaleSubmitter,
    IdempotencyFingerprintMismatch,
    ProviderRejected,
    ProviderTransportError,
    make_resume_token,
    verify_resume_token,
    normalize_image_upscale_receipt,
)
from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.receipts import LibTVReceiptStore, StoredReceipt, request_fingerprint


def _payload(request_id: str = "request-1", style: str = "Standard V2") -> dict[str, object]:
    return {
        "request_id": request_id,
        "team_id": "team-1",
        "model": "topaz-image-upscaler",
        "source_sha256": "a" * 64,
        "source_url": "https://assets.example/input.png",
        "style": style,
        "scale": 2,
        "response_cost": 0.25,
        "api_key": "key-1",
        "user_id": "user-1",
        "organization_id": "org-1",
    }


@pytest.fixture(scope="module")
def redis_url(tmp_path_factory):
    data_dir = Path(tmp_path_factory.mktemp("libtv-receipts-redis"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            "redis-server",
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--dir",
            str(data_dir),
            "--appendonly",
            "yes",
            "--appendfsync",
            "always",
            "--save",
            "",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = SyncRedis(host="127.0.0.1", port=port, decode_responses=True)
    for _ in range(100):
        try:
            if client.ping():
                break
        except Exception:
            time.sleep(0.01)
    else:
        process.terminate()
        raise RuntimeError("redis-server did not become ready")
    yield f"redis://127.0.0.1:{port}/0"
    client.close()
    process.terminate()
    process.wait(timeout=5)


async def _store(redis_url: str) -> LibTVReceiptStore:
    store = LibTVReceiptStore(Redis.from_url(redis_url, decode_responses=True))
    await store.redis.flushdb()
    return store


@pytest.mark.asyncio
async def test_concurrent_submitters_create_once_and_share_receipt(redis_url):
    store = await _store(redis_url)

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
async def test_duplicate_submitted_request_does_not_reupload_source(redis_url):
    store = await _store(redis_url)
    uploads = []

    class Provider:
        async def create(self, payload):
            uploads.append(payload["source_url"])
            return {"task_id": "task-1"}

    submitter = ImageUpscaleSubmitter(
        ("primary", Provider()), receipt_store=store, team_id="team-1", model="topaz-image-upscaler"
    )
    first = await submitter.submit(_payload())

    class UploadWouldFail:
        async def create(self, payload):
            raise AssertionError("duplicate submission must not upload or create")

    duplicate = await ImageUpscaleSubmitter(
        ("primary", UploadWouldFail()), receipt_store=store, team_id="team-1", model="topaz-image-upscaler"
    ).submit(_payload())

    assert first == duplicate
    assert uploads == ["https://assets.example/input.png"]


def test_not_submitted_receipt_retains_deployment_audit_identity():
    receipt = normalize_image_upscale_receipt(
        {
            "receipt": {
                "request_id": "request-1",
                "submission_state": "not_submitted",
                "deployment_id": "primary",
                "message": "source transfer failed",
            }
        },
        "request-1",
    )

    assert receipt.submission_state == "not_submitted"
    assert receipt.deployment_id == "primary"


@pytest.mark.asyncio
async def test_pre_create_failure_retains_deployment_in_retryable_receipt():
    class Provider:
        async def create(self, payload):
            raise ProviderTransportError("source transfer failed", crossed_create_boundary=False)

    receipt = await ImageUpscaleSubmitter(
        ("primary", Provider()),
        team_id="team-1",
        api_key="key-1",
        user_id="user-1",
        organization_id="org-1",
    ).submit(_payload())

    assert receipt.submission_state == "not_submitted"
    assert receipt.deployment_id == "primary"
    assert receipt.message == "source transfer failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("cost", [None, 0.0, float("nan"), float("inf")])
async def test_invalid_billing_contract_blocks_provider_create(cost):
    calls = []

    class Provider:
        async def create(self, payload):
            calls.append(payload)
            return {"task_id": "task-1"}

    receipt = await ImageUpscaleSubmitter(
        ("primary", Provider()),
        receipt_store=None,
        team_id="team-1",
        api_key="key-1",
        user_id="user-1",
        organization_id="org-1",
    ).submit({**_payload(), "response_cost": cost})

    assert calls == []
    assert receipt.submission_state == "not_submitted"


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_field", ["team_id", "api_key", "user_id", "organization_id"])
async def test_missing_billing_identity_blocks_provider_create(identity_field):
    calls = []

    class Provider:
        async def create(self, payload):
            calls.append(payload)
            return {"task_id": "task-1"}

    constructor_identity = {
        "team_id": "team-1",
        "api_key": "key-1",
        "user_id": "user-1",
        "organization_id": "org-1",
    }
    constructor_identity[identity_field] = None
    payload = _payload()
    payload.pop(identity_field)

    receipt = await ImageUpscaleSubmitter(("primary", Provider()), **constructor_identity).submit(payload)

    assert calls == []
    assert receipt.submission_state == "not_submitted"


@pytest.mark.asyncio
async def test_payload_billing_identity_allows_provider_create():
    calls = []

    class Provider:
        async def create(self, payload):
            calls.append(payload)
            return {"task_id": "task-1"}

    receipt = await ImageUpscaleSubmitter(("primary", Provider())).submit(_payload())

    assert len(calls) == 1
    assert receipt.submission_state == "submitted"


@pytest.mark.asyncio
async def test_deployment_attempt_receipts_are_distinct_and_enumerable(redis_url):
    store = await _store(redis_url)
    first = await store.claim("team-1", "topaz-image-upscaler", "request-1", "f" * 64, "primary")
    await store.transition(first.receipt, first.receipt_key, "rejected", message="capacity")
    second = await store.claim("team-1", "topaz-image-upscaler", "request-1", "f" * 64, "secondary")

    assert first.receipt_key != second.receipt_key
    assert second.outcome == "owner"
    attempts = await store.get_attempts("team-1", "topaz-image-upscaler", "request-1")
    assert {(attempt.deployment_id, attempt.submission_state) for attempt in attempts} == {
        ("primary", "rejected"),
        ("secondary", "submitting"),
    }


@pytest.mark.asyncio
async def test_receipt_persists_authenticated_identity_snapshot(redis_url):
    store = await _store(redis_url)

    class Provider:
        async def create(self, payload):
            return {"task_id": "task-identity"}

    receipt = await ImageUpscaleSubmitter(
        ("primary", Provider()),
        receipt_store=store,
        team_id="team-1",
        api_key="hashed-key-id",
        user_id="user-1",
        organization_id="org-1",
        model="topaz-image-upscaler",
    ).submit(_payload())

    stored = await store.get("team-1", "topaz-image-upscaler", "request-1")

    assert receipt.submission_state == "submitted"
    assert stored is not None
    assert stored.api_key == "hashed-key-id"
    assert stored.user_id == "user-1"
    assert stored.organization_id == "org-1"


@pytest.mark.asyncio
async def test_same_request_with_different_fingerprint_returns_conflict(redis_url):
    store = await _store(redis_url)

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
async def test_explicit_rejection_allows_secondary_claim(redis_url):
    store = await _store(redis_url)
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
    pool_key = store._pool_key("team-1", "topaz-image-upscaler", "request-1")
    pool = json.loads(await store.redis.get(pool_key))
    assert {(attempt["deployment_id"], attempt["submission_state"]) for attempt in pool["attempts"]} == {
        ("primary", "rejected"),
        ("secondary", "submitted"),
    }


@pytest.mark.asyncio
async def test_unknown_transport_locks_pool_for_secondary_submitter(redis_url):
    store = await _store(redis_url)
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
async def test_missing_receipt_after_pending_is_unknown_and_not_resubmittable(redis_url):
    store = await _store(redis_url)
    redis = store.redis
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
async def test_missing_receipt_still_rejects_a_different_fingerprint(redis_url):
    store = await _store(redis_url)
    redis = store.redis
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
async def test_durable_client_fails_closed_when_receipt_redis_is_not_ready(monkeypatch):
    class NotReadyStore:
        async def readiness(self):
            return False

    provider_calls = []

    async def unexpected_create(*args, **kwargs):
        provider_calls.append(1)
        return {"task_id": "task-1"}

    monkeypatch.setattr("litellm.llms.libtv.client.get_receipt_store", lambda: NotReadyStore())
    client = LibTVClient(token="token", webid="webid")
    monkeypatch.setattr(client, "acreate", unexpected_create)

    receipt = await client.asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://source.example/input.png",
        "Standard V2",
        2,
        "project",
        "request-not-ready",
        "primary",
        team_id="team-1",
        source_sha256="a" * 64,
        durable_receipts=True,
    )

    assert provider_calls == []
    assert receipt.submission_state == "not_submitted"
    assert receipt.message == "receipt store is not durable"


@pytest.mark.asyncio
async def test_unknown_eval_command_fails_closed_without_provider_create():
    class NoLuaRedis:
        async def eval(self, *args):
            raise ResponseError("unknown command 'eval'")

    calls = []

    class Provider:
        async def create(self, payload):
            calls.append(1)
            return {"task_id": "task-1"}

    receipt = await ImageUpscaleSubmitter(
        ("primary", Provider()),
        receipt_store=LibTVReceiptStore(NoLuaRedis()),
        team_id="team-1",
        model="topaz-image-upscaler",
    ).submit(_payload())

    assert calls == []
    assert receipt.submission_state == "not_submitted"
    assert receipt.message == "receipt store unavailable"


@pytest.mark.asyncio
async def test_transition_requires_expected_state_and_cannot_overwrite_submitted(redis_url):
    store = await _store(redis_url)
    claim = await store.claim("team-1", "topaz-image-upscaler", "request-1", "f" * 64, "primary")
    submitted = await store.transition(
        claim.receipt, claim.receipt_key, "submitted", provider_task_id="task-1", resume_token="token"
    )

    with pytest.raises(RedisError):
        await store.transition(submitted, claim.receipt_key, "unknown", expected_state="submitting")

    current = await store.get("team-1", "topaz-image-upscaler", "request-1")
    assert current is not None
    assert current.submission_state == "submitted"


@pytest.mark.asyncio
async def test_transition_with_null_resolution_tombstone_performs_cas(redis_url):
    """JSON null is not a resolved receipt in Redis Lua/cjson."""
    store = await _store(redis_url)
    claim = await store.claim("team-1", "topaz-image-upscaler", "request-1", "f" * 64, "primary")

    submitted = await store.transition(
        claim.receipt,
        claim.receipt_key,
        "submitted",
        provider_task_id="task-1",
        resume_token="token-1",
    )

    assert submitted.submission_state == "submitted"
    assert submitted.provider_task_id == "task-1"
    assert submitted.resolution_tombstone is None


@pytest.mark.asyncio
async def test_terminal_task_state_cannot_be_overwritten(redis_url):
    store = await _store(redis_url)
    claim = await store.claim("team-1", "topaz-image-upscaler", "request-1", "f" * 64, "primary")
    submitted = await store.transition(
        claim.receipt,
        claim.receipt_key,
        "submitted",
        provider_task_id="task-1",
        resume_token="token-1",
    )
    succeeded = await store.transition(
        submitted,
        claim.receipt_key,
        "submitted",
        expected_state="submitted",
        task_state="succeeded",
        terminal_result={"url": "https://provider.example/result.png", "provider_task_id": "task-1"},
    )
    repeated = await store.transition(
        succeeded,
        claim.receipt_key,
        "submitted",
        expected_state="submitted",
        task_state="failed",
    )

    assert repeated.task_state == "succeeded"


@pytest.mark.asyncio
async def test_terminal_receipt_persists_replayable_result(redis_url):
    store = await _store(redis_url)
    claim = await store.claim(
        "team-1",
        "topaz-image-upscaler",
        "request-1",
        "f" * 64,
        "primary",
        response_cost=0.25,
        api_key="key-1",
        user_id="user-1",
        organization_id="org-1",
    )
    submitted = await store.transition(
        claim.receipt,
        claim.receipt_key,
        "submitted",
        provider_task_id="task-1",
        resume_token="token-1",
    )

    result = {"url": "https://provider.example/result.png", "provider_task_id": "task-1"}
    succeeded = await store.transition(
        submitted,
        claim.receipt_key,
        "submitted",
        expected_state="submitted",
        task_state="succeeded",
        terminal_result=result,
    )
    repeated = await store.transition(
        succeeded,
        claim.receipt_key,
        "submitted",
        expected_state="submitted",
        task_state="succeeded",
        terminal_result={"url": "https://different.example/result.png", "provider_task_id": "task-1"},
    )

    assert succeeded.terminal_result == result
    assert repeated.terminal_result == result
    assert (await store.get("team-1", "topaz-image-upscaler", "request-1")).terminal_result == result


@pytest.mark.asyncio
async def test_repeated_finalize_replays_terminal_result_without_provider_or_duplicate_billing(
    redis_url, monkeypatch
):
    from litellm.proxy.image_endpoints import endpoints

    store = await _store(redis_url)
    claim = await store.claim(
        "team-1",
        "topaz-image-upscaler",
        "request-1",
        "f" * 64,
        "primary",
        response_cost=0.25,
        api_key="key-1",
        user_id="user-1",
        organization_id="org-1",
    )
    await store.transition(
        claim.receipt,
        claim.receipt_key,
        "submitted",
        provider_task_id="task-1",
        resume_token="token-1",
    )

    class Provider:
        calls = 0

        async def apoll_image_upscale(self, provider_task_id):
            self.calls += 1
            assert provider_task_id == "task-1"
            return {"status": 2, "urls": ["https://provider.example/result.png"]}

    provider = Provider()

    async def load(*_args):
        receipt = await store.get("team-1", "topaz-image-upscaler", "request-1")
        assert receipt is not None
        return store, receipt, claim.receipt_key, provider

    monkeypatch.setattr(endpoints, "_load_action_receipt", load)
    action = SimpleNamespace(model="topaz-image-upscaler", request_id="request-1")

    first = await endpoints._poll_image_upscale(action, object(), True)
    second = await endpoints._poll_image_upscale(action, object(), True)

    assert first.status_code == second.status_code == 200
    assert json.loads(first.body)["result"] == json.loads(second.body)["result"] == {
        "url": "https://provider.example/result.png",
        "provider_task_id": "task-1",
    }
    assert provider.calls == 1
    current = await store.get("team-1", "topaz-image-upscaler", "request-1")
    assert await store.redis.xlen("libtv:billing:outbox") == 1, current.billing_event_id


@pytest.mark.asyncio
async def test_receipt_persists_only_allowlisted_upscale_attribution(redis_url):
    store = await _store(redis_url)

    class Provider:
        async def create(self, payload):
            return {"task_id": "task-attribution"}

    receipt = await ImageUpscaleSubmitter(
        ("primary", Provider()),
        receipt_store=store,
        team_id="team-1",
        api_key="key-1",
        user_id="billing-user-1",
        organization_id="org-1",
    ).submit(
        {
            **_payload(),
            "scale": 4,
            "spend_logs_metadata": {
                "project_id": "project-1",
                "artifact_id": "artifact-1",
                "user_id": "owner-1",
                "untrusted": {"nested": "must-not-persist"},
            },
        }
    )

    stored = await store.get("team-1", "topaz-image-upscaler", receipt.request_id)
    assert stored is not None
    assert stored.scale == 4
    assert stored.project_id == "project-1"
    assert stored.artifact_id == "artifact-1"
    assert stored.attribution_user_id == "owner-1"
    assert "untrusted" not in stored.to_json()


def test_legacy_receipt_defaults_to_active_provider_task_state():
    receipt = StoredReceipt.from_json(
        json.dumps(
            {
                "team_id": "team-1",
                "model": "topaz-image-upscaler",
                "request_id": "request-1",
                "fingerprint": "f" * 64,
                "submission_state": "submitted",
                "deployment_id": "primary",
                "provider_task_id": "task-1",
            }
        )
    )

    assert receipt.task_state == "active"


def test_durable_resume_token_binds_receipt_identity():
    token = make_resume_token(
        "dep-1",
        "task-1",
        "secret",
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
    )

    assert verify_resume_token(
        token,
        "secret",
        deployment_id="dep-1",
        provider_task_id="task-1",
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
    )
    assert not verify_resume_token(token, "secret", team_id="other-team")


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

        async def delete(self, key):
            self.values.pop(key, None)

    ready = Redis({"appendonly": "yes", "appendfsync": "always"})
    assert await LibTVReceiptStore(ready).readiness()
    assert ready.values == {}

    not_ready = Redis({"appendonly": "yes", "appendfsync": "everysec"})
    assert not await LibTVReceiptStore(not_ready).readiness()
    assert not_ready.values == {}


@pytest.mark.asyncio
async def test_receipt_store_readiness_cleans_probe_after_write_read_failure():
    class Redis:
        def __init__(self):
            self.values = {}

        async def ping(self):
            return True

        async def config_get(self, *names):
            return {"appendonly": "yes", "appendfsync": "always"}

        async def set(self, key, value):
            self.values[key] = value

        async def get(self, key):
            return "not-ok"

        async def delete(self, key):
            self.values.pop(key, None)

    redis = Redis()
    assert not await LibTVReceiptStore(redis).readiness()
    assert redis.values == {}


@pytest.mark.asyncio
async def test_receipt_store_readiness_cleans_probe_when_cancelled():
    entered_get = asyncio.Event()
    release_get = asyncio.Event()

    class Redis:
        def __init__(self):
            self.values = {}

        async def ping(self):
            return True

        async def config_get(self, *names):
            return {"appendonly": "yes", "appendfsync": "always"}

        async def set(self, key, value):
            self.values[key] = value

        async def get(self, key):
            entered_get.set()
            await release_get.wait()
            return self.values.get(key)

        async def delete(self, key):
            self.values.pop(key, None)

    redis = Redis()
    task = asyncio.create_task(LibTVReceiptStore(redis).readiness())
    await entered_get.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert redis.values == {}


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("redis-server") is None, reason="redis-server is required for the AOF restart proof")
async def test_real_redis_aof_restart_preserves_submitted_receipt(tmp_path):
    data_dir = tmp_path / "redis-data"
    data_dir.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def start_server():
        process = subprocess.Popen(
            [
                "redis-server",
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                "--dir",
                str(data_dir),
                "--appendonly",
                "yes",
                "--appendfsync",
                "always",
                "--save",
                "",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = SyncRedis(host="127.0.0.1", port=port, decode_responses=True)
        for _ in range(100):
            try:
                if client.ping():
                    return process, client
            except Exception:
                time.sleep(0.01)
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError("redis-server did not become ready")

    process, sync_client = start_server()
    redis_url = f"redis://127.0.0.1:{port}/0"
    try:
        async_store = LibTVReceiptStore(Redis.from_url(redis_url, decode_responses=True))
        claim = await async_store.claim("team-1", "topaz-image-upscaler", "restart-request", "f" * 64, "primary")
        await async_store.transition(
            claim.receipt,
            claim.receipt_key,
            "submitted",
            provider_task_id="task-restart-1",
            resume_token="resume-restart-1",
        )
        config = sync_client.config_get("appendonly", "appendfsync")
        assert config["appendonly"] == "yes"
        assert config["appendfsync"] == "always"
        await async_store.redis.aclose()
        sync_client.close()
        process.terminate()
        process.wait(timeout=5)

        process, sync_client = start_server()
        restarted = LibTVReceiptStore(Redis.from_url(redis_url, decode_responses=True))
        receipt = await restarted.get_receipt("team-1", "topaz-image-upscaler", "restart-request")
        assert receipt is not None
        assert receipt.submission_state == "submitted"
        assert receipt.provider_task_id == "task-restart-1"
        assert receipt.resume_token == "resume-restart-1"
        await restarted.redis.aclose()
    finally:
        sync_client.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
