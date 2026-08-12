"""Contract tests for LiteLLM's strict validated-media fallback.

These tests deliberately use an in-process HTTP transport.  They do not make
network or provider calls.
"""

from __future__ import annotations

import base64
import asyncio
import hashlib

import httpx
import pytest

from litellm.llms.libtv.validated_transfer import (
    VALIDATION_VERSION,
    StrictValidatedTransferExecutor,
    ValidatedTransferError,
    ValidatedTransferRequest,
    ValidatedTransferSettings,
    ValidatedTransferRouter,
    get_shared_validated_transfer_router,
    reset_shared_validated_transfer_routers,
)
from litellm.llms.libtv.transfer import ValidatedDelegatedTransfer


def _png_bytes(size: tuple[int, int] = (4, 3)) -> bytes:
    assert size == (4, 3)
    # A small valid 4x3 RGB PNG fixture; keeping this literal makes RED usable
    # before the production Pillow dependency has been installed.
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEUlEQVR4nGP8z4AATEhsVA4AJnYBBZNezToAAAAASUVORK5CYII="
    )


def _request(*, source_url: str = "https://source.example/image.png", target_url: str = "https://target.example/part-1") -> ValidatedTransferRequest:
    payload = _png_bytes()
    return ValidatedTransferRequest.from_payload(
        {
            "type": "validated_media_transfer",
            "task_id": "task-1",
            "mode": "transfer",
            "source": {"url": source_url, "bytes": len(payload)},
            "quota_bytes": len(payload),
            "target": {"kind": "presigned_parts", "parts": [{"n": 1, "url": target_url}], "part_size": 1024},
        }
    )


def test_request_accepts_platform_quota_bytes_and_enforces_it(tmp_path):
    payload = {
        "type": "validated_media_transfer", "task_id": "task-1", "mode": "transfer",
        "source": {"url": "https://source.example/image.png"},
        "quota_bytes": 10,
        "target": {"kind": "presigned_parts", "parts": [{"n": 1, "url": "https://target.example/part-1"}], "part_size": 1024},
    }
    request = ValidatedTransferRequest.from_payload(payload)
    assert request.quota_bytes == 10
    assert request.task_id == "task-1"
    with pytest.raises(ValidatedTransferError, match="byte limit"):
        StrictValidatedTransferExecutor(_settings(tmp_path)).validate_request(
            ValidatedTransferRequest.from_payload({**payload, "source": {"url": payload["source"]["url"], "bytes": 11}})
        )


def _settings(tmp_path) -> ValidatedTransferSettings:
    return ValidatedTransferSettings(
        source_hosts=frozenset({"source.example"}),
        target_hosts=frozenset({"target.example"}),
        spool_dir=tmp_path,
        hard_cap=1024 * 1024,
        quota_bytes=1024 * 1024,
        max_dimensions=4096,
        max_pixels=4096 * 4096,
        max_parts=10,
        chunk_size=1024,
        put_retries=1,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://source.example/image.png",
        "https://user@source.example/image.png",
        "https://source.example/image.png#fragment",
        "https://source.example:444/image.png",
        "https://source.example.evil/image.png",
    ],
)
def test_request_rejects_invalid_source_url_boundaries(tmp_path, url):
    with pytest.raises(ValidatedTransferError, match="source URL"):
        StrictValidatedTransferExecutor(_settings(tmp_path)).validate_request(_request(source_url=url))


@pytest.mark.parametrize("status", [301, 302, 307, 308])
@pytest.mark.asyncio
async def test_executor_rejects_redirect_without_following_it(tmp_path, status):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"location": "https://source.example/other.png"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    executor = StrictValidatedTransferExecutor(_settings(tmp_path), client=client)

    with pytest.raises(ValidatedTransferError, match="source HTTP"):
        await executor.execute(_request())

    await client.aclose()


@pytest.mark.asyncio
async def test_executor_streams_valid_image_and_returns_canonical_metadata(tmp_path):
    image = _png_bytes()
    uploads: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, headers={"content-length": str(len(image)), "content-type": "text/plain"}, content=image)
        uploads.append(await request.aread())
        return httpx.Response(200, headers={"etag": '"etag-1"'})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    executor = StrictValidatedTransferExecutor(_settings(tmp_path), client=client)
    result = await executor.execute(_request())

    assert result["validation_version"] == VALIDATION_VERSION == "image-v1"
    assert result["mime"] == "image/png"
    assert result["width"] == 4
    assert result["height"] == 3
    assert result["etags"] == [{"n": 1, "etag": '"etag-1"'}]
    assert uploads == [image]
    assert list(tmp_path.iterdir()) == []
    await client.aclose()


@pytest.mark.asyncio
async def test_executor_removes_spool_after_stream_cap_rejection(tmp_path):
    body = b"x" * 64

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    settings = _settings(tmp_path)
    settings = settings.with_limits(hard_cap=32, quota_bytes=32)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

    with pytest.raises(ValidatedTransferError, match="byte limit"):
        await StrictValidatedTransferExecutor(settings, client=client).execute(_request())

    assert list(tmp_path.iterdir()) == []
    await client.aclose()


@pytest.mark.asyncio
async def test_executor_rejects_mismatched_declared_content_length(tmp_path):
    image = _png_bytes()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": str(len(image) + 1), "content-type": "image/png"}, content=image)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    request = ValidatedTransferRequest.from_payload(
        {"type": "validated_media_transfer", "mode": "probe", "source": {"url": "https://source.example/image.png"}, "hard_cap": 1024}
    )
    with pytest.raises(ValidatedTransferError, match="Content-Length"):
        await StrictValidatedTransferExecutor(_settings(tmp_path), client=client).execute(request)
    await client.aclose()


@pytest.mark.asyncio
async def test_executor_rejects_corrupt_body_despite_image_content_type(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"not an image")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    request = ValidatedTransferRequest.from_payload(
        {"type": "validated_media_transfer", "mode": "probe", "source": {"url": "https://source.example/image.png"}, "hard_cap": 1024}
    )
    with pytest.raises(ValidatedTransferError, match="invalid or unsafe image"):
        await StrictValidatedTransferExecutor(_settings(tmp_path), client=client).execute(request)
    await client.aclose()


@pytest.mark.asyncio
async def test_source_transport_failure_is_retryable_not_validation(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(ValidatedTransferError) as error:
        await StrictValidatedTransferExecutor(_settings(tmp_path), client=client).execute(_request())
    assert error.value.validation is False
    await client.aclose()


@pytest.mark.asyncio
async def test_router_uses_local_executor_without_redis_or_heartbeat(tmp_path):
    image = _png_bytes()
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200, content=image) if request.method == "GET" else httpx.Response(200, headers={"etag": "e"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = ValidatedTransferRouter(StrictValidatedTransferExecutor(_settings(tmp_path), client=client), redis=None)
    result = await router.execute(_request())
    assert result["route"] == "local_no_worker"
    assert calls == ["GET", "PUT"]
    await client.aclose()


@pytest.mark.asyncio
async def test_router_prefers_fresh_remote_worker_result(tmp_path):
    class Redis:
        async def zcount(self, *args): return 1
        async def set(self, *args, **kwargs): return None
        async def xadd(self, *args, **kwargs): return "1-0"
        async def brpop(self, *args, **kwargs):
            return "key", '{"ok": true, "result": {"validation_version": "image-v1", "bytes": 74, "mime": "image/png", "width": 4, "height": 3, "sha256": "' + "a" * 64 + '", "etags": [{"n": 1, "etag": "e"}]}}'

    router = ValidatedTransferRouter(StrictValidatedTransferExecutor(_settings(tmp_path)), redis=Redis())
    result = await router.execute(_request())
    assert result["route"] == "remote"
    assert result["validation_version"] == "image-v1"


@pytest.mark.asyncio
async def test_router_never_falls_back_after_worker_validation_rejection(tmp_path):
    class Redis:
        async def zcount(self, *args): return 1
        async def set(self, *args, **kwargs): return None
        async def xadd(self, *args, **kwargs): return "1-0"
        async def brpop(self, *args, **kwargs): return "key", '{"ok": false, "error_kind": "validation", "error": "bad image"}'

    with pytest.raises(ValidatedTransferError) as error:
        await ValidatedTransferRouter(StrictValidatedTransferExecutor(_settings(tmp_path)), redis=Redis()).execute(_request())
    assert error.value.validation is True


@pytest.mark.asyncio
async def test_router_fallback_claim_uses_platform_owner_status_and_deletes_lease(tmp_path):
    class Pipeline:
        async def watch(self, *keys): return None
        async def get(self, key): return self.redis.values.get(key)
        async def unwatch(self): return None
        def multi(self): return None
        def set(self, key, value, ex=None): self.pending.append(("set", key, value))
        def delete(self, key): self.pending.append(("delete", key))
        async def execute(self):
            for action in self.pending:
                if action[0] == "set":
                    self.redis.values[action[1]] = action[2]
                else:
                    self.redis.values.pop(action[1], None)

        async def __aenter__(self):
            self.pending = []
            return self

        async def __aexit__(self, *args):
            return None

        def __init__(self, redis):
            self.redis = redis

    class Redis:
        def __init__(self): self.values = {"worker:task:status:task-1": "claimed", "worker:task:lease:task-1": "lease"}
        async def zcount(self, *args): return 1
        async def set(self, *args, **kwargs): return None
        async def xadd(self, *args, **kwargs): return "1-0"
        async def brpop(self, *args, **kwargs): return None
        def pipeline(self, **kwargs): return Pipeline(self)
        async def get(self, key): return self.values.get(key)
        async def mget(self, *keys): return [self.values.get(key) for key in keys]

    redis = Redis()
    router = ValidatedTransferRouter(StrictValidatedTransferExecutor(_settings(tmp_path)), redis=redis, instance_id="sgp-1")
    assert await router._claim_fallback("task-1") is True
    assert redis.values["worker:task:status:task-1"] == "fallback_claimed"
    assert redis.values["worker:task:owner:task-1"] == "fallback:sgp-1"
    assert "worker:task:lease:task-1" not in redis.values
    assert await router._ownership_check("task-1")() is True


@pytest.mark.asyncio
async def test_shared_router_factory_reuses_executor_and_enforces_capacity_across_callers(tmp_path):
    image = _png_bytes()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    get_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_calls
        if request.method == "GET":
            get_calls += 1
            if get_calls == 1:
                first_started.set()
                await release_first.wait()
            return httpx.Response(200, content=image)
        return httpx.Response(200, headers={"etag": "e"})

    settings = _settings(tmp_path).with_limits(local_concurrency=1, local_queue_limit=0)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reset_shared_validated_transfer_routers()
    first = get_shared_validated_transfer_router(settings=settings, redis=None, client=client)
    second = get_shared_validated_transfer_router(settings=settings, redis=None, client=client)
    assert first is second
    assert first.executor is second.executor

    first_task = asyncio.create_task(first.execute(_request()))
    await first_started.wait()
    with pytest.raises(ValidatedTransferError, match="queue is saturated"):
        await second.execute(_request())
    release_first.set()
    await first_task
    await client.aclose()
    reset_shared_validated_transfer_routers()


@pytest.mark.asyncio
async def test_shared_router_factory_keys_wait_timeout(tmp_path):
    settings = _settings(tmp_path)
    reset_shared_validated_transfer_routers()
    first = get_shared_validated_transfer_router(settings=settings, redis=None, wait_timeout=3)
    same = get_shared_validated_transfer_router(settings=settings, redis=None, wait_timeout=3)
    different = get_shared_validated_transfer_router(settings=settings, redis=None, wait_timeout=9)
    assert first is same
    assert first is not different
    assert first.executor is different.executor
    assert first.wait_timeout == 3
    assert different.wait_timeout == 9
    reset_shared_validated_transfer_routers()


@pytest.mark.asyncio
async def test_router_claims_fallback_after_worker_transient_result(tmp_path):
    class Pipeline:
        def __init__(self, redis):
            self.redis = redis

        async def __aenter__(self):
            self.pending = []
            return self

        async def __aexit__(self, *args):
            return None

        async def watch(self, *keys):
            return None

        async def get(self, key):
            return self.redis.values.get(key)

        async def unwatch(self):
            return None

        def multi(self):
            return None

        def set(self, key, value, ex=None):
            self.pending.append(("set", key, value))

        def delete(self, key):
            self.pending.append(("delete", key))
        async def execute(self):
            for action in self.pending:
                if action[0] == "set":
                    self.redis.values[action[1]] = action[2]
                else:
                    self.redis.values.pop(action[1], None)

    class Redis:
        def __init__(self): self.values = {}
        async def zcount(self, *args): return 1
        async def set(self, key, value, **kwargs): self.values[key] = value
        async def xadd(self, *args, **kwargs): return "1-0"
        async def brpop(self, *args, **kwargs): return "key", '{"ok": false, "error_kind": "transient"}'
        def pipeline(self, **kwargs): return Pipeline(self)
        async def get(self, key): return self.values.get(key)
        async def mget(self, *keys): return [self.values.get(key) for key in keys]

    image = _png_bytes()
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=image) if request.method == "GET" else httpx.Response(200, headers={"etag": "e"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await ValidatedTransferRouter(
        StrictValidatedTransferExecutor(_settings(tmp_path), client=client), redis=Redis()
    ).execute(_request())
    assert result["route"] == "local_worker_failure"
    await client.aclose()


@pytest.mark.asyncio
async def test_validated_delegated_transfer_without_redis_uses_shared_strict_local_router(monkeypatch, tmp_path):
    image = _png_bytes()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=image)
            if request.method == "GET"
            else httpx.Response(200, headers={"etag": "strict-etag"})
        )
    )

    def factory(*, settings, redis, wait_timeout, **kwargs):
        return get_shared_validated_transfer_router(
            settings=settings, redis=redis, client=client, wait_timeout=wait_timeout
        )

    monkeypatch.setenv("LIBTV_PLATFORM_SOURCE_HOSTS", "source.example")
    monkeypatch.setenv("LIBTV_VALIDATED_TRANSFER_TARGET_HOSTS", "target.example")
    monkeypatch.setattr("litellm.llms.libtv.validated_transfer.get_shared_validated_transfer_router", factory)
    reset_shared_validated_transfer_routers()

    result = await ValidatedDelegatedTransfer(None).transfer(
        "https://source.example/image.png?X-Amz-Signature=abc&X-Amz-Expires=60",
        len(image),
        [{"n": 1, "url": "https://target.example/part-1"}],
        source_sha256=hashlib.sha256(image).hexdigest(),
        hard_cap=1024,
        part_size=1024,
    )

    assert result == [{"n": 1, "etag": "strict-etag"}]
    await client.aclose()
    reset_shared_validated_transfer_routers()
