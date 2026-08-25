from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Optional, Union

import httpx
import pytest
import redis.exceptions as redis_exceptions

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.causyn import CausynVideoHandler
from litellm.llms.causyn import handler as causyn_module
from litellm.llms.causyn.topaz import (
    TopazAccount,
    TopazAccountPool,
    TopazAdvance,
    TopazIndeterminateError,
    TopazVideoAdapter,
)
from litellm.llms.libtv.billing_outbox import CAUSYN_BILLING_STREAM_KEY, enqueue_causyn_billing
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.libtv.transfer import result_key, status_key
from litellm.llms.libtv import video_generate as video_generate_module
from litellm.llms.libtv.video_generate import VideoGenerateSettings
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider

TASK_ID = "0123456789abcdef0123456789abcdef"
VIDEO_ID = f"causyn_{TASK_ID}"
REFERENCE_URL = "https://source.example/reference.png"
STAGING_URL = "https://target.example/staging/video.mp4?signature=private"
SETTINGS = VideoGenerateSettings(
    source_hosts=frozenset({"source.example"}),
    target_hosts=frozenset({"target.example"}),
)


class FakeRedis:
    def __init__(self) -> None:
        self.alive: list[str] = ["worker-1"]
        self.capacity: dict[str, int] = {}
        self.status: dict[str, str] = {}
        self.results: dict[str, str] = {}
        self.billing_events: list[dict[str, object]] = []
        self.calls: list[tuple[str, object]] = []
        self.xadd_error: Exception | None = None

    async def zrangebyscore(self, key: str, minimum: float, maximum: str) -> list[str]:
        self.calls.append(("zrangebyscore", key))
        return self.alive

    async def hgetall(self, key: str) -> dict[str, int]:
        self.calls.append(("hgetall", key))
        return self.capacity

    async def set(self, key: str, value: str, **kwargs: object) -> bool:
        self.calls.append(("set", (key, value, kwargs)))
        self.status[key] = value
        return True

    async def xadd(self, key: str, values: dict[str, str]) -> str:
        self.calls.append(("xadd", (key, values)))
        if self.xadd_error is not None:
            raise self.xadd_error
        return "1-0"

    async def eval(self, script: str, numkeys: int, *args: object) -> list[str]:
        self.calls.append(("eval", (script, numkeys, args)))
        if numkeys == 2 and args[0] == CAUSYN_BILLING_STREAM_KEY:
            stream_key, marker_key, payload, event_id = args
            assert isinstance(stream_key, str)
            assert isinstance(marker_key, str)
            assert isinstance(payload, str)
            assert isinstance(event_id, str)
            if marker_key in self.status:
                return ["existing", self.status[marker_key]]
            self.status[marker_key] = event_id
            self.billing_events.append(json.loads(payload))
            return ["enqueued", event_id]
        stream_key, marker_key, payload, event_id = args
        assert isinstance(marker_key, str)
        if marker_key in self.status:
            return ["existing", self.status[marker_key]]
        if self.xadd_error is not None:
            raise self.xadd_error
        assert isinstance(payload, str)
        assert isinstance(stream_key, str)
        assert isinstance(event_id, str)
        self.status[marker_key] = event_id
        self.calls.append(("xadd", (stream_key, {"payload": payload})))
        return ["enqueued", event_id]

    async def delete(self, key: str) -> int:
        self.calls.append(("delete", key))
        self.status.pop(key, None)
        return 1

    async def get(self, key: str) -> str | None:
        self.calls.append(("get", key))
        return self.status.get(key)

    async def lindex(self, key: str, index: int) -> str | None:
        self.calls.append(("lindex", (key, index)))
        return self.results.get(key)

    def envelope(self) -> dict[str, Any]:
        xadd = next(value for name, value in self.calls if name == "xadd")
        _, values = xadd
        return json.loads(values["payload"])


class _TransactionalPipeline:
    def __init__(self, redis: TransactionalFakeRedis) -> None:
        self.redis = redis
        self.commands: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def __aenter__(self) -> _TransactionalPipeline:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def watch(self, *_: str) -> None:
        return None

    async def unwatch(self) -> None:
        return None

    async def get(self, key: str) -> str | None:
        return self.redis.status.get(key)

    def multi(self) -> None:
        return None

    def set(self, key: str, value: str, **kwargs: object) -> _TransactionalPipeline:
        self.commands.append(("set", (key, value), kwargs))
        return self

    def xadd(self, key: str, values: dict[str, str]) -> _TransactionalPipeline:
        self.commands.append(("xadd", (key, values), {}))
        return self

    async def execute(self) -> list[object]:
        if self.redis.watch_error:
            raise redis_exceptions.WatchError()
        if self.redis.execute_error is not None:
            raise self.redis.execute_error
        results: list[object] = []
        for name, args, kwargs in self.commands:
            if name == "set":
                results.append(await self.redis.set(args[0], args[1], **kwargs))
            else:
                results.append(await self.redis.xadd(args[0], args[1]))
        self.redis.pipeline_commands.append(tuple(name for name, _, _ in self.commands))
        return results


class TransactionalFakeRedis(FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.execute_error: Exception | None = None
        self.watch_error = False
        self.pipeline_commands: list[tuple[str, ...]] = []

    def pipeline(self, **_: object) -> _TransactionalPipeline:
        return _TransactionalPipeline(self)


class MetadataFailureRedis(FakeRedis):
    async def set(self, key: str, value: str, **kwargs: object) -> bool:
        if key.startswith("worker:task:metadata:"):
            self.calls.append(("set", (key, value, kwargs)))
            return False
        return await super().set(key, value, **kwargs)


class ExplodingRedis(FakeRedis):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation
        self.secret = "redis://user:secret@internal.example:6379/0"

    async def get(self, key: str) -> str | None:
        if self.operation == "get":
            raise RuntimeError(self.secret)
        return await super().get(key)

    async def set(self, key: str, value: str, **kwargs: object) -> bool:
        if self.operation == "set":
            raise RuntimeError(self.secret)
        return await super().set(key, value, **kwargs)

    async def eval(self, script: str, numkeys: int, *args: object) -> list[str]:
        if self.operation == "eval":
            raise RuntimeError(self.secret)
        return await super().eval(script, numkeys, *args)


@dataclass(frozen=True)
class FakeResponse:
    status_code: int
    content: bytes


class FakeContentGet:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse(status_code=200, content=b"video-bytes")
        self.calls: list[tuple[str, Optional[Union[float, httpx.Timeout]], bool]] = []

    async def __call__(
        self,
        url: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        follow_redirects: bool,
    ) -> FakeResponse:
        self.calls.append((url, timeout, follow_redirects))
        return self.response


class FakeBillingPersistence:
    def __init__(
        self,
        *,
        stored_usage: dict[str, object] | None = None,
        billed: bool = True,
        store_error: Exception | None = None,
        lookup_error: Exception | None = None,
        billing_error: Exception | None = None,
    ) -> None:
        self.stored_usage = stored_usage
        self.billed = billed
        self.store_error = store_error
        self.lookup_error = lookup_error
        self.billing_error = billing_error
        self.store_calls: list[tuple[str, float, str | None]] = []
        self.lookup_calls: list[str] = []
        self.billing_calls: list[tuple[str, float, float]] = []

    async def store_video_task_usage(
        self,
        billing_key: str,
        duration_seconds: float,
        video_resolution: str | None,
    ) -> None:
        self.store_calls.append((billing_key, duration_seconds, video_resolution))
        if self.store_error is not None:
            raise self.store_error

    async def get_video_task_usage(self, billing_key: str) -> dict[str, object] | None:
        self.lookup_calls.append(billing_key)
        if self.lookup_error is not None:
            raise self.lookup_error
        return self.stored_usage

    async def mark_video_billed(
        self,
        billing_key: str,
        duration_seconds: float,
        response_cost: float,
    ) -> bool:
        self.billing_calls.append((billing_key, duration_seconds, response_cost))
        if self.billing_error is not None:
            raise self.billing_error
        return self.billed


@pytest.fixture(autouse=True)
def enable_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", "true")
    monkeypatch.delenv("OH_DRAMA_INTERNAL_VIDEO_ENABLED", raising=False)


def handler(
    redis: FakeRedis,
    content_get: FakeContentGet | None = None,
    persistence: FakeBillingPersistence | None = None,
    billing_enqueue: Any | None = None,
    topaz_adapter: object | None = None,
) -> CausynVideoHandler:
    async def _enqueue(redis: object, event: object) -> bool:
        if billing_enqueue is not None:
            return await billing_enqueue(redis, event)
        # The Redis/Lua outbox contract has dedicated integration coverage.
        return True

    return CausynVideoHandler(
        redis_factory=lambda: redis,
        settings_factory=lambda: SETTINGS,
        content_get=content_get or FakeContentGet(),
        refresh_staging_url=lambda task_id, timeout: _refresh_url(task_id),
        task_id_factory=lambda: TASK_ID,
        clock=lambda: 2_000_000_000.0,
        persistence_factory=lambda: persistence,
        billing_enqueue=_enqueue,
        topaz_adapter=topaz_adapter,
    )


async def _refresh_url(task_id: str) -> str:
    assert task_id == TASK_ID
    return STAGING_URL


def create_kwargs(**optional_overrides: object) -> dict[str, object]:
    optional_params: dict[str, object] = {
        "seconds": "5",
        "resolution": "768x512",
        "aspect_ratio": "3:2",
        "reference_images": [REFERENCE_URL],
        "generate_audio": True,
        "seed": 7,
        "model_info": {"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1},
    }
    optional_params.update(optional_overrides)
    return {
        "model": "causyn-1.0",
        "prompt": "animate the reference",
        "api_key": None,
        "api_base": None,
        "optional_params": optional_params,
        "logging_obj": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": "768x512", "size": "768x512"},
    ],
)
async def test_async_create_accepts_resolution_and_compatible_size_aliases(
    overrides: dict[str, object],
) -> None:
    redis = FakeRedis()

    response = await handler(redis).avideo_generation(**create_kwargs(**overrides))

    assert response.status == "queued"
    assert redis.envelope()["request"]["resolution"] == "768x512"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": "512x768"},
        {"resolution": 768},
        {"resolution": None, "size": None},
        {"resolution": None, "size": "768x512"},
        {"resolution": None, "size": 768},
        {"resolution": "768x512", "size": "512x768"},
        {"unknown_parameter": "unexpected"},
    ],
)
async def test_async_create_rejects_invalid_resolution_contract(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(CustomLLMError) as exc_info:
        await handler(FakeRedis()).avideo_generation(**create_kwargs(**overrides))

    assert exc_info.value.status_code == 400


def completed_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "validation_version": "video-v1",
        "staging_key": f"staging/video-tasks/{TASK_ID}.mp4",
        "staging_url": STAGING_URL,
        "etag": "etag-1",
        "bytes": 1234,
        "content_type": "video/mp4",
        "duration_seconds": 5.0,
        "width": 768,
        "height": 512,
        "sha256": "a" * 64,
        "ark_task_id": "internal-provider-task-id",
        "adopted": False,
        "debug_url": "https://internal.example/debug",
    }
    result.update(overrides)
    return result


def set_status(redis: FakeRedis, status: str, result: dict[str, object] | None = None) -> None:
    redis.status[status_key(TASK_ID)] = status
    if result is not None:
        redis.results[result_key(TASK_ID)] = json.dumps({"ok": True, "result": result})


def set_billing_metadata(
    redis: FakeRedis,
    *,
    duration_seconds: float = 5.0,
    video_resolution: str = "768x512",
    pricing: dict[str, object] | None = None,
    attribution: dict[str, object] | None = None,
    version: str | None = None,
    raw: str | None = None,
) -> None:
    if raw is None:
        raw = json.dumps(
            {
                "version": version or causyn_module.CAUSYN_BILLING_METADATA_VERSION,
                "duration_seconds": duration_seconds,
                "video_resolution": video_resolution,
                "pricing": pricing
                or {
                    "model": "causyn-1.0",
                    "id": "causyn-price-v1",
                    "output_cost_per_second_768x512": 0.1,
                },
                "attribution": attribution
                or {
                    "api_key": None,
                    "team_id": None,
                    "user_id": None,
                    "organization_id": None,
                },
            }
        )
    redis.status[f"worker:task:metadata:{TASK_ID}"] = raw


class FakeTopazAdapter:
    def __init__(self) -> None:
        self.advance_calls = 0
        self.content_calls = 0

    async def advance(
        self, task_id: str, source_url_factory: object, *, validate_source: object = None
    ) -> TopazAdvance:
        assert task_id == TASK_ID
        self.advance_calls += 1
        return TopazAdvance(
            "completed", provider_task_id="topaz-task", result_url="https://topaz/result.mp4", attempt_index=1
        )

    async def content(self, task_id: str) -> bytes:
        assert task_id == TASK_ID
        self.content_calls += 1
        return b"upscaled-video"


class FailedTopazAdapter:
    async def advance(
        self, task_id: str, source_url_factory: object, *, validate_source: object = None
    ) -> TopazAdvance:
        assert task_id == TASK_ID
        return TopazAdvance("failed", attempt_index=4)

    async def content(self, task_id: str) -> bytes:
        assert task_id == TASK_ID
        return b"unreachable"


class IndeterminateTopazAdapter:
    def __init__(self, operation: str) -> None:
        self.operation = operation

    async def advance(
        self, task_id: str, source_url_factory: object, *, validate_source: object = None
    ) -> TopazAdvance:
        assert task_id == TASK_ID
        if self.operation == "status":
            raise TopazIndeterminateError("Topaz submission state is indeterminate")
        return TopazAdvance("completed", provider_task_id="topaz-task", attempt_index=1)

    async def content(self, task_id: str) -> bytes:
        assert task_id == TASK_ID
        raise TopazIndeterminateError("Topaz submission state is indeterminate")


def set_v2_billing_metadata(redis: FakeRedis, *, requested_resolution: str = "2k") -> None:
    redis.status[f"worker:task:metadata:{TASK_ID}"] = json.dumps(
        {
            "version": causyn_module.CAUSYN_BILLING_METADATA_VERSION_V2,
            "duration_seconds": 5.0,
            "source_resolution": "768x512",
            "requested_resolution": requested_resolution,
            "pricing": {
                "model": "causyn-1.0",
                "id": "causyn-price-v2",
                "output_cost_per_second_768x512": 0.1,
                "output_cost_per_second_2k": 0.6,
            },
            "attribution": {
                "api_key": None,
                "team_id": None,
                "user_id": None,
                "organization_id": None,
            },
        }
    )


@pytest.mark.asyncio
async def test_async_create_2k_keeps_worker_768_and_persists_strict_v2_metadata() -> None:
    redis = FakeRedis()
    response = await handler(redis).avideo_generation(
        **create_kwargs(
            resolution="2k",
            model_info={
                "id": "causyn-price-v2",
                "output_cost_per_second_768x512": 0.1,
                "output_cost_per_second_2k": 0.6,
            },
        )
    )

    assert response.size == "2k"
    assert redis.envelope()["request"]["resolution"] == "768x512"
    metadata = json.loads(redis.status[f"worker:task:metadata:{TASK_ID}"])
    assert metadata["version"] == causyn_module.CAUSYN_BILLING_METADATA_VERSION_V2
    assert metadata["source_resolution"] == "768x512"
    assert metadata["requested_resolution"] == "2k"
    assert "video_resolution" not in metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolution", "model_info"),
    [
        ("768x512", {"id": "only-2k", "output_cost_per_second_2k": 0.6}),
        ("2k", {"id": "only-native", "output_cost_per_second_768x512": 0.1}),
    ],
)
async def test_create_rejects_pricing_without_requested_resolution_rate(
    resolution: str,
    model_info: dict[str, object],
) -> None:
    redis = FakeRedis()

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs(resolution=resolution, model_info=model_info))

    assert exc_info.value.status_code == 503
    assert redis.status == {}


@pytest.mark.asyncio
async def test_legacy_v1_metadata_with_only_2k_rate_is_retryable_not_assertion_error() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_billing_metadata(
        redis,
        pricing={"model": "causyn-1.0", "id": "only-2k", "output_cost_per_second_2k": 0.6},
    )

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_status(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 503
    assert "only-2k" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set"])
async def test_handler_redis_failures_are_stable_retryable_errors(operation: str) -> None:
    redis = ExplodingRedis(operation)

    with pytest.raises(CustomLLMError) as exc_info:
        if operation == "get":
            await handler(redis).avideo_status(VIDEO_ID, None, None, {}, None)
        else:
            await handler(redis).avideo_generation(**create_kwargs())

    assert exc_info.value.status_code == 503
    assert "redis://" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_handler_adapter_ordinary_failure_is_stable_retryable_error() -> None:
    class ExplodingAdapter:
        async def advance(self, *_args: object, **_kwargs: object) -> TopazAdvance:
            raise RuntimeError("redis://user:secret@internal.example:6379/0")

        async def content(self, *_args: object, **_kwargs: object) -> bytes:
            raise RuntimeError("redis://user:secret@internal.example:6379/0")

    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_v2_billing_metadata(redis)

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, topaz_adapter=ExplodingAdapter()).avideo_status(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 503
    assert "redis://" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_handler_state_store_eval_failure_is_stable_retryable_error() -> None:
    class StateClient:
        async def aensure_libtv_url(
            self,
            kind: str,
            url: str,
            data: bytes | None,
            default_name: str,
            *,
            allow_cache: bool = True,
        ) -> str:
            return "https://libtv.example/imported.mp4"

        async def acreate(
            self,
            model_key: str,
            vendor: str,
            task_type: str,
            params: dict[str, object],
            project_name: str,
            *,
            allow_cached_project_retry: bool,
            paid_submission: bool,
        ) -> dict[str, object]:
            return {"task_id": "topaz-state"}

        async def apoll_once(self, task_id: str, task_type: str) -> dict[str, object]:
            return {"status": 1}

    redis = ExplodingRedis("eval")
    set_status(redis, "done", completed_result())
    set_v2_billing_metadata(redis)
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: StateClient(),
        redis_factory=lambda: redis,
        now=lambda: 100.0,
    )

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, topaz_adapter=adapter).avideo_status(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 503
    assert "redis://" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_2k_status_and_content_are_idempotent_and_bill_only_0_6_per_second() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_v2_billing_metadata(redis)
    topaz = FakeTopazAdapter()
    provider = handler(redis, billing_enqueue=enqueue_causyn_billing, topaz_adapter=topaz)
    first, second, content = await asyncio.gather(
        provider.avideo_status(VIDEO_ID, None, None, {}, None),
        provider.avideo_status(VIDEO_ID, None, None, {}, None),
        provider.avideo_content(VIDEO_ID, None, None, {}, None),
    )

    assert first.status == second.status == "completed"
    assert first.size == second.size == "2k"
    assert first.object_store_result is None
    assert content == b"upscaled-video"
    assert len(redis.billing_events) == 1
    assert redis.billing_events[0]["request_id"] == f"causyn:{TASK_ID}"
    assert redis.billing_events[0]["provider_task_id"] == TASK_ID
    assert redis.billing_events[0]["response_cost"] == pytest.approx(3.0)
    assert "topaz-task" not in json.dumps(redis.billing_events[0])
    assert topaz.advance_calls == 3
    assert topaz.content_calls == 1


@pytest.mark.asyncio
async def test_2k_final_failure_uses_product_neutral_public_error() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_v2_billing_metadata(redis)

    response = await handler(redis, topaz_adapter=FailedTopazAdapter()).avideo_status(VIDEO_ID, None, None, {}, None)

    serialized = json.dumps(response.model_dump())
    assert response.status == "failed"
    assert response.error == {"message": "causyn 2K processing failed", "kind": "provider"}
    assert "causyn 2K processing" in serialized
    assert all(term not in serialized.lower() for term in ("topaz", "upscale", "超分"))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["status", "content"])
async def test_2k_indeterminate_public_errors_hide_provider_implementation(
    operation: str,
) -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_v2_billing_metadata(redis)
    provider = handler(redis, topaz_adapter=IndeterminateTopazAdapter(operation))

    with pytest.raises(CustomLLMError) as exc_info:
        if operation == "status":
            await provider.avideo_status(VIDEO_ID, None, None, {}, None)
        else:
            await provider.avideo_content(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 503
    public_error = str(exc_info.value)
    assert "causyn 2K processing indeterminate" in public_error
    assert all(term not in public_error.lower() for term in ("topaz", "upscale", "超分"))


@pytest.mark.asyncio
async def test_v2_native_status_keeps_0_1_per_second_outer_spend() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_v2_billing_metadata(redis, requested_resolution="768x512")

    response = await handler(redis, billing_enqueue=enqueue_causyn_billing).avideo_status(
        VIDEO_ID, None, None, {}, None
    )

    assert response.size == "768x512"
    assert response._hidden_params["response_cost"] == pytest.approx(0.5)
    assert len(redis.billing_events) == 1
    assert redis.billing_events[0]["request_id"] == f"causyn:{TASK_ID}"
    assert redis.billing_events[0]["response_cost"] == pytest.approx(0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("generate_audio", [True, False])
async def test_async_create_preserves_generate_audio_in_worker_envelope(generate_audio: bool) -> None:
    redis = FakeRedis()
    response = await handler(redis).avideo_generation(**create_kwargs(generate_audio=generate_audio))

    assert isinstance(response, VideoObject)
    decoded = decode_video_id_with_provider(response.id)
    assert decoded["video_id"] == TASK_ID
    assert decoded["model_id"] == "causyn-1-0"
    assert response.status == "queued"
    assert response.model == "causyn-1.0"
    assert response.seconds == "5"
    assert response.size == "768x512"
    assert response.usage == {"duration_seconds": 5, "video_resolution": "768x512"}
    assert response._hidden_params["response_cost"] == 0.0
    assert redis.envelope() == {
        "type": "video_generate",
        "task_id": TASK_ID,
        "deadline_ts": 2_000_001_800.0,
        "model": "causyn-1.0",
        "request": {
            "prompt": "animate the reference",
            "duration_seconds": 5,
            "resolution": "768x512",
            "ratio": "3:2",
            "generate_audio": generate_audio,
            "seed": 7,
            "references": [{"role": "reference", "media_type": "image", "url": REFERENCE_URL}],
        },
    }


@pytest.mark.asyncio
async def test_async_create_defaults_generate_audio_on_in_worker_envelope() -> None:
    redis = FakeRedis()
    params = create_kwargs()["optional_params"]
    assert isinstance(params, dict)
    params = dict(params)
    params.pop("generate_audio")

    await handler(redis).avideo_generation(**create_kwargs(**params))

    assert redis.envelope()["request"]["generate_audio"] is True


@pytest.mark.asyncio
async def test_worker_envelope_rejects_non_boolean_generate_audio() -> None:
    redis = FakeRedis()
    await handler(redis).avideo_generation(**create_kwargs())
    envelope = redis.envelope()
    envelope["request"]["generate_audio"] = "true"

    with pytest.raises(video_generate_module.VideoGenerateError) as exc_info:
        video_generate_module._validate_shape(envelope)

    assert exc_info.value.code == "invalid_params"


def test_worker_envelope_rejects_explicit_null_staging_upload() -> None:
    envelope = {
        "task_id": TASK_ID,
        "model": "causyn-1.0",
        "deadline_ts": 2_000_001_800.0,
        "request": {"prompt": "animate", "duration_seconds": 5, "resolution": "768x512"},
        "staging_upload": None,
    }

    with pytest.raises(video_generate_module.VideoGenerateError) as exc_info:
        video_generate_module._validate_shape(envelope)

    assert exc_info.value.code == "invalid_params"


@pytest.mark.asyncio
async def test_async_create_persists_billing_metadata_without_usage_db() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence()

    response = await handler(redis, persistence=persistence).avideo_generation(
        **create_kwargs(model_info={"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1})
    )

    assert response.status == "queued"
    assert persistence.store_calls == []
    assert json.loads(redis.status[f"worker:task:metadata:{TASK_ID}"]) == {
        "attribution": {
            "api_key": None,
            "organization_id": None,
            "team_id": None,
            "user_id": None,
        },
        "duration_seconds": 5.0,
        "pricing": {
            "id": "causyn-price-v1",
            "model": "causyn-1.0",
            "output_cost_per_second_768x512": 0.1,
        },
        "version": causyn_module.CAUSYN_BILLING_METADATA_VERSION_V2,
        "source_resolution": "768x512",
        "requested_resolution": "768x512",
    }


@pytest.mark.asyncio
async def test_async_create_rejects_incomplete_pricing_before_enqueue() -> None:
    redis = FakeRedis()

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs(model_info={"output_cost_per_second_768x512": 0.1}))

    assert exc_info.value.status_code == 503
    assert redis.status == {}
    assert redis.calls == []


@pytest.mark.asyncio
async def test_async_create_strips_only_outer_pricing_and_owner_whitespace() -> None:
    redis = FakeRedis()

    await handler(redis).avideo_generation(
        **create_kwargs(
            model_info={"id": "  causyn-price-v1  ", "output_cost_per_second_768x512": 0.1},
            metadata={"user_api_key_team_id": "  team-1  "},
        )
    )

    metadata = json.loads(redis.status[f"worker:task:metadata:{TASK_ID}"])
    assert metadata["pricing"]["id"] == "causyn-price-v1"
    assert metadata["attribution"]["team_id"] == "team-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_overrides",
    [
        {"model_info": {"id": "   ", "output_cost_per_second_768x512": 0.1}},
        {"metadata": {"user_api_key_team_id": "   "}},
    ],
)
async def test_async_create_rejects_blank_pricing_or_owner_id_before_enqueue(
    create_overrides: dict[str, object],
) -> None:
    redis = FakeRedis()

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs(**create_overrides))

    assert exc_info.value.status_code == 503
    assert redis.status == {}
    assert redis.calls == []


@pytest.mark.asyncio
async def test_async_create_survives_usage_persistence_failure() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(store_error=RuntimeError("db unavailable"))

    response = await handler(redis, persistence=persistence).avideo_generation(**create_kwargs())

    assert response.status == "queued"
    assert persistence.store_calls == []


@pytest.mark.asyncio
async def test_async_create_commits_status_metadata_and_stream_in_one_pipeline() -> None:
    redis = TransactionalFakeRedis()

    response = await handler(redis).avideo_generation(
        **create_kwargs(model_info={"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1})
    )

    assert response.status == "queued"
    assert redis.pipeline_commands == [("set", "set", "xadd")]
    assert redis.status[status_key(TASK_ID)] == "queued"
    assert f"worker:task:metadata:{TASK_ID}" in redis.status
    assert redis.envelope()["request"]["duration_seconds"] == 5


@pytest.mark.asyncio
async def test_async_create_pipeline_failure_does_not_leave_partial_task() -> None:
    redis = TransactionalFakeRedis()
    redis.execute_error = RuntimeError("transaction failed")

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs())

    assert exc_info.value.status_code == 503
    assert status_key(TASK_ID) not in redis.status
    assert f"worker:task:metadata:{TASK_ID}" not in redis.status
    assert not any(name == "xadd" for name, _ in redis.calls)


@pytest.mark.asyncio
async def test_async_create_fallback_metadata_failure_rolls_back_status() -> None:
    redis = MetadataFailureRedis()

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs())

    assert exc_info.value.status_code == 503
    assert status_key(TASK_ID) not in redis.status
    assert f"worker:task:metadata:{TASK_ID}" not in redis.status
    assert not any(name == "xadd" for name, _ in redis.calls)


@pytest.mark.asyncio
async def test_async_create_watch_conflict_is_an_idempotent_duplicate() -> None:
    redis = TransactionalFakeRedis()
    redis.watch_error = True

    response = await handler(redis).avideo_generation(**create_kwargs())

    assert response.status == "queued"
    assert status_key(TASK_ID) not in redis.status
    assert not any(name in {"set", "xadd"} for name, _ in redis.calls)


@pytest.mark.asyncio
async def test_public_litellm_video_api_dispatches_to_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    provider = handler(redis)
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "causyn", "custom_handler": provider}])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "causyn"])

    response = await litellm.avideo_generation(
        model="causyn/causyn-1.0",
        prompt="animate the reference",
        model_info={"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1},
        seconds="5",
        size="768x512",
        aspect_ratio="3:2",
        reference_images=[REFERENCE_URL],
    )

    assert isinstance(response, VideoObject)
    assert decode_video_id_with_provider(response.id)["video_id"] == TASK_ID
    assert redis.envelope()["request"]["references"][0]["url"] == REFERENCE_URL


def _install_public_causyn_handler(monkeypatch: pytest.MonkeyPatch, provider: CausynVideoHandler) -> None:
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "causyn", "custom_handler": provider}])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "causyn"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"extra_body": {"resolution": "768x512"}},
        {"size": "768x512"},
        {"size": "768x512", "extra_body": {"resolution": "768x512"}},
        {"size": "768x512", "extra_body": {"size": "512x768"}},
    ],
)
async def test_public_litellm_video_api_merges_resolution_contract(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    redis = FakeRedis()
    _install_public_causyn_handler(monkeypatch, handler(redis))

    response = await litellm.avideo_generation(
        model="causyn/causyn-1.0",
        prompt="animate the reference",
        model_info={"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1},
        seconds="5",
        aspect_ratio="3:2",
        reference_images=[REFERENCE_URL],
        **overrides,
    )

    assert isinstance(response, VideoObject)
    assert redis.envelope()["request"]["resolution"] == "768x512"
    assert "size" not in redis.envelope()["request"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"size": "768x512", "extra_body": {"resolution": "512x768"}},
        {"extra_body": {"resolution": None}},
        {"extra_body": {"unknown_parameter": None}},
        {"extra_body": {"resolution": "512x768"}},
        {"resolution": 768},
        {"size": 768},
    ],
)
async def test_public_litellm_video_api_rejects_resolution_contract(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    redis = FakeRedis()
    _install_public_causyn_handler(monkeypatch, handler(redis))

    with pytest.raises(BadRequestError) as exc_info:
        await litellm.avideo_generation(
            model="causyn/causyn-1.0",
            prompt="animate the reference",
            model_info={"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1},
            seconds="5",
            aspect_ratio="3:2",
            reference_images=[REFERENCE_URL],
            **overrides,
        )

    assert exc_info.value.status_code == 400
    assert redis.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_status", "public_status"),
    [("queued", "queued"), ("claimed", "in_progress")],
)
async def test_status_maps_nonterminal_worker_states(worker_status: str, public_status: str) -> None:
    redis = FakeRedis()
    set_status(redis, worker_status)

    response = await handler(redis).avideo_status(VIDEO_ID, None, None, {}, None)

    assert response.status == public_status
    assert response.object_store_result is None


@pytest.mark.asyncio
async def test_sync_status_bridges_to_async_contract_inside_running_loop() -> None:
    redis = FakeRedis()
    set_status(redis, "claimed")

    response = handler(redis).video_status(VIDEO_ID, None, None, {}, None)

    assert response.status == "in_progress"


@pytest.mark.asyncio
async def test_completed_status_exposes_staging_metadata_without_signed_download_url() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    response = await handler(redis).avideo_status(VIDEO_ID, None, None, {}, None)
    serialized = response.model_dump()

    assert response.status == "completed"
    assert response.usage == {"duration_seconds": 5.0, "video_resolution": "768x512"}
    assert response.object_store_result == {
        "validation_version": "video-v1",
        "staging_key": f"staging/video-tasks/{TASK_ID}.mp4",
        "etag": "etag-1",
        "bytes": 1234,
        "content_type": "video/mp4",
        "duration_seconds": 5.0,
        "width": 768,
        "height": 512,
        "sha256": "a" * 64,
    }
    assert "signature=private" not in json.dumps(serialized)
    assert "internal-provider-task-id" not in json.dumps(serialized)
    assert "internal.example" not in json.dumps(serialized)


@pytest.mark.asyncio
async def test_completed_status_rejects_staging_key_for_another_task() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result(staging_key="staging/video-tasks/other.mp4"))

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_status(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == "causyn video result is unavailable"


@pytest.mark.asyncio
async def test_completed_status_uses_worker_usage_and_durable_price() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
        billed=True,
    )
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    response = await handler(redis, persistence=persistence).avideo_status(
        VIDEO_ID,
        None,
        None,
        {},
        None,
    )

    assert response.usage == {"duration_seconds": 5.0, "video_resolution": "768x512"}
    assert response._hidden_params["response_cost"] == pytest.approx(0.5)
    assert persistence.lookup_calls == []
    assert not persistence.billing_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_kind",
    [
        "missing",
        "bad_json",
        "bad_version",
        "missing_pricing_id",
        "blank_pricing_id",
        "blank_owner_id",
        "non_finite_rate",
        "duration",
        "resolution",
        "bad_api_key",
    ],
)
async def test_completed_status_rejects_invalid_durable_metadata(invalid_kind: str) -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)
    metadata_key = f"worker:task:metadata:{TASK_ID}"
    metadata = json.loads(redis.status[metadata_key])
    if invalid_kind == "missing":
        del redis.status[metadata_key]
    elif invalid_kind == "bad_json":
        redis.status[metadata_key] = "{"  # malformed durable record
    elif invalid_kind == "bad_version":
        metadata["version"] = "causyn-video-billing-v0"
        redis.status[metadata_key] = json.dumps(metadata)
    elif invalid_kind == "missing_pricing_id":
        del metadata["pricing"]["id"]
        redis.status[metadata_key] = json.dumps(metadata)
    elif invalid_kind == "blank_pricing_id":
        metadata["pricing"]["id"] = "   "
        redis.status[metadata_key] = json.dumps(metadata)
    elif invalid_kind == "blank_owner_id":
        metadata["attribution"]["team_id"] = "   "
        redis.status[metadata_key] = json.dumps(metadata)
    elif invalid_kind == "non_finite_rate":
        metadata["pricing"]["output_cost_per_second_768x512"] = float("inf")
        redis.status[metadata_key] = json.dumps(metadata)
    elif invalid_kind == "duration":
        metadata["duration_seconds"] = 4.0
        redis.status[metadata_key] = json.dumps(metadata)
    elif invalid_kind == "resolution":
        metadata["video_resolution"] = "512x768"
        redis.status[metadata_key] = json.dumps(metadata)
    else:
        metadata["attribution"]["api_key"] = "sk-plaintext"
        redis.status[metadata_key] = json.dumps(metadata)

    enqueue_calls: list[object] = []

    async def enqueue(redis: object, event: object) -> bool:
        enqueue_calls.append(event)
        return True

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, billing_enqueue=enqueue).avideo_status(
            VIDEO_ID,
            None,
            None,
            {"model_info": {"id": "attacker", "output_cost_per_second_768x512": 99.0}},
            None,
        )

    assert exc_info.value.status_code == 503
    assert enqueue_calls == []


@pytest.mark.asyncio
async def test_repeated_completed_status_keeps_authoritative_cost_without_duplicate_event() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
        billed=False,
    )
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)
    redis.status[f"causyn:billing:enqueued:{TASK_ID}"] = f"causyn-video:{TASK_ID}"

    response = await handler(redis, persistence=persistence).avideo_status(
        VIDEO_ID,
        None,
        None,
        {"model_info": {"output_cost_per_second_768x512": 0.1}},
        None,
    )

    assert response._hidden_params["response_cost"] == pytest.approx(0.5)
    assert persistence.billing_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persistence",
    [
        None,
        FakeBillingPersistence(lookup_error=RuntimeError("db unavailable")),
        FakeBillingPersistence(stored_usage=None),
        FakeBillingPersistence(stored_usage={"duration_seconds": "5.0", "video_resolution": "768x512"}),
        FakeBillingPersistence(stored_usage={"duration_seconds": 2.0, "video_resolution": "768x512"}),
        FakeBillingPersistence(stored_usage={"duration_seconds": 9.0, "video_resolution": "768x512"}),
        FakeBillingPersistence(stored_usage={"duration_seconds": 5.0, "video_resolution": "512x768"}),
    ],
)
async def test_completed_status_ignores_invalid_persistence_usage(
    persistence: FakeBillingPersistence | None,
) -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    response = await handler(redis, persistence=persistence).avideo_status(
        VIDEO_ID,
        None,
        None,
        {"model_info": {"output_cost_per_second_768x512": 0.1}},
        None,
    )

    assert response.status == "completed"
    assert response._hidden_params["response_cost"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_completed_status_without_durable_price_is_retryable() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
    )
    set_status(redis, "done", completed_result())

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, persistence=persistence).avideo_status(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 503
    assert persistence.billing_calls == []


@pytest.mark.asyncio
async def test_content_finalizes_billing_before_download() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    captured: list[object] = []
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    async def enqueue(redis: object, event: object) -> bool:
        captured.append(event)
        return True

    content = await handler(redis, content_get, billing_enqueue=enqueue).avideo_content(
        VIDEO_ID, None, None, {}, None, timeout=12.0
    )

    assert content == b"video-bytes"
    assert len(captured) == 1
    assert captured[0].response_cost == pytest.approx(0.5)
    assert content_get.calls == [(STAGING_URL, 12.0, False)]


@pytest.mark.asyncio
async def test_content_outbox_failure_is_retryable_and_does_not_download() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    async def fail_enqueue(redis: object, event: object) -> bool:
        raise RuntimeError("redis unavailable")

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, content_get, billing_enqueue=fail_enqueue).avideo_content(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 503
    assert content_get.calls == []


@pytest.mark.asyncio
async def test_completed_status_outbox_failure_is_retryable() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
    )
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    async def fail_enqueue(redis: object, event: object) -> bool:
        raise RuntimeError("redis unavailable")

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, persistence=persistence, billing_enqueue=fail_enqueue).avideo_status(
            VIDEO_ID,
            None,
            None,
            {"model_info": {"output_cost_per_second_768x512": 0.1}},
            None,
        )

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_completed_status_retries_after_outbox_failure() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)
    outcomes = iter([False, True])

    async def enqueue(redis: object, event: object) -> bool:
        return next(outcomes)

    provider = handler(redis, billing_enqueue=enqueue)
    with pytest.raises(CustomLLMError) as exc_info:
        await provider.avideo_status(
            VIDEO_ID,
            None,
            None,
            {"model_info": {"output_cost_per_second_768x512": 0.1}},
            None,
        )
    assert exc_info.value.status_code == 503

    response = await provider.avideo_status(
        VIDEO_ID,
        None,
        None,
        {"model_info": {"output_cost_per_second_768x512": 0.1}},
        None,
    )
    assert response._hidden_params["response_cost"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_completed_status_uses_durable_proxy_attribution() -> None:
    redis = FakeRedis()
    auth_hash = "a" * 64
    captured: list[object] = []

    class Logging:
        model_call_details = {
            "litellm_params": {
                "metadata": {
                    "user_api_key_hash": auth_hash,
                    "user_api_key_team_id": "team-1",
                    "user_api_key_user_id": "user-1",
                    "user_api_key_org_id": "org-1",
                }
            }
        }

    async def enqueue(redis: object, event: object) -> bool:
        captured.append(event)
        return True

    create = create_kwargs(
        model_info={"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1},
        metadata={
            "user_api_key_hash": "b" * 64,
            "user_api_key_team_id": "attacker-team",
        },
    )
    create["logging_obj"] = Logging()
    await handler(redis, billing_enqueue=enqueue).avideo_generation(**create)
    metadata = json.loads(redis.status[f"worker:task:metadata:{TASK_ID}"])
    assert metadata["attribution"] == {
        "api_key": auth_hash,
        "team_id": "team-1",
        "user_id": "user-1",
        "organization_id": "org-1",
    }

    set_status(redis, "done", completed_result())
    response = await handler(redis, billing_enqueue=enqueue).avideo_status(
        VIDEO_ID,
        None,
        None,
        {
            "model_info": {"id": "attacker-price", "output_cost_per_second_768x512": 99.0},
            "metadata": {
                "user_api_key_hash": "c" * 64,
                "user_api_key_team_id": "attacker-team",
            },
        },
        None,
    )

    assert response._hidden_params["response_cost"] == pytest.approx(0.5)
    assert len(captured) == 1
    event = captured[0]
    assert event.api_key == auth_hash
    assert event.team_id == "team-1"
    assert event.user_id == "user-1"
    assert event.organization_id == "org-1"


@pytest.mark.asyncio
async def test_completed_status_does_not_persist_plaintext_api_key() -> None:
    redis = FakeRedis()
    captured: list[object] = []

    class Logging:
        model_call_details = {"litellm_params": {"metadata": {"user_api_key": "sk-secret"}}}

    async def enqueue(redis: object, event: object) -> bool:
        captured.append(event)
        return True

    create = create_kwargs(model_info={"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1})
    create["logging_obj"] = Logging()
    await handler(redis, billing_enqueue=enqueue).avideo_generation(**create)
    metadata = json.loads(redis.status[f"worker:task:metadata:{TASK_ID}"])
    assert metadata["attribution"] == {
        "api_key": None,
        "organization_id": None,
        "team_id": None,
        "user_id": None,
    }

    set_status(redis, "done", completed_result())
    await handler(redis, billing_enqueue=enqueue).avideo_status(VIDEO_ID, None, None, {}, None)
    assert captured[0].api_key is None


@pytest.mark.asyncio
async def test_content_returns_bytes_from_validated_platform_download_url_without_redirects() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    content = await handler(redis, content_get).avideo_content(VIDEO_ID, None, None, {}, None, timeout=12.0)

    assert content == b"video-bytes"
    assert content_get.calls == [(STAGING_URL, 12.0, False)]


@pytest.mark.asyncio
async def test_sync_content_bridges_to_async_contract_inside_running_loop() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    set_status(redis, "done", completed_result())
    set_billing_metadata(redis)

    content = handler(redis, content_get).video_content(VIDEO_ID, None, None, {}, None, timeout=12.0)

    assert content == b"video-bytes"
    assert content_get.calls == [(STAGING_URL, 12.0, False)]


@pytest.mark.asyncio
async def test_content_ignores_cached_download_url_and_uses_refreshed_allowlist_url() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    set_status(redis, "done", completed_result(staging_url="https://attacker.example/video.mp4"))
    set_billing_metadata(redis)

    content = await handler(redis, content_get).avideo_content(VIDEO_ID, None, None, {}, None)

    assert content == b"video-bytes"
    assert content_get.calls == [(STAGING_URL, None, False)]


@pytest.mark.asyncio
async def test_content_rejects_staging_key_for_another_task() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    set_status(redis, "done", completed_result(staging_key="staging/video-tasks/other.mp4"))

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, content_get).avideo_content(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 502
    assert content_get.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "", "false", "0", "no", "off", "unexpected"])
async def test_create_flag_is_strictly_fail_closed(monkeypatch: pytest.MonkeyPatch, raw: str | None) -> None:
    redis = FakeRedis()
    if raw is None:
        monkeypatch.delenv("DRAMA_INTERNAL_VIDEO_ENABLED", raising=False)
    else:
        monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", raw)

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs())

    assert exc_info.value.status_code == 403
    assert redis.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["1", "true", "TRUE", " true ", "yes", "on"])
async def test_create_flag_accepts_only_true_case_insensitively(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    redis = FakeRedis()
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", raw)

    response = await handler(redis).avideo_generation(**create_kwargs())

    assert response.status == "queued"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
async def test_create_flag_accepts_oh_alias(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    redis = FakeRedis()
    monkeypatch.delenv("DRAMA_INTERNAL_VIDEO_ENABLED", raising=False)
    monkeypatch.setenv("OH_DRAMA_INTERNAL_VIDEO_ENABLED", raw)

    response = await handler(redis).avideo_generation(**create_kwargs())

    assert response.status == "queued"


@pytest.mark.asyncio
async def test_create_flag_primary_env_has_priority_over_oh_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", "false")
    monkeypatch.setenv("OH_DRAMA_INTERNAL_VIDEO_ENABLED", "true")

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs())

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_create_flag_empty_primary_env_does_not_fall_back_to_oh_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis()
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", "")
    monkeypatch.setenv("OH_DRAMA_INTERNAL_VIDEO_ENABLED", "true")

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs())

    assert exc_info.value.status_code == 403
    assert redis.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"seconds": 5},
        {"seconds": "2"},
        {"seconds": "9"},
        {"seconds": "5.0"},
        {"size": "512x768"},
        {"aspect_ratio": "2:3"},
        {"reference_images": []},
        {"reference_images": [REFERENCE_URL] * 5},
        {"reference_images": [123]},
        {"reference_videos": [REFERENCE_URL]},
        {"generate_audio": "true"},
        {"seed": True},
        {"characters": [{"id": "character-1"}]},
        {"parameters": {"seed": 7}},
        {"unsupported_provider_param": "unexpected"},
    ],
)
async def test_create_rejects_invalid_field_types_and_hardware_envelope(overrides: dict[str, object]) -> None:
    redis = FakeRedis()

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs(**overrides))

    assert exc_info.value.status_code == 400
    assert redis.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "video_id",
    [
        "",
        "video_0123456789abcdef0123456789abcdef",
        "causyn_",
        "causyn_0123456789abcdef0123456789abcde",
        "causyn_0123456789abcdef0123456789abcdef0",
        "causyn_0123456789ABCDEF0123456789ABCDEF",
        "causyn_0123456789abcdef0123456789abcdeg",
        "causyn_../../0123456789abcdef0123456789abcdef",
    ],
)
async def test_video_id_parser_fails_closed_before_redis(video_id: str) -> None:
    redis = FakeRedis()

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_status(video_id, None, None, {}, None)

    assert exc_info.value.status_code == 400
    assert redis.calls == []


def test_standard_video_route_decoder_recognizes_opaque_causyn_id() -> None:
    decoded = decode_video_id_with_provider(VIDEO_ID)

    assert decoded == {
        "custom_llm_provider": "causyn",
        "model_id": "causyn-1-0",
        "video_id": VIDEO_ID,
    }


@pytest.mark.asyncio
async def test_unknown_task_is_not_misclassified_as_worker_failure() -> None:
    redis = FakeRedis()

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_status(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 404
    assert str(exc_info.value) == "causyn video was not found"


@pytest.mark.asyncio
async def test_worker_failure_does_not_expose_worker_message_or_invalid_code() -> None:
    redis = FakeRedis()
    redis.status[status_key(TASK_ID)] = "failed"
    redis.results[result_key(TASK_ID)] = json.dumps(
        {
            "ok": False,
            "error": "redis://user:secret@internal:6379: upstream failed",
            "error_kind": "transient",
        }
    )

    response = await handler(redis).avideo_status(VIDEO_ID, None, None, {}, None)
    serialized = json.dumps(response.model_dump())

    assert response.status == "failed"
    assert response.error == {
        "code": "worker_failed",
        "message": "video generation failed",
        "kind": "transient",
    }
    assert "secret" not in serialized
    assert "internal" not in serialized


@pytest.mark.asyncio
async def test_unclassified_enqueue_error_is_stable_and_does_not_leak_driver_details() -> None:
    redis = FakeRedis()
    redis.xadd_error = RuntimeError("redis://user:secret@internal:6379 unavailable")

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis).avideo_generation(**create_kwargs())

    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == "causyn video service unavailable"
    assert "secret" not in str(exc_info.value)
    assert "internal" not in str(exc_info.value)


def test_handler_has_no_object_store_credential_or_signing_surface() -> None:
    source = inspect.getsource(causyn_module).lower()
    constructor_parameters = inspect.signature(CausynVideoHandler).parameters

    assert not {"api_key", "access_key", "secret_key", "bucket"}.intersection(constructor_parameters)
    for forbidden in ("oss2", "boto3", "presign", "access_key", "secret_key"):
        assert forbidden not in source


@pytest.mark.parametrize(
    "method_name",
    ["avideo_generation", "avideo_status", "avideo_content", "video_generation", "video_status", "video_content"],
)
def test_handler_method_signature_matches_custom_llm_contract(method_name: str) -> None:
    handler_signature = inspect.signature(getattr(CausynVideoHandler, method_name))
    base_signature = inspect.signature(getattr(CustomLLM, method_name))

    assert list(handler_signature.parameters) == list(base_signature.parameters)
    assert [parameter.default for parameter in handler_signature.parameters.values()] == [
        parameter.default for parameter in base_signature.parameters.values()
    ]


def test_handler_is_real_custom_llm_and_sync_generation_remains_unsupported() -> None:
    provider = handler(FakeRedis())

    assert isinstance(provider, CustomLLM)
    with pytest.raises(NotImplementedError):
        provider.video_generation("causyn-1.0", "prompt", None, None, {}, None)
