from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Optional, Union

import httpx
import pytest

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.causyn import CausynVideoHandler
from litellm.llms.causyn import handler as causyn_module
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
) -> CausynVideoHandler:
    return CausynVideoHandler(
        redis_factory=lambda: redis,
        settings_factory=lambda: SETTINGS,
        content_get=content_get or FakeContentGet(),
        task_id_factory=lambda: TASK_ID,
        clock=lambda: 2_000_000_000.0,
        persistence_factory=lambda: persistence,
    )


def create_kwargs(**optional_overrides: object) -> dict[str, object]:
    optional_params: dict[str, object] = {
        "seconds": "5",
        "resolution": "768x512",
        "aspect_ratio": "3:2",
        "reference_images": [REFERENCE_URL],
        "generate_audio": True,
        "seed": 7,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("generate_audio", [True, False])
async def test_async_create_preserves_generate_audio_in_worker_envelope(generate_audio: bool) -> None:
    redis = FakeRedis()
    response = await handler(redis).avideo_generation(**create_kwargs(generate_audio=generate_audio))

    assert isinstance(response, VideoObject)
    assert decode_video_id_with_provider(response.id)["video_id"] == TASK_ID
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
async def test_async_create_records_usage_for_completion_billing() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence()

    response = await handler(redis, persistence=persistence).avideo_generation(**create_kwargs())

    assert response.status == "queued"
    assert persistence.store_calls == [(f"causyn:{TASK_ID}", 5.0, "768x512")]


@pytest.mark.asyncio
async def test_async_create_survives_usage_persistence_failure() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(store_error=RuntimeError("db unavailable"))

    response = await handler(redis, persistence=persistence).avideo_generation(**create_kwargs())

    assert response.status == "queued"
    assert persistence.store_calls == [(f"causyn:{TASK_ID}", 5.0, "768x512")]


@pytest.mark.asyncio
async def test_public_litellm_video_api_dispatches_to_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    provider = handler(redis)
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "causyn", "custom_handler": provider}])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "causyn"])

    response = await litellm.avideo_generation(
        model="causyn/causyn-1.0",
        prompt="animate the reference",
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
async def test_completed_status_exposes_staging_metadata_without_signed_download_url() -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())

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
async def test_completed_status_bills_once_from_persisted_usage() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
        billed=True,
    )
    set_status(redis, "done", completed_result())

    response = await handler(redis, persistence=persistence).avideo_status(
        VIDEO_ID,
        None,
        None,
        {"model_info": {"output_cost_per_second_768x512": 0.1}},
        None,
    )

    assert response.usage == {"duration_seconds": 5.0, "video_resolution": "768x512"}
    assert response._hidden_params["response_cost"] == pytest.approx(0.5)
    assert persistence.lookup_calls == [f"causyn:{TASK_ID}"]
    assert persistence.billing_calls == [(f"causyn:{TASK_ID}", 5.0, pytest.approx(0.5))]


@pytest.mark.asyncio
async def test_repeated_completed_status_has_zero_cost() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
        billed=False,
    )
    set_status(redis, "done", completed_result())

    response = await handler(redis, persistence=persistence).avideo_status(
        VIDEO_ID,
        None,
        None,
        {"model_info": {"output_cost_per_second_768x512": 0.1}},
        None,
    )

    assert response._hidden_params["response_cost"] == 0.0
    assert persistence.billing_calls == [(f"causyn:{TASK_ID}", 5.0, pytest.approx(0.5))]


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
async def test_completed_status_without_valid_persistence_usage_has_zero_cost(
    persistence: FakeBillingPersistence | None,
) -> None:
    redis = FakeRedis()
    set_status(redis, "done", completed_result())

    response = await handler(redis, persistence=persistence).avideo_status(
        VIDEO_ID,
        None,
        None,
        {"model_info": {"output_cost_per_second_768x512": 0.1}},
        None,
    )

    assert response.status == "completed"
    assert response._hidden_params["response_cost"] == 0.0


@pytest.mark.asyncio
async def test_completed_status_without_price_has_zero_cost_and_does_not_mark_billed() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
    )
    set_status(redis, "done", completed_result())

    response = await handler(redis, persistence=persistence).avideo_status(VIDEO_ID, None, None, {}, None)

    assert response._hidden_params["response_cost"] == 0.0
    assert persistence.billing_calls == []


@pytest.mark.asyncio
async def test_completed_status_billing_failure_has_zero_cost() -> None:
    redis = FakeRedis()
    persistence = FakeBillingPersistence(
        stored_usage={"duration_seconds": 5.0, "video_resolution": "768x512"},
        billing_error=RuntimeError("db unavailable"),
    )
    set_status(redis, "done", completed_result())

    response = await handler(redis, persistence=persistence).avideo_status(
        VIDEO_ID,
        None,
        None,
        {"model_info": {"output_cost_per_second_768x512": 0.1}},
        None,
    )

    assert response._hidden_params["response_cost"] == 0.0


@pytest.mark.asyncio
async def test_content_returns_bytes_from_validated_platform_download_url_without_redirects() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    set_status(redis, "done", completed_result())

    content = await handler(redis, content_get).avideo_content(VIDEO_ID, None, None, {}, None, timeout=12.0)

    assert content == b"video-bytes"
    assert content_get.calls == [(STAGING_URL, 12.0, False)]


@pytest.mark.asyncio
async def test_content_rejects_download_url_outside_existing_target_allowlist() -> None:
    redis = FakeRedis()
    content_get = FakeContentGet()
    set_status(redis, "done", completed_result(staging_url="https://attacker.example/video.mp4"))

    with pytest.raises(CustomLLMError) as exc_info:
        await handler(redis, content_get).avideo_content(VIDEO_ID, None, None, {}, None)

    assert exc_info.value.status_code == 502
    assert content_get.calls == []
    assert "attacker.example" not in str(exc_info.value)


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


def test_handler_is_real_custom_llm_and_sync_methods_are_explicitly_unsupported() -> None:
    provider = handler(FakeRedis())

    assert isinstance(provider, CustomLLM)
    with pytest.raises(NotImplementedError):
        provider.video_generation("causyn-1.0", "prompt", None, None, {}, None)
