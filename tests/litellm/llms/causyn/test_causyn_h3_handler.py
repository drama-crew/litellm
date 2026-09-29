from __future__ import annotations

import time
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
from pydantic import ValidationError

import litellm.llms.causyn.handler as mod
from litellm.llms.causyn.handler import CausynVideoHandler
from litellm.llms.causyn.context_ir import ContextIRService
from litellm.llms.causyn.context_ir_store import ContextIRStore
from litellm.llms.causyn.video_prompt import VideoPromptInput, deliver_video_prompt, rewritten_video_payload
from litellm.llms.causyn.h3_prompt import RewriteResult, RewriteUsage
from litellm.llms.custom_llm import CustomLLMError
from litellm.llms.libtv.video_generate import alive_zset_key, task_type_for_references
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)


async def fake_submit(payload, billing):
    async def rewrite(spec):
        return RewriteResult(prompt="structured: " + spec.prompt, usage=RewriteUsage(), system_sha256="a" * 64)

    async def deliver(task):
        await mod.enqueue_video_generate(
            rewritten_video_payload(task).model_dump(mode="json"),
            redis_factory=lambda: None,
            settings=mod.VideoGenerateSettings.from_environment(),
        )

    async def settle(task):
        pass

    async with fakeredis.aioredis.FakeRedis() as redis:
        service = ContextIRService(ContextIRStore(redis), rewrite=rewrite, deliver=deliver, settle=settle)
        task = await service.create(
            VideoPromptInput.model_validate(payload.request).context_ir(),
            owner="test",
            billing=billing,
            task_id="h3_ir_" + payload.task_id,
            video_payload=payload.model_dump(mode="json"),
            listed=False,
        )
        await service.process(task.id)
        assert (await service.store.get(task.id)).status == "succeeded"


TASK_ID = "0123456789abcdef0123456789abcdef"


class _Recorder:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def __call__(self, payload, *, redis_factory, settings):
        self.payloads.append(payload)
        return payload["task_id"]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_SOURCE_HOSTS", "source.example")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_TARGET_HOSTS", "target.example")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_ALLOW_HTTP", "false")
    monkeypatch.setenv("DRAMA_CAUSYN_1_1_ENABLED", "true")


@pytest.fixture
def enqueued(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(mod, "enqueue_video_generate", recorder)
    return recorder


def _params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "seconds": "5",
        "size": "768p",
        "aspect_ratio": "16:9",
        "model_info": {
            "id": "causyn-1-1",
            "output_cost_per_second_768p": 5.0,
        },
    }
    params.update(overrides)
    return params


@pytest.mark.asyncio
async def test_h3_enqueues_text_to_video_with_v4_metadata(enqueued: _Recorder) -> None:
    video = await CausynVideoHandler(
        prompt_submit=fake_submit, task_id_factory=lambda: TASK_ID, clock=lambda: 2_000_000_000.0
    ).avideo_generation(
        model="causyn-1.1",
        prompt="a cat crosses the room",
        api_key=None,
        api_base=None,
        optional_params=_params(),
        logging_obj=None,
    )
    payload = enqueued.payloads[0]
    assert payload["model"] == "causyn-1.1"
    assert payload["deadline_ts"] == 2_000_001_800.0
    assert payload["request"] == {
        "prompt": "structured: a cat crosses the room",
        "duration_seconds": 5,
        "resolution": "768p",
        "ratio": "16:9",
        "generate_audio": True,
        "references": [],
    }
    assert payload["task_metadata"] == {
        "version": "causyn-video-billing-v4",
        "context_ir_task_id": "h3_ir_" + TASK_ID,
        "prompt_rewrite_model": "qwen/qwen3.8-flash",
        "prompt_rewrite_system_sha256": "a" * 64,
        "duration_seconds": 5.0,
        "source_resolution": "1344x768",
        "requested_resolution": "768p",
        "pricing": {
            "model": "causyn-1.1",
            "id": "causyn-1-1",
            "output_cost_per_second_768p": 5.0,
        },
        "attribution": {
            "api_key": None,
            "team_id": None,
            "user_id": None,
            "organization_id": None,
        },
        "model": "causyn-1.1",
        "geometry_profile": "hyperflow-official-v1",
        "ratio": "16:9",
    }
    decoded = decode_video_id_with_provider(video.id)
    assert decoded["model_id"] == "causyn-1-1"
    assert video.model == "causyn-1.1"
    assert video.size == "768p"


def test_h3_uses_an_independent_deadline_constant() -> None:
    assert mod._MODEL_SPECS[mod.CAUSYN_H3_MODEL].deadline_seconds == mod.CAUSYN_H3_DEADLINE_SECONDS
    assert mod._MODEL_SPECS[mod.CAUSYN_MODEL].deadline_seconds == mod.CAUSYN_DEADLINE_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (
            {"image": "https://source.example/first.png"},
            [
                {
                    "role": "first_frame",
                    "media_type": "image",
                    "url": "https://source.example/first.png",
                }
            ],
        ),
        (
            {
                "image": "https://source.example/first.png",
                "last_image": "https://source.example/last.png",
            },
            [
                {
                    "role": "first_frame",
                    "media_type": "image",
                    "url": "https://source.example/first.png",
                },
                {
                    "role": "last_frame",
                    "media_type": "image",
                    "url": "https://source.example/last.png",
                },
            ],
        ),
    ],
)
async def test_h3_maps_keyframes_to_worker_references(
    enqueued: _Recorder, params: dict[str, object], expected: list[dict[str, str]]
) -> None:
    await CausynVideoHandler(
        prompt_submit=fake_submit,
    ).avideo_generation(
        model="causyn-1.1",
        prompt="continuous movement",
        api_key=None,
        api_base=None,
        optional_params=_params(**params),
        logging_obj=None,
    )
    assert enqueued.payloads[0]["request"]["references"] == expected


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        ("16:9", "1344x768"),
        ("9:16", "768x1344"),
        ("1:1", "768x768"),
        ("4:3", "1024x768"),
        ("3:4", "768x1024"),
        ("21:9", "1792x768"),
    ],
)
def test_h3_dimensions_match_the_32_pixel_grid(ratio: str, expected: str) -> None:
    assert mod._h3_source_resolution(ratio) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"seconds": "3"},
        {"seconds": "16"},
        {"size": "2k"},
        {"aspect_ratio": "3:2"},
        {"generate_audio": False},
        {"reference_images": ["https://source.example/ref.png"]},
        {"last_image": "https://source.example/last.png"},
    ],
)
def test_h3_rejects_parameters_outside_phase_one(overrides: dict[str, object]) -> None:
    with pytest.raises(CustomLLMError) as exc:
        mod._request("causyn-1.1", "prompt", _params(**overrides))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_h3_flag_is_independent_from_causyn_1_0(monkeypatch: pytest.MonkeyPatch, enqueued: _Recorder) -> None:
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", "true")
    monkeypatch.delenv("DRAMA_CAUSYN_1_1_ENABLED")
    monkeypatch.delenv("OH_DRAMA_CAUSYN_1_1_ENABLED", raising=False)
    with pytest.raises(CustomLLMError) as exc:
        await CausynVideoHandler(
            prompt_submit=fake_submit,
        ).avideo_generation(
            model="causyn-1.1",
            prompt="prompt",
            api_key=None,
            api_base=None,
            optional_params=_params(),
            logging_obj=None,
        )
    assert exc.value.status_code == 403
    assert enqueued.payloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "ratio", "source", "width", "height"),
    [
        ("causyn-video-billing-v3", "16:9", "1344x768", 1344, 768),
        ("causyn-video-billing-v4", "16:9", "1344x756", 1344, 756),
        ("causyn-video-billing-v4", "adaptive", "adaptive", 768, 1024),
    ],
)
async def test_h3_completed_status_restores_model_and_billing(
    monkeypatch: pytest.MonkeyPatch,
    version,
    ratio,
    source,
    width,
    height,
) -> None:
    async def fetch_status(task_id: str, *, redis: object) -> dict[str, object]:
        return {
            "ok": True,
            "task_id": task_id,
            "status": "succeeded",
            "result": {
                "validation_version": "video-v1",
                "staging_key": f"staging/video-tasks/{task_id}.mp4",
                "etag": "etag-1",
                "bytes": 100,
                "content_type": "video/mp4",
                "duration_seconds": 5.0 + 1 / 24,
                "width": width,
                "height": height,
                "sha256": "a" * 64,
            },
        }

    async def fetch_metadata(task_id: str, *, redis: object) -> dict[str, object]:
        return {
            "version": version,
            "model": "causyn-1.1",
            "duration_seconds": 5.0,
            "source_resolution": source,
            "requested_resolution": "768p",
            "ratio": ratio,
            "pricing": {
                "model": "causyn-1.1",
                "id": "causyn-1-1",
                "output_cost_per_second_768p": 5.0,
            },
            "attribution": {
                "api_key": None,
                "team_id": None,
                "user_id": None,
                "organization_id": None,
            },
        }

    billed: list[object] = []

    async def enqueue_billing(redis: object, event: object) -> bool:
        billed.append(event)
        return True

    monkeypatch.setattr(mod, "fetch_video_generate_status", fetch_status)
    monkeypatch.setattr(mod, "fetch_video_generate_task_metadata", fetch_metadata)
    video_id = encode_video_id_with_provider(TASK_ID, "causyn", "causyn-1-1")
    handler = CausynVideoHandler(
        prompt_submit=fake_submit, redis_factory=lambda: SimpleNamespace(), billing_enqueue=enqueue_billing
    )
    video = await handler.avideo_status(video_id, None, None, {}, None)
    assert video.status == "completed"
    assert video.model == "causyn-1.1"
    assert video.size == "768p"
    assert video._hidden_params["response_cost"] == 25.0
    assert len(billed) == 1
    assert billed[0].model == "causyn-1.1"


@pytest.mark.parametrize("ratio", [None, "adaptive", "16:9", "9:16", "21:9"])
@pytest.mark.parametrize("last", [False, True])
def test_keyframe_requests_normalize_valid_ratios_to_adaptive(ratio, last):
    params = _params(aspect_ratio=ratio, image="https://source.example/first.png")
    if last:
        params["last_image"] = "https://source.example/last.png"
    request, _, _, source, _ = mod._request("causyn-1.1", "prompt", params)
    assert request["ratio"] == "adaptive"
    assert source == "adaptive"


@pytest.mark.parametrize("ratio", [None, "adaptive"])
def test_text_only_requires_an_explicit_ratio(ratio):
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", _params(aspect_ratio=ratio))
    assert error.value.status_code == 400


def test_text_only_accepts_21_9_as_one_of_the_six_deployed_ratios():
    request, duration, _, source, _ = mod._request("causyn-1.1", "prompt", _params(aspect_ratio="21:9"))
    assert request["ratio"] == "21:9"
    assert source == mod._vdn_source_resolution("21:9", duration, "hyperflow-official-v1")


def test_duration_budget_does_not_change_billing_profile():
    request, duration, requested, source, _ = mod._request("causyn-1.1", "prompt", _params(seconds="15"))
    assert duration == 15
    assert requested == request["resolution"] == "768p"
    assert source == "1344x768"


@pytest.mark.parametrize(
    ("ratio", "duration", "expected"),
    [
        ("16:9", 5, "1344x768"),
        ("16:9", 9, "1344x768"),
        ("16:9", 10, "1344x768"),
        ("16:9", 15, "1344x768"),
        ("9:16", 10, "768x1344"),
        ("9:16", 15, "768x1344"),
        ("4:3", 11, "1024x768"),
        ("4:3", 12, "1024x768"),
        ("3:4", 12, "768x1024"),
        ("21:9", 9, "1536x672"),
        ("21:9", 15, "1536x672"),
        ("1:1", 5, "768x768"),
        ("1:1", 15, "768x768"),
    ],
)
def test_hyperflow_official_geometry_ignores_the_duration_budget(ratio: str, duration: int, expected: str) -> None:
    """C3: HyperFlow 的官方几何按比例固定，不随时长缩水。

    旧的 vdn-adaptive-v1 画幅公式会在这些时长/比例组合下把画布缩小以适应
    spatial-temporal 像素预算，导致 worker 按 ARK 官方公式产出的画布与计费存的
    source_resolution 逐字节比对不上，任务永远卡在 in_progress（生产实测
    public.collect 无限重试）。
    """
    assert mod._vdn_source_resolution(ratio, duration, "hyperflow-official-v1") == expected


@pytest.mark.parametrize(("ratio", "duration"), [("16:9", 10), ("21:9", 15), ("4:3", 12), ("9:16", 15)])
def test_the_legacy_adaptive_profile_still_shrinks_long_jobs(ratio: str, duration: int) -> None:
    """确认 geometry_profile 真的被接线到几何计算里，而不是加了字段却没人读。

    旧 profile 专门用来恢复部署前已经持久化的任务，它的画幅算法必须原样保留。
    """
    official = mod._vdn_source_resolution(ratio, duration, "hyperflow-official-v1")
    legacy = mod._vdn_source_resolution(ratio, duration, "vdn-adaptive-v1")
    assert legacy != official


def test_geometry_profile_is_irrelevant_once_ratio_is_adaptive() -> None:
    assert (
        mod._vdn_source_resolution("adaptive", 10, "hyperflow-official-v1")
        == mod._vdn_source_resolution("adaptive", 10, "vdn-adaptive-v1")
        == "adaptive"
    )


def _ref(url: str) -> dict[str, str]:
    return {"role": "reference", "media_type": "image", "url": url}


def test_ref2va_references_keep_explicit_ratio_and_source_resolution():
    params = _params(
        aspect_ratio="9:16",
        references=[_ref("https://source.example/r1.png"), _ref("https://source.example/r2.png")],
    )
    request, duration, _, source, _ = mod._request("causyn-1.1", "prompt", params)
    assert request["ratio"] == "9:16"
    assert source == mod._vdn_source_resolution("9:16", duration, "hyperflow-official-v1")
    assert request["references"] == [
        _ref("https://source.example/r1.png"),
        _ref("https://source.example/r2.png"),
    ]


def test_ref2va_up_to_nine_references_accepted():
    refs = [_ref(f"https://source.example/r{i}.png") for i in range(9)]
    params = _params(aspect_ratio="16:9", references=refs)
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", params)
    assert len(request["references"]) == 9


def test_ref2va_rejects_more_than_nine_references():
    refs = [_ref(f"https://source.example/r{i}.png") for i in range(10)]
    params = _params(aspect_ratio="16:9", references=refs)
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", params)
    assert error.value.status_code == 400


@pytest.mark.parametrize("ratio", [None, "adaptive"])
def test_ref2va_requires_explicit_ratio(ratio):
    params = _params(aspect_ratio=ratio, references=[_ref("https://source.example/r1.png")])
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", params)
    assert error.value.status_code == 400


def test_ref2va_rejects_mixing_reference_role_with_keyframes():
    params = _params(
        aspect_ratio="16:9",
        references=[
            _ref("https://source.example/r1.png"),
            {"role": "first_frame", "media_type": "image", "url": "https://source.example/first.png"},
        ],
    )
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", params)
    assert error.value.status_code == 400


async def test_h3_maps_ref2va_references_to_worker_references(enqueued: _Recorder) -> None:
    refs = [_ref("https://source.example/r1.png"), _ref("https://source.example/r2.png")]
    await CausynVideoHandler(
        prompt_submit=fake_submit,
    ).avideo_generation(
        model="causyn-1.1",
        prompt="a subject preserved across shots",
        api_key=None,
        api_base=None,
        optional_params=_params(aspect_ratio="16:9", references=refs),
        logging_obj=None,
    )
    assert enqueued.payloads[0]["request"]["references"] == refs
    assert enqueued.payloads[0]["request"]["ratio"] == "16:9"


def _media_ref(media_type: str, url: str) -> dict[str, str]:
    return {"role": "reference", "media_type": media_type, "url": url}


def test_ref2va_preserves_mixed_media_reference_order():
    refs = [
        _media_ref("audio", "https://source.example/a.mp3"),
        _media_ref("image", "https://source.example/r1.png"),
        _media_ref("video", "https://source.example/r1.mp4"),
    ]
    params = _params(aspect_ratio="16:9", references=refs)
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", params)
    assert request["references"] == refs


def test_ref2va_rejects_more_than_three_videos():
    refs = [_media_ref("video", f"https://source.example/v{i}.mp4") for i in range(4)]
    params = _params(aspect_ratio="16:9", references=refs)
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", params)
    assert error.value.status_code == 400


def test_ref2va_rejects_more_than_three_audios():
    refs = [_media_ref("image", "https://source.example/r1.png")] + [
        _media_ref("audio", f"https://source.example/a{i}.mp3") for i in range(4)
    ]
    params = _params(aspect_ratio="16:9", references=refs)
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", params)
    assert error.value.status_code == 400


def test_ref2va_rejects_audio_without_image_or_video():
    refs = [_media_ref("audio", "https://source.example/a.mp3")]
    params = _params(aspect_ratio="16:9", references=refs)
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", params)
    assert error.value.status_code == 400


def test_ref2va_accepts_audio_paired_with_video():
    refs = [
        _media_ref("video", "https://source.example/r1.mp4"),
        _media_ref("audio", "https://source.example/a.mp3"),
    ]
    params = _params(aspect_ratio="16:9", references=refs)
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", params)
    assert request["references"] == refs


def test_prompt_processing_direct_when_a_video_reference_is_present():
    refs = [_media_ref("video", "https://source.example/r1.mp4")]
    params = _params(aspect_ratio="16:9", references=refs)
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", params)
    assert request["prompt_processing"] == "direct"


def test_prompt_processing_direct_when_an_audio_reference_is_present():
    refs = [
        _media_ref("image", "https://source.example/r1.png"),
        _media_ref("audio", "https://source.example/a.mp3"),
    ]
    params = _params(aspect_ratio="16:9", references=refs)
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", params)
    assert request["prompt_processing"] == "direct"


@pytest.mark.parametrize(
    "refs",
    [
        [],
        [_ref("https://source.example/r1.png")],
        [{"role": "first_frame", "media_type": "image", "url": "https://source.example/first.png"}],
    ],
)
def test_prompt_processing_omitted_for_image_only_or_text_only_requests(refs):
    params = _params(aspect_ratio="16:9", references=refs) if refs else _params(aspect_ratio="16:9")
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", params)
    assert "prompt_processing" not in request


async def test_h3_forwards_prompt_processing_direct_to_the_worker_after_the_ir_rewrite(
    enqueued: _Recorder,
) -> None:
    refs = [_media_ref("video", "https://source.example/r1.mp4")]
    await CausynVideoHandler(
        prompt_submit=fake_submit,
    ).avideo_generation(
        model="causyn-1.1",
        prompt="a subject preserved across shots",
        api_key=None,
        api_base=None,
        optional_params=_params(aspect_ratio="16:9", references=refs),
        logging_obj=None,
    )
    assert enqueued.payloads[0]["request"]["prompt_processing"] == "direct"
    assert enqueued.payloads[0]["request"]["prompt"] == "structured: a subject preserved across shots"


async def test_h3_video_reference_request_clears_the_real_worker_gate() -> None:
    """`enqueued` swaps in a recorder for `enqueue_video_generate`, so it never
    runs the real `libtv.video_generate._validate_shape`. That shape gate's
    closed request-key set didn't know about `prompt_processing`, so every H3
    request carrying a video or audio reference was rejected at the render
    queue with `invalid_params: unrecognized request field(s): prompt_processing`
    -- the rewrite ran (and billed a real model call) but the video never
    rendered. This drives the real `enqueue_video_generate` (via the real
    `deliver_video_prompt`) end to end against a fake Redis, so a regression
    here fails the task instead of a mocked call."""
    refs = [_media_ref("video", "https://source.example/r1.mp4")]

    async def real_submit(payload, billing):
        async def rewrite(spec):
            return RewriteResult(prompt="structured: " + spec.prompt, usage=RewriteUsage(), system_sha256="a" * 64)

        async def settle(task):
            pass

        async with fakeredis.aioredis.FakeRedis() as ir_redis, fakeredis.aioredis.FakeRedis() as worker_redis:
            task_type = task_type_for_references(tuple(refs))
            await worker_redis.zadd(alive_zset_key(task_type), {"worker-1": time.time()})

            async def deliver(task):
                await deliver_video_prompt(task, redis_factory=lambda: worker_redis)

            service = ContextIRService(ContextIRStore(ir_redis), rewrite=rewrite, deliver=deliver, settle=settle)
            task = await service.create(
                VideoPromptInput.model_validate(payload.request).context_ir(),
                owner="test",
                billing=billing,
                task_id="h3_ir_" + payload.task_id,
                video_payload=payload.model_dump(mode="json"),
                listed=False,
            )
            await service.process(task.id)
            stored = await service.store.get(task.id)
            assert stored.status == "succeeded", stored.error

    await CausynVideoHandler(
        prompt_submit=real_submit,
    ).avideo_generation(
        model="causyn-1.1",
        prompt="a subject preserved across shots",
        api_key=None,
        api_base=None,
        optional_params=_params(aspect_ratio="16:9", references=refs),
        logging_obj=None,
    )


@pytest.mark.parametrize("seed", [0, 4294967295, 42])
def test_seed_within_range_is_forwarded(seed):
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", _params(seed=seed))
    assert request["seed"] == seed


def test_seed_omitted_when_not_provided():
    request, _, _, _, _ = mod._request("causyn-1.1", "prompt", _params())
    assert "seed" not in request


@pytest.mark.parametrize("seed", [-1, 4294967296, True, 1.5, "42"])
def test_seed_out_of_range_or_wrong_type_is_rejected(seed):
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", _params(seed=seed))
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_h3_prompt_over_7000_characters_is_a_bad_request_not_a_service_error() -> None:
    with pytest.raises(CustomLLMError) as error:
        await CausynVideoHandler(prompt_submit=fake_submit, task_id_factory=lambda: TASK_ID).avideo_generation(
            model="causyn-1.1",
            prompt="x" * 7001,
            api_key=None,
            api_base=None,
            optional_params=_params(),
            logging_obj=None,
        )
    assert error.value.status_code == 400


class TestBillingAcceptsTheGeometryThePipelineProduces:
    """16:9 与 9:16 的 canvas 和 output 不同，而裁剪从未实现。

    几何表里 `Geometry("16:9", 1344, 768, 1344, 756)`：canvas 是 1344×768，
    output 是 1344×756（1344×768 其实是 7:4）。要得到真 16:9 必须裁到 756，而
    756 不是 32 的倍数、模型产不出来，所以设计是「按 canvas 生成、再裁到 output」。

    但 `crop_video()` 在 litellm、h3-ark、video-worker 三处都没有任何调用方——
    裁剪从未实现。计费却拿 output 去和 worker 实际产出逐字符比对，于是 16:9 和
    9:16 必然失败：视频生成成功、上传成功，API 却永远停在 in_progress，
    `public.collect` 无限重试（生产实测 7850 次）。

    修法是让比对接受**流水线实际产出的几何**（canvas），同时仍接受 output——
    这样哪天裁剪真的接上了，也不会反过来把它判成不匹配。
    """

    @staticmethod
    def _metadata(ratio: str, source: str):
        from litellm.llms.causyn.handler import _DurableTaskMetadataV4

        # 这份形状抄自生产上一条真实任务的 metadata，不是我编的
        return _DurableTaskMetadataV4.model_validate(
            {
                "version": "causyn-video-billing-v4",
                "geometry_profile": "vdn-adaptive-v1",
                "model": "causyn-1.1",
                "duration_seconds": 5.0,
                "source_resolution": source,
                "requested_resolution": "768p",
                "ratio": ratio,
                "pricing": {"id": "causyn-1-1", "model": "causyn-1.1", "output_cost_per_second_768p": 5.0},
                "attribution": {
                    "api_key": "a" * 64,
                    "user_id": "99e724bb-4938-4ecf-88b2-66f584314829",
                    "team_id": "99e724bb-4938-4ecf-88b2-66f584314829",
                    "organization_id": None,
                },
            }
        )

    @staticmethod
    def _result(width: int, height: int):
        from litellm.llms.causyn.handler import _WorkerResult

        return _WorkerResult.model_validate(
            {
                "validation_version": "video-v1",
                "staging_key": "staging/x.mp4",
                "etag": '"0"',
                "bytes": 1,
                "content_type": "video/mp4",
                "duration_seconds": 5.175,
                "width": width,
                "height": height,
                "sha256": "0" * 64,
            }
        )

    def test_landscape_canvas_is_accepted(self):
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert _result_geometry_matches(
            self._metadata("16:9", "1344x756"), self._result(1344, 768)
        ), "流水线产出 canvas，计费必须接受它，否则视频永远交付不了"

    def test_portrait_canvas_is_accepted(self):
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert _result_geometry_matches(
            self._metadata("9:16", "756x1344"), self._result(768, 1344)
        )

    def test_the_cropped_output_is_still_accepted(self):
        """裁剪哪天接上了，也不能反过来被判成不匹配。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert _result_geometry_matches(
            self._metadata("16:9", "1344x756"), self._result(1344, 756)
        )

    def test_an_unrelated_geometry_is_still_rejected(self):
        """放宽不等于放弃：与这个比例无关的尺寸仍须拒绝。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert not _result_geometry_matches(
            self._metadata("16:9", "1344x756"), self._result(640, 480)
        )


class TestAdmittedGeometriesSurviveAnyArkRollout:
    """I-A：几何 profile 只是记账用的审计字段，不能决定任务能否交付。

    生产 ARK 在 hyperflow8、vdn8（vdn-adaptive-v1）与旧版非 vdn 的 h3 后端之间
    切换时不会通知 LiteLLM，回滚也一样。持久化的 `geometry_profile` 只反映任务
    创建那一刻的假设；worker 最终按哪台 ARK 的几何交付，`_admitted_geometries`
    必须与之独立，接受三套后端已知会产出的每一种几何。
    """

    @staticmethod
    def _metadata(ratio: str, geometry_profile: str, source: str, duration: float = 15.0):
        from litellm.llms.causyn.handler import _DurableTaskMetadataV4

        return _DurableTaskMetadataV4.model_validate(
            {
                "version": "causyn-video-billing-v4",
                "geometry_profile": geometry_profile,
                "model": "causyn-1.1",
                "duration_seconds": duration,
                "source_resolution": source,
                "requested_resolution": "768p",
                "ratio": ratio,
                "pricing": {"id": "causyn-1-1", "model": "causyn-1.1", "output_cost_per_second_768p": 5.0},
                "attribution": {"api_key": None, "user_id": None, "team_id": None, "organization_id": None},
            }
        )

    @staticmethod
    def _result(width: int, height: int):
        from litellm.llms.causyn.handler import _WorkerResult

        return _WorkerResult.model_validate(
            {
                "validation_version": "video-v1",
                "staging_key": "staging/x.mp4",
                "etag": '"0"',
                "bytes": 1,
                "content_type": "video/mp4",
                "duration_seconds": 15.2,
                "width": width,
                "height": height,
                "sha256": "0" * 64,
            }
        )

    @staticmethod
    def _vdn_canvas(ratio: str, duration: int):
        frames = duration * 24 + (5 - duration * 24) % 17
        return mod.resolve_geometry(ratio=ratio, frames=frames)

    def test_a_new_profile_task_delivered_at_the_vdn_canvas_geometry_settles(self):
        """15s、16:9 上，hyperflow 的固定 canvas（1344x768）与 vdn-adaptive-v1
        按像素预算收缩出的 canvas（992x576）截然不同。记录的是新 profile，但
        回滚后由一台 vdn8 ARK 交付——必须仍然结算。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        duration = 15
        vdn_canvas = self._vdn_canvas("16:9", duration)
        source = mod._vdn_source_resolution("16:9", duration, "hyperflow-official-v1")
        assert _result_geometry_matches(
            self._metadata("16:9", "hyperflow-official-v1", source, duration=duration),
            self._result(vdn_canvas.width, vdn_canvas.height),
        )

    def test_an_old_profile_task_delivered_at_the_hyperflow_canvas_still_settles(self):
        """反方向：记录的是旧 profile（vdn-adaptive-v1 收缩后的几何），但一台
        hyperflow8 ARK 按官方固定 canvas 交付——例如任务在后端切换前后被重试。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        duration = 15
        hyperflow = mod.GEOMETRIES["16:9"]
        source = mod._vdn_source_resolution("16:9", duration, "vdn-adaptive-v1")
        assert _result_geometry_matches(
            self._metadata("16:9", "vdn-adaptive-v1", source, duration=duration),
            self._result(hyperflow.width, hyperflow.height),
        )

    def test_a_legacy_non_vdn_h3_backend_delivering_21_9_settles_regardless_of_profile(self):
        """生产 ARK 环境曾经跑过不带 vdn 的旧 h3 后端；它对 21:9 的几何是
        1792x768（`_h3_source_resolution`），既不是 hyperflow 也不是 vdn-adaptive-v1
        算出的几何。两种记录的 profile 都必须接受它。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        duration = 15
        hyperflow_source = mod._vdn_source_resolution("21:9", duration, "hyperflow-official-v1")
        vdn_source = mod._vdn_source_resolution("21:9", duration, "vdn-adaptive-v1")
        assert _result_geometry_matches(
            self._metadata("21:9", "hyperflow-official-v1", hyperflow_source, duration=duration),
            self._result(1792, 768),
        )
        assert _result_geometry_matches(
            self._metadata("21:9", "vdn-adaptive-v1", vdn_source, duration=duration),
            self._result(1792, 768),
        )

    def test_a_geometry_unrelated_to_any_known_ark_backend_is_still_rejected(self):
        from litellm.llms.causyn.handler import _result_geometry_matches

        duration = 15
        source = mod._vdn_source_resolution("16:9", duration, "hyperflow-official-v1")
        assert not _result_geometry_matches(
            self._metadata("16:9", "hyperflow-official-v1", source, duration=duration),
            self._result(640, 480),
        )

    def test_a_genuine_vdn8_native_21_9_canvas_settles_regardless_of_profile(self):
        """causyn-1.1 是真正跑在 vdn8 上的型号，而 vdn8 自己的几何公式
        (`vdn_geometry.GEOMETRIES`) 本就有一条原生 21:9 几何——与旧版非 vdn
        h3 后端的 1792x768 是两码事，且随 duration 收缩。无论持久化的
        profile 是哪个，vdn8 真实交付的这张 canvas 都必须结算。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        duration = 15
        vdn_native = self._vdn_canvas("21:9", duration)
        hyperflow_source = mod._vdn_source_resolution("21:9", duration, "hyperflow-official-v1")
        vdn_source = mod._vdn_source_resolution("21:9", duration, "vdn-adaptive-v1")
        assert _result_geometry_matches(
            self._metadata("21:9", "hyperflow-official-v1", hyperflow_source, duration=duration),
            self._result(vdn_native.width, vdn_native.height),
        )
        assert _result_geometry_matches(
            self._metadata("21:9", "vdn-adaptive-v1", vdn_source, duration=duration),
            self._result(vdn_native.width, vdn_native.height),
        )


class TestAdaptiveGeometryWindowMatchesArkRounding:
    """I-C：FL2VA 的宽高比窗口曾用 0.399-2.506 这两个写死的数字，但 ARK 的
    `_fit_geometry` 在输入关键帧取到 [0.4, 2.5] 边界、再按 32 像素取整之后，
    实际产出的画幅比可以跑到 [0.3846, 2.6]（例如 11-14s 的 1248x480，比值
    2.6）。窗口现在从 `_fit_geometry` 本身在这些边界上的输出反推，而不是
    手写一个近似值。"""

    @staticmethod
    def _metadata(source: str, duration: float = 15.0):
        from litellm.llms.causyn.handler import _DurableTaskMetadataV4

        return _DurableTaskMetadataV4.model_validate(
            {
                "version": "causyn-video-billing-v4",
                "geometry_profile": "vdn-adaptive-v1",
                "model": "causyn-1.1",
                "duration_seconds": duration,
                "source_resolution": source,
                "requested_resolution": "768p",
                "ratio": "adaptive",
                "pricing": {"id": "causyn-1-1", "model": "causyn-1.1", "output_cost_per_second_768p": 5.0},
                "attribution": {"api_key": None, "user_id": None, "team_id": None, "organization_id": None},
            }
        )

    @staticmethod
    def _result(width: int, height: int):
        from litellm.llms.causyn.handler import _WorkerResult

        return _WorkerResult.model_validate(
            {
                "validation_version": "video-v1",
                "staging_key": "staging/x.mp4",
                "etag": '"0"',
                "bytes": 1,
                "content_type": "video/mp4",
                "duration_seconds": 5.175,
                "width": width,
                "height": height,
                "sha256": "0" * 64,
            }
        )

    def test_1536x608_from_a_near_5_2_keyframe_at_5s_settles(self):
        """review 报告里明确举出的例子：2.526 的宽高比，5-8s 都会产出这张 canvas。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert _result_geometry_matches(self._metadata("adaptive", duration=5.0), self._result(1536, 608))

    def test_1248x480_at_13s_is_the_true_extreme_and_settles(self):
        """13s 的预算收缩让比值推到 2.6，比 review 建议的 0.39-2.56 兜底还宽——
        这正是要求"从 ARK 的数学推导，而不是写死一个近似值"的原因。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert _result_geometry_matches(self._metadata("adaptive", duration=13.0), self._result(1248, 480))

    def test_the_portrait_mirror_at_13s_settles(self):
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert _result_geometry_matches(self._metadata("adaptive", duration=13.0), self._result(480, 1248))

    def test_a_result_beyond_the_true_extreme_is_still_rejected(self):
        """放宽不等于放弃：比 ARK 实际能产出的最极端画幅比还夸张的结果仍须拒绝。"""
        from litellm.llms.causyn.handler import _result_geometry_matches

        assert not _result_geometry_matches(self._metadata("adaptive", duration=13.0), self._result(1280, 480))


class TestGeometryProfileOnPersistedMetadata:
    """C3: geometry_profile 必须真的接进校验，旧任务与新任务各走各的公式。"""

    @staticmethod
    def _base(**overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "version": "causyn-video-billing-v4",
            "model": "causyn-1.1",
            "duration_seconds": 15.0,
            "requested_resolution": "768p",
            "ratio": "16:9",
            "pricing": {"id": "causyn-1-1", "model": "causyn-1.1", "output_cost_per_second_768p": 5.0},
            "attribution": {"api_key": None, "user_id": None, "team_id": None, "organization_id": None},
        }
        base.update(overrides)
        return base

    def test_a_blob_persisted_before_this_field_existed_defaults_to_the_adaptive_profile(self):
        from litellm.llms.causyn.handler import _DurableTaskMetadataV4

        stored = self._base(source_resolution=mod._vdn_source_resolution("16:9", 15, "vdn-adaptive-v1"))
        assert "geometry_profile" not in stored
        metadata = _DurableTaskMetadataV4.model_validate(stored)
        assert metadata.geometry_profile == "vdn-adaptive-v1"

    def test_a_new_task_persists_the_official_fixed_canvas_at_a_long_duration(self):
        from litellm.llms.causyn.handler import _DurableTaskMetadataV4

        stored = self._base(geometry_profile="hyperflow-official-v1", source_resolution="1344x768")
        metadata = _DurableTaskMetadataV4.model_validate(stored)
        assert metadata.geometry_profile == "hyperflow-official-v1"

    def test_the_official_profile_rejects_the_stale_adaptive_shrunk_resolution(self):
        from litellm.llms.causyn.handler import _DurableTaskMetadataV4

        stored = self._base(
            geometry_profile="hyperflow-official-v1",
            source_resolution=mod._vdn_source_resolution("16:9", 15, "vdn-adaptive-v1"),
        )
        with pytest.raises(ValidationError):
            _DurableTaskMetadataV4.model_validate(stored)


@pytest.mark.asyncio
async def test_h3_long_duration_text_to_video_persists_the_fixed_hyperflow_canvas(enqueued: _Recorder) -> None:
    """回归 C3：15s 的 16:9 t2va 过去会把 source_resolution 缩到自适应预算算出的
    更小画幅，worker 按 ARK 官方几何交付 1344x768 后计费判不匹配，任务永远卡在
    in_progress。"""
    await CausynVideoHandler(
        prompt_submit=fake_submit, task_id_factory=lambda: TASK_ID, clock=lambda: 2_000_000_000.0
    ).avideo_generation(
        model="causyn-1.1",
        prompt="a long take across the market",
        api_key=None,
        api_base=None,
        optional_params=_params(seconds="15"),
        logging_obj=None,
    )
    metadata = enqueued.payloads[0]["task_metadata"]
    assert metadata["geometry_profile"] == "hyperflow-official-v1"
    assert metadata["source_resolution"] == "1344x768"

    from litellm.llms.causyn.handler import _DurableTaskMetadataV4, _WorkerResult, _result_geometry_matches

    result = _WorkerResult.model_validate(
        {
            "validation_version": "video-v1",
            "staging_key": "staging/x.mp4",
            "etag": '"0"',
            "bytes": 1,
            "content_type": "video/mp4",
            "duration_seconds": 15.2,
            "width": 1344,
            "height": 768,
            "sha256": "0" * 64,
        }
    )
    assert _result_geometry_matches(_DurableTaskMetadataV4.model_validate(metadata), result)
