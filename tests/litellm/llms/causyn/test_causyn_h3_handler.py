from __future__ import annotations

from types import SimpleNamespace

import fakeredis.aioredis
import pytest

import litellm.llms.causyn.handler as mod
from litellm.llms.causyn.handler import CausynVideoHandler
from litellm.llms.causyn.context_ir import ContextIRService
from litellm.llms.causyn.context_ir_store import ContextIRStore
from litellm.llms.causyn.video_prompt import VideoPromptInput, rewritten_video_payload
from litellm.llms.causyn.h3_prompt import RewriteResult, RewriteUsage
from litellm.llms.custom_llm import CustomLLMError
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
        "source_resolution": "1344x756",
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
        "geometry_profile": "vdn-adaptive-v1",
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


@pytest.mark.parametrize("ratio", [None, "adaptive", "21:9"])
def test_text_only_requires_one_of_five_deployed_ratios(ratio):
    with pytest.raises(CustomLLMError) as error:
        mod._request("causyn-1.1", "prompt", _params(aspect_ratio=ratio))
    assert error.value.status_code == 400


def test_duration_budget_does_not_change_billing_profile():
    request, duration, requested, source, _ = mod._request("causyn-1.1", "prompt", _params(seconds="15"))
    assert duration == 15
    assert requested == request["resolution"] == "768p"
    width, height = map(int, source.split("x"))
    assert width < 1344 and height < 756
    assert width * 9 == height * 16


def _ref(url: str) -> dict[str, str]:
    return {"role": "reference", "media_type": "image", "url": url}


def test_ref2va_references_keep_explicit_ratio_and_source_resolution():
    params = _params(
        aspect_ratio="9:16",
        references=[_ref("https://source.example/r1.png"), _ref("https://source.example/r2.png")],
    )
    request, duration, _, source, _ = mod._request("causyn-1.1", "prompt", params)
    assert request["ratio"] == "9:16"
    assert source == mod._vdn_source_resolution("9:16", duration)
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
