from __future__ import annotations

from types import SimpleNamespace

import pytest

import litellm.llms.causyn.handler as mod
from litellm.llms.causyn.handler import CausynVideoHandler
from litellm.llms.custom_llm import CustomLLMError
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)


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
async def test_h3_enqueues_text_to_video_with_v3_metadata(enqueued: _Recorder) -> None:
    video = await CausynVideoHandler(task_id_factory=lambda: TASK_ID, clock=lambda: 2_000_000_000.0).avideo_generation(
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
        "prompt": "a cat crosses the room",
        "duration_seconds": 5,
        "resolution": "768p",
        "ratio": "16:9",
        "generate_audio": True,
        "references": [],
    }
    assert payload["task_metadata"] == {
        "version": "causyn-video-billing-v3",
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
    await CausynVideoHandler().avideo_generation(
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
        await CausynVideoHandler().avideo_generation(
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
async def test_h3_completed_status_restores_model_and_billing(
    monkeypatch: pytest.MonkeyPatch,
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
                "width": 1344,
                "height": 768,
                "sha256": "a" * 64,
            },
        }

    async def fetch_metadata(task_id: str, *, redis: object) -> dict[str, object]:
        return {
            "version": "causyn-video-billing-v3",
            "model": "causyn-1.1",
            "duration_seconds": 5.0,
            "source_resolution": "1344x768",
            "requested_resolution": "768p",
            "ratio": "16:9",
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
    handler = CausynVideoHandler(redis_factory=lambda: SimpleNamespace(), billing_enqueue=enqueue_billing)
    video = await handler.avideo_status(video_id, None, None, {}, None)
    assert video.status == "completed"
    assert video.model == "causyn-1.1"
    assert video.size == "768p"
    assert video._hidden_params["response_cost"] == 25.0
    assert len(billed) == 1
    assert billed[0].model == "causyn-1.1"
