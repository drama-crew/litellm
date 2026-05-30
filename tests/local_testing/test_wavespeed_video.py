import httpx

import litellm
from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig
from litellm.types.router import GenericLiteLLMParams


def test_wavespeed_video_create_maps_openai_params_to_seedance_image_to_video() -> None:
    config = WaveSpeedVideoConfig()
    data, files, url = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="POV camera moves through a neon hotel corridor.",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params={
            "input_reference": "https://assets.local/first.png",
            "seconds": "5",
            "size": "720x1280",
            "extra_body": {"generate_audio": False},
        },
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )

    assert files == []
    assert url == "https://api.wavespeed.ai/api/v3/bytedance/seedance-2.0/image-to-video"
    assert data["prompt"] == "POV camera moves through a neon hotel corridor."
    assert data["image"] == "https://assets.local/first.png"
    assert data["duration"] == 5
    assert data["duration_seconds"] == 5
    assert data["aspect_ratio"] == "9:16"
    assert data["resolution"] == "720p"
    assert data["generate_audio"] is False


def test_wavespeed_video_create_response_sets_usage_for_cost_calculator() -> None:
    config = WaveSpeedVideoConfig()
    response = httpx.Response(200, json={"id": "task-1", "status": "created"})

    video = config.transform_video_create_response(
        model="bytedance/seedance-2.0-fast",
        raw_response=response,
        logging_obj=None,
        custom_llm_provider="wavespeed",
        request_data={"duration": 5, "resolution": "720p"},
    )

    assert video.id.startswith("video_")
    assert video.status == "queued"
    assert video.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}


def test_wavespeed_video_status_poll_maps_completed_status_and_extracts_task_id() -> None:
    config = WaveSpeedVideoConfig()
    encoded_id = config.transform_video_create_response(
        model="bytedance/seedance-2.0-fast",
        raw_response=httpx.Response(200, json={"id": "task-9", "status": "created"}),
        logging_obj=None,
        custom_llm_provider="wavespeed",
        request_data={"duration": 5},
    ).id

    url, params = config.transform_video_status_retrieve_request(
        video_id=encoded_id,
        api_base="https://api.wavespeed.ai/api/v3",
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )
    assert url == "https://api.wavespeed.ai/api/v3/predictions/task-9/result"
    assert params == {}

    video = config.transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={"data": {"id": "task-9", "status": "completed"}},
        ),
        logging_obj=None,
    )
    assert video.id == "task-9"
    assert video.status == "completed"


def test_wavespeed_video_content_download_returns_bytes() -> None:
    config = WaveSpeedVideoConfig()
    response = httpx.Response(
        200,
        json={
            "data": {
                "id": "task-7",
                "status": "completed",
                "outputs": ["https://cdn.wavespeed.ai/task-7.mp4"],
            }
        },
    )

    captured = {}

    def _fake_download(url: str) -> bytes:
        captured["url"] = url
        return b"mp4-bytes"

    config._download_video = _fake_download  # type: ignore[attr-defined]

    content = config.transform_video_content_response(
        raw_response=response,
        logging_obj=None,
    )

    assert content == b"mp4-bytes"
    assert captured["url"] == "https://cdn.wavespeed.ai/task-7.mp4"


def test_litellm_registers_wavespeed_video_provider() -> None:
    provider_config = litellm.ProviderConfigManager.get_provider_video_config(
        model="bytedance/seedance-2.0-fast",
        provider=litellm.LlmProviders.WAVESPEED,
    )

    assert isinstance(provider_config, WaveSpeedVideoConfig)
