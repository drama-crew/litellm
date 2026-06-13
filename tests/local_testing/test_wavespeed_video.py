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
    assert url == "https://api.wavespeed.ai/api/v3/bytedance/seedance-2.0-fast/image-to-video"
    assert data["prompt"] == "POV camera moves through a neon hotel corridor."
    assert data["image"] == "https://assets.local/first.png"
    assert data["duration"] == 5
    assert data["duration_seconds"] == 5
    assert data["aspect_ratio"] == "9:16"
    assert data["resolution"] == "720p"
    assert data["generate_audio"] is False


def test_wavespeed_video_create_maps_480p_portrait_size_to_480p_resolution() -> None:
    config = WaveSpeedVideoConfig()
    data, _files, _url = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="First-person POV inside a dim old elevator, vertical.",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params={"seconds": "15", "size": "480x854"},
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )

    assert data["aspect_ratio"] == "9:16"
    assert data["resolution"] == "480p"


def test_wavespeed_video_resolution_tiers_floor_on_shorter_side() -> None:
    config = WaveSpeedVideoConfig()
    assert config._resolution("480x854") == "480p"
    assert config._resolution("720x1280") == "720p"
    assert config._resolution("1080x1920") == "1080p"
    assert config._resolution("1920x1080") == "1080p"


def test_wavespeed_map_openai_params_idempotent_preserves_aspect_and_resolution() -> None:
    config = WaveSpeedVideoConfig()
    first = config.map_openai_params(
        {"seconds": "5", "size": "480x854"}, "bytedance/seedance-2.0-fast", drop_params=False
    )
    assert first["aspect_ratio"] == "9:16"
    assert first["resolution"] == "480p"
    second = config.map_openai_params(first, "bytedance/seedance-2.0-fast", drop_params=False)
    assert second["aspect_ratio"] == "9:16"
    assert second["resolution"] == "480p"


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


def test_wavespeed_video_create_defaults_duration_when_seconds_omitted() -> None:
    config = WaveSpeedVideoConfig()
    data, _files, _url = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="POV walking through a rainy neon alley.",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params={
            "input_reference": "https://assets.local/first.png",
        },
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )

    assert data["duration"] == 5
    assert data["duration_seconds"] == 5

    video = config.transform_video_create_response(
        model="bytedance/seedance-2.0-fast",
        raw_response=httpx.Response(200, json={"id": "task-1", "status": "created"}),
        logging_obj=None,
        custom_llm_provider="wavespeed",
        request_data=data,
    )
    assert video.usage["duration_seconds"] == 5.0


def test_wavespeed_video_create_falls_back_to_input_duration_params() -> None:
    config = WaveSpeedVideoConfig()

    top_level, _f, _u = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="A drone shot over a misty forest.",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params={"duration": 8},
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )
    assert top_level["duration"] == 8
    assert top_level["duration_seconds"] == 8

    via_extra_body, _f2, _u2 = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="A drone shot over a misty forest.",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params={"extra_body": {"duration_seconds": "12"}},
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )
    assert via_extra_body["duration"] == 12
    assert via_extra_body["duration_seconds"] == 12


def test_wavespeed_video_create_prefers_seconds_over_input_duration() -> None:
    config = WaveSpeedVideoConfig()
    data, _files, _url = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="A drone shot over a misty forest.",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params={"seconds": "6", "duration": 8},
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )
    assert data["duration"] == 6
    assert data["duration_seconds"] == 6


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


def _create_url(model: str, params: dict, prompt: str = "A cinematic shot.") -> str:
    config = WaveSpeedVideoConfig()
    _data, _files, url = config.transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )
    return url


def _create_data(model: str, params: dict, prompt: str = "A cinematic shot.") -> dict:
    config = WaveSpeedVideoConfig()
    data, _files, _url = config.transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )
    return data


def test_wavespeed_route_honors_model_slug_std() -> None:
    url = _create_url("bytedance/seedance-2.0", {"input_reference": "https://assets.local/first.png"})
    assert url.endswith("/bytedance/seedance-2.0/image-to-video")


def test_wavespeed_route_honors_model_slug_fast() -> None:
    url = _create_url(
        "bytedance/seedance-2.0-fast",
        {"input_reference": "https://assets.local/first.png"},
    )
    assert url.endswith("/bytedance/seedance-2.0-fast/image-to-video")


def test_wavespeed_route_strips_wavespeed_prefix() -> None:
    url = _create_url(
        "wavespeed/bytedance/seedance-2.0-fast",
        {"input_reference": "https://assets.local/first.png"},
    )
    assert url.endswith("/bytedance/seedance-2.0-fast/image-to-video")


def test_wavespeed_route_unknown_model_defaults_std() -> None:
    url = _create_url("bytedance/whatever", {"input_reference": "https://assets.local/first.png"})
    assert url.endswith("/bytedance/seedance-2.0/image-to-video")


def test_wavespeed_text_to_video_route_from_model() -> None:
    data = _create_data(
        "bytedance/seedance-2.0",
        {"reference_images": ["https://assets.local/a.png"]},
    )
    url = _create_url(
        "bytedance/seedance-2.0",
        {"reference_images": ["https://assets.local/a.png"]},
    )
    assert "image" not in data
    assert url.endswith("/bytedance/seedance-2.0/text-to-video")


def test_wavespeed_generate_audio_first_class() -> None:
    data = _create_data("bytedance/seedance-2.0", {"generate_audio": True})
    assert data["generate_audio"] is True


def test_wavespeed_explicit_resolution_without_size() -> None:
    data = _create_data("bytedance/seedance-2.0", {"resolution": "1080p"})
    assert data["resolution"] == "1080p"


def test_wavespeed_explicit_aspect_ratio_overrides_size() -> None:
    data = _create_data("bytedance/seedance-2.0", {"size": "1280x720", "aspect_ratio": "21:9"})
    assert data["aspect_ratio"] == "21:9"
    assert data["resolution"] == "720p"


def test_wavespeed_extra_body_still_overrides_top_level_generate_audio() -> None:
    data = _create_data(
        "bytedance/seedance-2.0",
        {"generate_audio": True, "extra_body": {"generate_audio": False}},
    )
    assert data["generate_audio"] is False


def test_wavespeed_image_to_video_last_image() -> None:
    params = {
        "input_reference": "https://assets.local/first.png",
        "last_image": "https://assets.local/last.png",
    }
    data = _create_data("bytedance/seedance-2.0", params)
    url = _create_url("bytedance/seedance-2.0", params)
    assert data["image"] == "https://assets.local/first.png"
    assert data["last_image"] == "https://assets.local/last.png"
    assert url.endswith("/image-to-video")


def test_wavespeed_image_to_video_no_last_image_key_when_absent() -> None:
    data = _create_data(
        "bytedance/seedance-2.0",
        {"input_reference": "https://assets.local/first.png"},
    )
    assert "last_image" not in data


def test_wavespeed_text_to_video_reference_images() -> None:
    params = {"reference_images": ["https://assets.local/a.png", "https://assets.local/b.png"]}
    data = _create_data("bytedance/seedance-2.0", params)
    url = _create_url("bytedance/seedance-2.0", params)
    assert data["reference_images"] == [
        "https://assets.local/a.png",
        "https://assets.local/b.png",
    ]
    assert "image" not in data
    assert url.endswith("/text-to-video")


def test_wavespeed_text_to_video_reference_audios() -> None:
    params = {
        "reference_images": ["https://assets.local/a.png"],
        "reference_audios": ["https://assets.local/x.mp3"],
    }
    data = _create_data("bytedance/seedance-2.0", params)
    url = _create_url("bytedance/seedance-2.0", params)
    assert data["reference_audios"] == ["https://assets.local/x.mp3"]
    assert url.endswith("/text-to-video")


def test_wavespeed_text_to_video_omits_empty_reference_audios() -> None:
    data = _create_data(
        "bytedance/seedance-2.0",
        {"reference_images": ["https://assets.local/a.png"], "reference_audios": []},
    )
    assert "reference_audios" not in data


def test_wavespeed_image_present_wins_over_reference_images() -> None:
    params = {
        "input_reference": "https://assets.local/first.png",
        "reference_images": ["https://assets.local/a.png"],
    }
    data = _create_data("bytedance/seedance-2.0", params)
    url = _create_url("bytedance/seedance-2.0", params)
    assert data["image"] == "https://assets.local/first.png"
    assert url.endswith("/image-to-video")
    assert "reference_images" not in data


def test_wavespeed_image_to_video_prunes_text_to_video_only_params() -> None:
    params = {
        "input_reference": "https://assets.local/first.png",
        "reference_images": ["https://assets.local/a.png"],
        "reference_audios": ["https://assets.local/x.mp3"],
    }
    data = _create_data("bytedance/seedance-2.0", params)
    url = _create_url("bytedance/seedance-2.0", params)
    assert url.endswith("/image-to-video")
    assert data["image"] == "https://assets.local/first.png"
    assert "reference_images" not in data
    assert "reference_audios" not in data


def test_wavespeed_text_to_video_prunes_last_image() -> None:
    params = {"last_image": "https://assets.local/last.png"}
    data = _create_data("bytedance/seedance-2.0", params)
    url = _create_url("bytedance/seedance-2.0", params)
    assert url.endswith("/text-to-video")
    assert "last_image" not in data


def test_wavespeed_image_first_class_wins_over_input_reference() -> None:
    data = _create_data(
        "bytedance/seedance-2.0",
        {"image": "https://assets.local/A.png", "input_reference": "https://assets.local/B.png"},
    )
    assert data["image"] == "https://assets.local/A.png"


def test_wavespeed_generate_audio_false_is_preserved() -> None:
    data = _create_data("bytedance/seedance-2.0", {"generate_audio": False})
    assert data["generate_audio"] is False


def test_wavespeed_supported_params_include_new_first_class() -> None:
    config = WaveSpeedVideoConfig()
    params = config.get_supported_openai_params("bytedance/seedance-2.0")
    for key in (
        "last_image",
        "reference_images",
        "reference_audios",
        "generate_audio",
        "aspect_ratio",
        "resolution",
    ):
        assert key in params


def _local_cost_map() -> dict:
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    return GetModelCostMap.load_local_model_cost_map()


def test_wavespeed_pricing_std_seedance_entry_exists_with_1080p() -> None:
    entry = _local_cost_map().get("wavespeed/bytedance/seedance-2.0")
    assert entry is not None
    assert entry["mode"] == "video_generation"
    assert entry["litellm_provider"] == "wavespeed"
    assert entry["output_cost_per_video_per_second"] > 0
    assert "1920x1080" in entry["supported_resolutions"]


def test_wavespeed_pricing_fast_entry_has_no_1080p() -> None:
    entry = _local_cost_map().get("wavespeed/bytedance/seedance-2.0-fast")
    assert entry is not None
    assert "1920x1080" not in entry["supported_resolutions"]


# --- Media upload (data: -> WaveSpeed-hosted URL) ---


class _StubResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


class _StubClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, files=None):
        self.calls.append({"url": url, "headers": headers, "files": files})
        return self._responses.pop(0)


class _AsyncStubClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def post(self, url, headers=None, files=None):
        self.calls.append({"url": url, "headers": headers, "files": files})
        return self._responses.pop(0)


_PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgo="


def test_prepare_request_media_uploads_data_url_and_replaces_with_hosted_url():
    config = WaveSpeedVideoConfig()
    client = _StubClient([_StubResponse({"data": {"download_url": "https://ws.host/abc.png"}})])
    params = {"reference_images": [_PNG_DATA_URL, "https://already.hosted/x.png"]}
    out = config.prepare_request_media(
        params, api_key="k", api_base="https://api.wavespeed.ai/api/v3", headers={}, client=client
    )
    assert out["reference_images"] == ["https://ws.host/abc.png", "https://already.hosted/x.png"]
    assert len(client.calls) == 1
    assert client.calls[0]["url"] == "https://api.wavespeed.ai/api/v3/media/upload/binary"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer k"


def test_prepare_request_media_dedups_identical_data_urls():
    config = WaveSpeedVideoConfig()
    client = _StubClient([_StubResponse({"data": {"download_url": "https://ws.host/once.png"}})])
    params = {"reference_images": [_PNG_DATA_URL, _PNG_DATA_URL]}
    out = config.prepare_request_media(
        params, api_key="k", api_base="https://api.wavespeed.ai/api/v3", headers={}, client=client
    )
    assert out["reference_images"] == ["https://ws.host/once.png", "https://ws.host/once.png"]
    assert len(client.calls) == 1


def test_prepare_request_media_fail_open_drops_failed_ref_keeps_rest():
    config = WaveSpeedVideoConfig()
    good = "data:image/png;base64,Z29vZA=="
    client = _StubClient(
        [_StubResponse({}, status=500), _StubResponse({"data": {"download_url": "https://ws.host/good.png"}})]
    )
    params = {"reference_images": [_PNG_DATA_URL, good]}
    out = config.prepare_request_media(
        params, api_key="k", api_base="https://api.wavespeed.ai/api/v3", headers={}, client=client
    )
    assert out["reference_images"] == ["https://ws.host/good.png"]


def test_prepare_request_media_scalar_image_drop_on_failure_routes_text_to_video():
    config = WaveSpeedVideoConfig()
    client = _StubClient([_StubResponse({}, status=500)])
    params = {"image": _PNG_DATA_URL, "reference_images": []}
    out = config.prepare_request_media(
        params, api_key="k", api_base="https://api.wavespeed.ai/api/v3", headers={}, client=client
    )
    assert "image" not in out
    data, _files, url = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="p",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params=out,
        litellm_params=GenericLiteLLMParams(api_key="k"),
        headers={},
    )
    assert url.endswith("/text-to-video")
    assert "image" not in data


def test_prepare_request_media_http_urls_pass_through_no_upload():
    config = WaveSpeedVideoConfig()
    client = _StubClient([])
    params = {"reference_images": ["https://already.hosted/a.png", "https://already.hosted/b.png"]}
    out = config.prepare_request_media(
        params, api_key="k", api_base="https://api.wavespeed.ai/api/v3", headers={}, client=client
    )
    assert out["reference_images"] == ["https://already.hosted/a.png", "https://already.hosted/b.png"]
    assert client.calls == []


def test_prepare_request_media_noop_without_api_key():
    config = WaveSpeedVideoConfig()
    client = _StubClient([])
    params = {"reference_images": [_PNG_DATA_URL]}
    out = config.prepare_request_media(
        params, api_key=None, api_base="https://api.wavespeed.ai/api/v3", headers={}, client=client
    )
    assert out["reference_images"] == [_PNG_DATA_URL]
    assert client.calls == []


def test_require_download_url_fallbacks():
    config = WaveSpeedVideoConfig()
    assert config._require_download_url({"data": {"download_url": "a"}}) == "a"
    assert config._require_download_url({"data": {"url": "b"}}) == "b"
    assert config._require_download_url({"url": "c"}) == "c"
    import pytest

    with pytest.raises(ValueError):
        config._require_download_url({"data": {}})


def test_decode_data_url_extracts_mime_and_bytes():
    config = WaveSpeedVideoConfig()
    name, raw, mime = config._decode_data_url("data:audio/mpeg;base64,aGVsbG8=")
    assert mime == "audio/mpeg"
    assert raw == b"hello"
    assert name.endswith(".mp3")


def test_map_openai_params_normalizes_image_dialect_resolution():
    config = WaveSpeedVideoConfig()
    mapped = config.map_openai_params({"resolution": "1k", "aspect_ratio": "9:16"}, model="m", drop_params=False)
    assert mapped["resolution"] == "720p"
    assert mapped["aspect_ratio"] == "9:16"
    mapped2 = config.map_openai_params({"resolution": "weird"}, model="m", drop_params=False)
    assert "resolution" not in mapped2


def test_map_openai_params_normalizes_resolution_arriving_via_extra_body():
    config = WaveSpeedVideoConfig()
    mapped = config.map_openai_params({"extra_body": {"resolution": "1k"}}, model="m", drop_params=False)
    assert mapped["resolution"] == "720p"
    dropped = config.map_openai_params({"extra_body": {"resolution": "weird"}}, model="m", drop_params=False)
    assert "resolution" not in dropped


import asyncio


def test_async_prepare_request_media_uploads_data_url():
    config = WaveSpeedVideoConfig()
    client = _AsyncStubClient([_StubResponse({"data": {"download_url": "https://ws.host/x.png"}})])
    params = {"reference_images": [_PNG_DATA_URL]}
    out = asyncio.run(
        config.async_prepare_request_media(
            params, api_key="k", api_base="https://api.wavespeed.ai/api/v3", headers={}, client=client
        )
    )
    assert out["reference_images"] == ["https://ws.host/x.png"]
    assert len(client.calls) == 1
