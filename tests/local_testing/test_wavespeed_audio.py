from __future__ import annotations

import httpx
import pytest

import litellm
from litellm.llms.wavespeed.audio.transformation import (
    WaveSpeedAudioConfig,
    is_wavespeed_audio_model,
)
from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.videos.utils import VideoGenerationRequestUtils

_BASE = "https://api.wavespeed.ai/api/v3"
_PARAMS = GenericLiteLLMParams(api_key="test-key")


def _submit(model: str, prompt: str = "test", extra: dict | None = None) -> tuple[dict, list, str]:
    config = WaveSpeedAudioConfig()
    return config.transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base=_BASE,
        video_create_optional_request_params=extra or {},
        litellm_params=_PARAMS,
        headers={},
    )


# ---------------------------------------------------------------------------
# is_wavespeed_audio_model helper
# ---------------------------------------------------------------------------


def test_is_wavespeed_audio_model_recognises_all_eight() -> None:
    for path in (
        "minimax/music-2.6",
        "elevenlabs/music",
        "mureka-ai/mureka-v8/generate-song",
        "mirelo-ai/sfx-1.6/text-to-audio",
        "mirelo-ai/sfx-1.6/video-to-video",
        "kwaivgi/kling-text-to-audio",
        "minimax/voice-clone",
        "minimax/voice-design",
    ):
        assert is_wavespeed_audio_model(path), f"expected {path} to be audio"
        assert is_wavespeed_audio_model(f"wavespeed/{path}"), f"expected wavespeed/{path} to be audio"


def test_is_wavespeed_audio_model_rejects_video_models() -> None:
    for path in (
        "bytedance/seedance-2.0",
        "bytedance/seedance-2.0-fast",
        "wavespeed/bytedance/seedance-2.0",
    ):
        assert not is_wavespeed_audio_model(path), f"expected {path} not to be audio"


# ---------------------------------------------------------------------------
# Submit — URL routing (passthrough)
# ---------------------------------------------------------------------------


def test_music_2_6_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("minimax/music-2.6", "A happy pop song")
    assert url == f"{_BASE}/minimax/music-2.6"


def test_music_2_6_with_wavespeed_prefix_strips_prefix() -> None:
    _data, _files, url = _submit("wavespeed/minimax/music-2.6", "A happy pop song")
    assert url == f"{_BASE}/minimax/music-2.6"


def test_eleven_music_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("elevenlabs/music", "A jazzy background track")
    assert url == f"{_BASE}/elevenlabs/music"


def test_mureka_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("mureka-ai/mureka-v8/generate-song", "Epic orchestral")
    assert url == f"{_BASE}/mureka-ai/mureka-v8/generate-song"


def test_mirelo_sfx_text_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("mirelo-ai/sfx-1.6/text-to-audio", "Rainstorm")
    assert url == f"{_BASE}/mirelo-ai/sfx-1.6/text-to-audio"


def test_mirelo_sfx_video_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("mirelo-ai/sfx-1.6/video-to-video", "Match the action")
    assert url == f"{_BASE}/mirelo-ai/sfx-1.6/video-to-video"


def test_kling_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("kwaivgi/kling-text-to-audio", "Sword clash sound")
    assert url == f"{_BASE}/kwaivgi/kling-text-to-audio"


def test_voice_clone_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("minimax/voice-clone", "Preview text")
    assert url == f"{_BASE}/minimax/voice-clone"


def test_voice_design_submit_hits_correct_path() -> None:
    _data, _files, url = _submit("minimax/voice-design", "Preview text")
    assert url == f"{_BASE}/minimax/voice-design"


# ---------------------------------------------------------------------------
# Submit — payload shape
# ---------------------------------------------------------------------------


def test_music_2_6_carries_prompt_and_lyrics() -> None:
    # Params arrive pre-flattened (extra_body already merged by get_optional_params_video_generation)
    data, _files, _url = _submit(
        "minimax/music-2.6",
        "A happy pop song",
        extra={"lyrics": "la la la\noh yeah"},
    )
    assert data["prompt"] == "A happy pop song"
    assert data["lyrics"] == "la la la\noh yeah"


def test_music_2_6_no_video_params_injected() -> None:
    data, _files, _url = _submit("minimax/music-2.6", "happy pop")
    assert "duration" not in data
    assert "duration_seconds" not in data
    assert "resolution" not in data
    assert "aspect_ratio" not in data


def test_voice_clone_carries_audio_and_custom_voice_id() -> None:
    # Params arrive pre-flattened (extra_body already merged by get_optional_params_video_generation)
    data, _files, _url = _submit(
        "minimax/voice-clone",
        "Preview sentence",
        extra={
            "audio": "https://cdn.example.com/sample.mp3",
            "custom_voice_id": "drama_org1_abc123xyz456",
        },
    )
    assert data["audio"] == "https://cdn.example.com/sample.mp3"
    assert data["custom_voice_id"] == "drama_org1_abc123xyz456"
    assert data["prompt"] == "Preview sentence"


def test_voice_design_carries_prompt_and_custom_voice_id() -> None:
    # Params arrive pre-flattened (extra_body already merged by get_optional_params_video_generation)
    data, _files, _url = _submit(
        "minimax/voice-design",
        "A warm, gentle female voice",
        extra={
            "custom_voice_id": "drama_org1_xyz789abc012",
            "text": "Hello, this is a preview.",
        },
    )
    assert data["prompt"] == "A warm, gentle female voice"
    assert data["custom_voice_id"] == "drama_org1_xyz789abc012"
    assert data["text"] == "Hello, this is a preview."


def test_mirelo_sfx_video_to_video_carries_video_from_reference_videos() -> None:
    data, _files, _url = _submit(
        "mirelo-ai/sfx-1.6/video-to-video",
        "Add dramatic music",
        extra={"reference_videos": ["https://cdn.example.com/clip.mp4"]},
    )
    assert data["video"] == "https://cdn.example.com/clip.mp4"
    assert "reference_videos" not in data


def test_mirelo_sfx_reference_videos_uses_first_element() -> None:
    data, _files, _url = _submit(
        "mirelo-ai/sfx-1.6/video-to-video",
        "Add sound",
        extra={"reference_videos": ["https://cdn.example.com/a.mp4", "https://cdn.example.com/b.mp4"]},
    )
    assert data["video"] == "https://cdn.example.com/a.mp4"


def test_audio_files_is_empty_list() -> None:
    _data, files, _url = _submit("minimax/music-2.6", "pop")
    assert files == []


# ---------------------------------------------------------------------------
# Poll / content lifecycle reuses base class (no override needed)
# ---------------------------------------------------------------------------


def test_audio_create_response_sets_duration_1_for_per_request_billing() -> None:
    config = WaveSpeedAudioConfig()
    video = config.transform_video_create_response(
        model="minimax/music-2.6",
        raw_response=httpx.Response(200, json={"id": "task-m1", "status": "created"}),
        logging_obj=None,
        custom_llm_provider="wavespeed",
        request_data={},
    )
    assert video.usage == {"duration_seconds": 1.0}, (
        "audio models need duration_seconds=1 so cost = output_cost_per_video_per_second * 1"
    )


def test_audio_poll_url_is_same_as_video_poll() -> None:
    config = WaveSpeedAudioConfig()
    encoded_id = config.transform_video_create_response(
        model="minimax/music-2.6",
        raw_response=httpx.Response(200, json={"id": "task-audio-1", "status": "created"}),
        logging_obj=None,
        custom_llm_provider="wavespeed",
        request_data={},
    ).id

    url, params = config.transform_video_status_retrieve_request(
        video_id=encoded_id,
        api_base=_BASE,
        litellm_params=_PARAMS,
        headers={},
    )
    assert url == f"{_BASE}/predictions/task-audio-1/result"
    assert params == {}


def test_audio_content_returns_bytes_from_output_url() -> None:
    config = WaveSpeedAudioConfig()
    response = httpx.Response(
        200,
        json={
            "data": {
                "id": "task-audio-2",
                "status": "completed",
                "outputs": ["https://cdn.wavespeed.ai/task-audio-2.mp3"],
            }
        },
    )
    captured: dict[str, str] = {}

    def _fake_download(url: str) -> bytes:
        captured["url"] = url
        return b"mp3-bytes"

    config._download_video = _fake_download  # type: ignore[attr-defined]

    content = config.transform_video_content_response(raw_response=response, logging_obj=None)
    assert content == b"mp3-bytes"
    assert captured["url"] == "https://cdn.wavespeed.ai/task-audio-2.mp3"


def test_audio_status_completed_maps_correctly() -> None:
    config = WaveSpeedAudioConfig()
    video = config.transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={"data": {"id": "task-audio-3", "status": "completed"}},
        ),
        logging_obj=None,
    )
    assert video.status == "completed"
    assert video.id == "task-audio-3"


# ---------------------------------------------------------------------------
# Provider config dispatch
# ---------------------------------------------------------------------------


def test_litellm_dispatches_audio_config_for_music_model() -> None:
    config = litellm.ProviderConfigManager.get_provider_video_config(
        model="minimax/music-2.6",
        provider=litellm.LlmProviders.WAVESPEED,
    )
    assert isinstance(config, WaveSpeedAudioConfig)


def test_litellm_dispatches_audio_config_with_wavespeed_prefix() -> None:
    config = litellm.ProviderConfigManager.get_provider_video_config(
        model="wavespeed/minimax/music-2.6",
        provider=litellm.LlmProviders.WAVESPEED,
    )
    assert isinstance(config, WaveSpeedAudioConfig)


def test_litellm_dispatches_video_config_for_seedance_model() -> None:
    config = litellm.ProviderConfigManager.get_provider_video_config(
        model="bytedance/seedance-2.0-fast",
        provider=litellm.LlmProviders.WAVESPEED,
    )
    assert isinstance(config, WaveSpeedVideoConfig)
    assert not isinstance(config, WaveSpeedAudioConfig)


@pytest.mark.parametrize(
    "model",
    [
        "minimax/music-2.6",
        "elevenlabs/music",
        "mureka-ai/mureka-v8/generate-song",
        "mirelo-ai/sfx-1.6/text-to-audio",
        "mirelo-ai/sfx-1.6/video-to-video",
        "kwaivgi/kling-text-to-audio",
        "minimax/voice-clone",
        "minimax/voice-design",
    ],
)
def test_litellm_dispatches_audio_config_for_all_eight_models(model: str) -> None:
    config = litellm.ProviderConfigManager.get_provider_video_config(
        model=model,
        provider=litellm.LlmProviders.WAVESPEED,
    )
    assert isinstance(config, WaveSpeedAudioConfig), f"expected WaveSpeedAudioConfig for {model}"


# ---------------------------------------------------------------------------
# Pricing entries
# ---------------------------------------------------------------------------


def _cost_map() -> dict:
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    return GetModelCostMap.load_local_model_cost_map()


@pytest.mark.parametrize(
    "model,expected_cost",
    [
        ("wavespeed/minimax/music-2.6", 0.15),
        ("wavespeed/elevenlabs/music", 0.083),
        ("wavespeed/mureka-ai/mureka-v8/generate-song", 0.045),
        ("wavespeed/mirelo-ai/sfx-1.6/text-to-audio", 0.01),
        ("wavespeed/mirelo-ai/sfx-1.6/video-to-video", 0.01),
        ("wavespeed/kwaivgi/kling-text-to-audio", 0.035),
        ("wavespeed/minimax/voice-clone", 0.5),
        ("wavespeed/minimax/voice-design", 0.5),
    ],
)
def test_audio_model_pricing_entry_exists_with_correct_cost(model: str, expected_cost: float) -> None:
    entry = _cost_map().get(model)
    assert entry is not None, f"no pricing entry for {model}"
    assert entry["litellm_provider"] == "wavespeed"
    assert entry["mode"] == "video_generation"
    assert entry["output_cost_per_video_per_second"] == pytest.approx(expected_cost)


# ---------------------------------------------------------------------------
# Full-path regression tests (stage 0-1 flattening → transform)
# These drive the REAL call chain:
#   VideoGenerationRequestUtils.get_optional_params_video_generation
#   → transform_video_create_request
# to ensure audio params survive the extra_body flattening stage and reach the wire.
# ---------------------------------------------------------------------------


def _full_path_submit(
    model: str,
    prompt: str = "test",
    raw_params: dict | None = None,
) -> tuple[dict, list, str]:
    """Drive the same chain litellm uses on the real /v1/videos path."""
    config = WaveSpeedAudioConfig()
    from litellm.types.videos.main import VideoCreateOptionalRequestParams

    optional_params = VideoGenerationRequestUtils.get_optional_params_video_generation(
        model=model,
        video_generation_provider_config=config,
        video_generation_optional_params=VideoCreateOptionalRequestParams(**(raw_params or {})),
    )
    return config.transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base=_BASE,
        video_create_optional_request_params=optional_params,
        litellm_params=_PARAMS,
        headers={},
    )


def test_fullpath_music_2_6_lyrics_reaches_wire() -> None:
    data, _files, _url = _full_path_submit(
        "minimax/music-2.6",
        "A happy pop song",
        raw_params={"extra_body": {"lyrics": "la la la\noh yeah"}},
    )
    assert data["prompt"] == "A happy pop song", "prompt missing"
    assert data.get("lyrics") == "la la la\noh yeah", (
        "lyrics dropped on real /v1/videos path; extra_body was double-mapped"
    )


def test_fullpath_voice_clone_audio_and_custom_voice_id_reach_wire() -> None:
    data, _files, _url = _full_path_submit(
        "minimax/voice-clone",
        "Preview sentence",
        raw_params={
            "extra_body": {
                "audio": "https://cdn.example.com/sample.mp3",
                "custom_voice_id": "drama_org1_abc123xyz456",
            }
        },
    )
    assert data.get("audio") == "https://cdn.example.com/sample.mp3", "audio field dropped"
    assert data.get("custom_voice_id") == "drama_org1_abc123xyz456", "custom_voice_id dropped"


def test_fullpath_voice_design_text_field_reaches_wire() -> None:
    data, _files, _url = _full_path_submit(
        "minimax/voice-design",
        "Warm gentle voice",
        raw_params={
            "extra_body": {
                "custom_voice_id": "drama_org1_xyz789abc012",
                "text": "Hello, this is a preview.",
            }
        },
    )
    assert data.get("text") == "Hello, this is a preview.", "text field dropped"
    assert data.get("custom_voice_id") == "drama_org1_xyz789abc012", "custom_voice_id dropped"


def test_fullpath_no_video_params_leak_for_music_model() -> None:
    data, _files, _url = _full_path_submit("minimax/music-2.6", "happy pop")
    for forbidden in ("duration", "duration_seconds", "resolution", "aspect_ratio"):
        assert forbidden not in data, f"video param {forbidden!r} leaked into audio wire body"


def test_fullpath_reference_videos_become_video_field() -> None:
    data, _files, _url = _full_path_submit(
        "mirelo-ai/sfx-1.6/video-to-video",
        "Add dramatic music",
        raw_params={"reference_videos": ["https://cdn.example.com/clip.mp4"]},
    )
    assert data.get("video") == "https://cdn.example.com/clip.mp4", "reference_videos not mapped to video"
    assert "reference_videos" not in data


def test_fullpath_reference_audios_become_audio_field() -> None:
    data, _files, _url = _full_path_submit(
        "minimax/voice-clone",
        "Clone this voice",
        raw_params={"reference_audios": ["https://cdn.example.com/sample.mp3"]},
    )
    assert data.get("audio") == "https://cdn.example.com/sample.mp3", (
        "reference_audios[0] should be routed to audio field for voice-clone"
    )
    assert "reference_audios" not in data, "reference_audios should not appear on wire"


def test_fullpath_explicit_audio_param_wins_over_reference_audios() -> None:
    data, _files, _url = _full_path_submit(
        "minimax/voice-clone",
        "Clone this voice",
        raw_params={
            "reference_audios": ["https://cdn.example.com/fallback.mp3"],
            "extra_body": {"audio": "https://cdn.example.com/explicit.mp3"},
        },
    )
    assert data.get("audio") == "https://cdn.example.com/explicit.mp3", (
        "explicit audio param in extra_body should override reference_audios[0]"
    )
