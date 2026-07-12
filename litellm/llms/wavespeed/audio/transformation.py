from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import httpx
from httpx._types import RequestFiles

from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider


_AUDIO_MODELS: frozenset[str] = frozenset(
    (
        "minimax/music-2.6",
        "elevenlabs/music",
        "mureka-ai/mureka-v8/generate-song",
        "mirelo-ai/sfx-1.6/text-to-audio",
        "mirelo-ai/sfx-1.6/video-to-video",
        "kwaivgi/kling-text-to-audio",
        "minimax/voice-clone",
        "minimax/voice-design",
    )
)


def _strip_provider_prefix(model: str) -> str:
    """Remove leading 'wavespeed/' prefix if present."""
    if model.startswith("wavespeed/"):
        return model[len("wavespeed/"):]
    return model


class WaveSpeedAudioConfig(WaveSpeedVideoConfig):
    """
    WaveSpeed async task config for audio models (music, sfx, voice ops).

    Rides the same create→status→content lifecycle as WaveSpeedVideoConfig.
    The key differences:
      - Routing is pure passthrough: /api/v3/{model_path} (no bytedance/ prefix,
        no image-to-video/text-to-video discrimination).
      - No video-specific params are injected (duration, resolution, aspect_ratio).
      - All caller params in extra_body are forwarded to the wire unchanged.
    """

    def map_openai_params(
        self,
        video_create_optional_params: Dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        params = video_create_optional_params
        mapped: Dict[str, Any] = {}
        if params.get("reference_audios"):
            mapped["reference_audios"] = list(params["reference_audios"])
        if params.get("reference_videos"):
            mapped["reference_videos"] = list(params["reference_videos"])
        extra_body = params.get("extra_body") or {}
        if isinstance(extra_body, dict):
            mapped.update(extra_body)
        return mapped

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict[str, Any], RequestFiles, str]:
        params = self.map_openai_params(video_create_optional_request_params, model, drop_params=False)
        data: Dict[str, Any] = {"prompt": prompt, **params}

        # reference_videos is not a WaveSpeed wire field; pop and expose as `video`
        # for video-to-audio models (e.g. mirelo sfx video-to-video).
        reference_videos = data.pop("reference_videos", None)
        if reference_videos:
            data["video"] = reference_videos[0]

        # reference_audios is similarly not a standard WaveSpeed audio wire field;
        # for voice-clone the caller provides `audio` directly via extra_body.
        data.pop("reference_audios", None)

        model_path = _strip_provider_prefix(model)
        url = f"{api_base.rstrip('/')}/{model_path}"
        return data, [], url


    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
    ) -> VideoObject:
        payload = raw_response.json()
        task_id = self._task_id(payload)
        status = self._status(self._raw_status(payload))
        video = VideoObject(id=task_id, object="video", status=status, model=model)
        if custom_llm_provider and video.id:
            video.id = encode_video_id_with_provider(video.id, custom_llm_provider, model)
        # Audio models are billed per-request.  Set duration_seconds=1 so that
        # the cost calculator computes:  cost = output_cost_per_video_per_second * 1
        # (the pricing field stores the flat per-request price).
        video.usage = {"duration_seconds": 1.0}
        return video


def is_wavespeed_audio_model(model: str) -> bool:
    """Return True if the model string refers to one of the 8 supported audio models."""
    path = _strip_provider_prefix(model)
    return path in _AUDIO_MODELS
