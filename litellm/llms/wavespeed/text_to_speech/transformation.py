from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


_TTS_PASSTHROUGH_KEYS: frozenset[str] = frozenset(
    (
        "voice_id",
        "emotion",
        "pitch",
        "volume",
        "language_boost",
        "sample_rate",
        "bitrate",
        "channel",
        "pronunciation_dict",
        "english_normalization",
    )
)

_SPEECH_MODELS: frozenset[str] = frozenset(
    (
        "minimax-speech-2.8-turbo",
        "minimax-speech-2.8-hd",
    )
)


def is_wavespeed_speech_model(model: str) -> bool:
    path = model[len("openai/") :] if model.startswith("openai/") else model
    return path in _SPEECH_MODELS


class WaveSpeedTextToSpeechConfig(BaseTextToSpeechConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["response_format", "speed", "instructions", *sorted(_TTS_PASSTHROUGH_KEYS)]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict = {},
    ) -> tuple[str | None, dict]:
        passthrough = {key: kwargs[key] for key in _TTS_PASSTHROUGH_KEYS if kwargs.get(key) is not None}
        existing_extra_body = kwargs.get("extra_body")
        extra_body = {**existing_extra_body, **passthrough} if isinstance(existing_extra_body, dict) else passthrough
        mapped_params = {**optional_params, "extra_body": extra_body} if extra_body else dict(optional_params)
        return voice, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        validated_headers = {**headers, "Content-Type": "application/json"}
        if api_key:
            validated_headers["Authorization"] = f"Bearer {api_key}"
        return validated_headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required for WaveSpeed text-to-speech requests.")
        return f"{api_base.rstrip('/')}/audio/speech"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        params = dict(optional_params)
        extra_body = params.pop("extra_body", None)
        dict_body = {"model": model, "input": input, "voice": voice}
        for key, value in params.items():
            if value is not None:
                dict_body[key] = value
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None:
                    dict_body[key] = value
        return TextToSpeechRequestData(dict_body=dict_body, headers=headers)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(raw_response)
