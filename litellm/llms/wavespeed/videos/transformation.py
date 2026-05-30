from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)


class WaveSpeedVideoConfig(BaseVideoConfig):
    """
    Configuration for WaveSpeed Seedance image-to-video generation.

    WaveSpeed exposes an async task API:
    1. POST /bytedance/seedance-2.0/image-to-video creates a prediction task
    2. The task returns a task id immediately
    3. GET /predictions/{task_id}/result polls status and, when complete,
       returns the output video URL(s) which are then downloaded.
    """

    default_api_base = "https://api.wavespeed.ai/api/v3"
    image_to_video_route = "/bytedance/seedance-2.0/image-to-video"
    text_to_video_route = "/bytedance/seedance-2.0/text-to-video"
    poll_route = "/predictions/{task_id}/result"
    default_duration_seconds = 5

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_body",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped: Dict[str, Any] = {}
        if video_create_optional_params.get("input_reference"):
            mapped["image"] = video_create_optional_params["input_reference"]
        if video_create_optional_params.get("size"):
            size = str(video_create_optional_params["size"])
            mapped["aspect_ratio"] = self._aspect_ratio(size)
            mapped["resolution"] = self._resolution(size)
        extra_body = video_create_optional_params.get("extra_body") or {}
        if isinstance(extra_body, dict):
            mapped.update(extra_body)
        duration = self._resolve_duration_seconds(video_create_optional_params, mapped)
        mapped["duration"] = duration
        mapped["duration_seconds"] = duration
        return mapped

    @classmethod
    def _resolve_duration_seconds(cls, params: Dict[str, Any], mapped: Dict[str, Any]) -> int:
        for value in (
            params.get("seconds"),
            params.get("duration"),
            params.get("duration_seconds"),
            mapped.get("duration"),
            mapped.get("duration_seconds"),
        ):
            if value is not None:
                return int(float(str(value)))
        return cls.default_duration_seconds

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        key = api_key or (litellm_params.api_key if litellm_params else None) or get_secret_str("WAVESPEED_API_KEY")
        if not key:
            raise ValueError("WAVESPEED_API_KEY is required for WaveSpeed video generation")
        auth = key if str(key).lower().startswith("bearer ") else f"Bearer {key}"
        headers.update({"Authorization": auth, "Content-Type": "application/json"})
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return (api_base or self.default_api_base).rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        params = self.map_openai_params(video_create_optional_request_params, model, drop_params=False)
        data: Dict[str, Any] = {"prompt": prompt, **params}
        route = self.image_to_video_route if data.get("image") else self.text_to_video_route
        return data, [], f"{api_base.rstrip('/')}{route}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        payload = raw_response.json()
        task_id = self._task_id(payload)
        status = self._status(self._raw_status(payload))
        video = VideoObject(id=task_id, object="video", status=status, model=model)
        if custom_llm_provider and video.id:
            video.id = encode_video_id_with_provider(video.id, custom_llm_provider, model)
        request_data = request_data or {}
        usage: Dict[str, Any] = {}
        duration = request_data.get("duration") or request_data.get("duration_seconds")
        if duration is not None:
            usage["duration_seconds"] = float(duration)
        resolution = request_data.get("resolution")
        if resolution:
            usage["video_resolution"] = str(resolution).lower()
        video.usage = usage
        return video

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        task_id = extract_original_video_id(video_id)
        return (
            f"{api_base.rstrip('/')}{self.poll_route.format(task_id=task_id)}",
            {},
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        payload = raw_response.json()
        task_id = self._task_id(payload)
        status = self._status(self._raw_status(payload))
        video = VideoObject(id=task_id, object="video", status=status)
        if custom_llm_provider and video.id:
            video.id = encode_video_id_with_provider(video.id, custom_llm_provider, None)
        return video

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        task_id = extract_original_video_id(video_id)
        return (
            f"{api_base.rstrip('/')}{self.poll_route.format(task_id=task_id)}",
            {},
        )

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> bytes:
        video_url = self._extract_output_url(raw_response.json())
        return self._download_video(video_url)

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> bytes:
        video_url = self._extract_output_url(raw_response.json())
        async_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.WAVESPEED,
        )
        video_response = await async_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def _download_video(self, video_url: str) -> bytes:
        client: HTTPHandler = _get_httpx_client()
        video_response = client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("WaveSpeed video remix is not supported")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("WaveSpeed video remix is not supported")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("WaveSpeed video listing is not supported")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("WaveSpeed video listing is not supported")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("WaveSpeed video delete is not supported")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> VideoObject:
        raise NotImplementedError("WaveSpeed video delete is not supported")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    @staticmethod
    def _task_id(payload: Dict[str, Any]) -> str:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        return str(payload.get("id") or payload.get("task_id") or data.get("id") or "")

    @staticmethod
    def _raw_status(payload: Dict[str, Any]) -> str:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        return str(payload.get("status") or data.get("status") or "queued")

    @staticmethod
    def _extract_output_url(payload: Dict[str, Any]) -> str:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        outputs = payload.get("outputs") or payload.get("output") or data.get("outputs") or data.get("output")
        url: Optional[str] = None
        if isinstance(outputs, list) and outputs:
            url = outputs[0]
        elif isinstance(outputs, str):
            url = outputs
        if not url:
            status = WaveSpeedVideoConfig._status(WaveSpeedVideoConfig._raw_status(payload))
            if status in {"queued", "in_progress"}:
                raise ValueError(f"WaveSpeed video is still processing (status: {status}). Please wait and try again.")
            raise ValueError("WaveSpeed video output URL not found in response. Video may not be ready yet.")
        return url

    @staticmethod
    def _status(status: str) -> str:
        normalized = status.lower()
        if normalized in {"created", "queued", "pending"}:
            return "queued"
        if normalized in {"processing", "running"}:
            return "in_progress"
        if normalized in {"completed", "succeeded", "success"}:
            return "completed"
        if normalized in {"failed", "error"}:
            return "failed"
        return "queued"

    @staticmethod
    def _aspect_ratio(size: str) -> str:
        width, height = (int(part) for part in size.lower().split("x", 1))
        return "9:16" if height >= width else "16:9"

    @staticmethod
    def _resolution(size: str) -> str:
        width, height = (int(part) for part in size.lower().split("x", 1))
        return "720p" if min(width, height) <= 720 else "1080p"
