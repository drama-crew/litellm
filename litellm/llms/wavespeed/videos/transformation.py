from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm._logging import verbose_logger
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
    Configuration for WaveSpeed Seedance video generation.

    WaveSpeed exposes an async task API:
    1. POST /bytedance/{model_slug}/{image-to-video|text-to-video} creates a prediction task
    2. The task returns a task id immediately
    3. GET /predictions/{task_id}/result polls status and, when complete,
       returns the output video URL(s) which are then downloaded.

    The route is derived from the model id (slug) and whether a first-frame
    image is present: an image routes image-to-video, otherwise text-to-video.
    """

    default_api_base = "https://api.wavespeed.ai/api/v3"
    poll_route = "/predictions/{task_id}/result"
    default_duration_seconds = 5
    default_model_slug = "seedance-2.0"
    allowed_model_slugs = ("seedance-2.0", "seedance-2.0-fast")

    @classmethod
    def _model_slug(cls, model: str) -> str:
        slug = model.split("/")[-1].strip().lower()
        if slug in cls.allowed_model_slugs:
            return slug
        return cls.default_model_slug

    @classmethod
    def _image_to_video_route(cls, model: str) -> str:
        return f"/bytedance/{cls._model_slug(model)}/image-to-video"

    @classmethod
    def _text_to_video_route(cls, model: str) -> str:
        return f"/bytedance/{cls._model_slug(model)}/text-to-video"

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "image",
            "last_image",
            "reference_images",
            "reference_audios",
            "generate_audio",
            "aspect_ratio",
            "resolution",
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
        params = video_create_optional_params
        mapped: Dict[str, Any] = {}
        first_frame = params.get("image") or params.get("input_reference")
        if first_frame:
            mapped["image"] = first_frame
        if params.get("last_image"):
            mapped["last_image"] = params["last_image"]
        if params.get("reference_images"):
            mapped["reference_images"] = list(params["reference_images"])
        if params.get("reference_audios"):
            mapped["reference_audios"] = list(params["reference_audios"])
        if params.get("generate_audio") is not None:
            mapped["generate_audio"] = params["generate_audio"]
        if params.get("size"):
            size = str(params["size"])
            mapped["aspect_ratio"] = self._aspect_ratio(size)
            mapped["resolution"] = self._resolution(size)
        if params.get("aspect_ratio") is not None:
            mapped["aspect_ratio"] = params["aspect_ratio"]
        if params.get("resolution") is not None:
            normalized = self._normalize_resolution(params["resolution"])
            if normalized is not None:
                mapped["resolution"] = normalized
            else:
                mapped.pop("resolution", None)
        extra_body = params.get("extra_body") or {}
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
        if data.get("image"):
            route = self._image_to_video_route(model)
            for key in ("reference_images", "reference_audios"):
                data.pop(key, None)
        else:
            route = self._text_to_video_route(model)
            data.pop("last_image", None)
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

    _upload_route = "/media/upload/binary"
    _media_scalar_keys = ("image", "input_reference", "last_image")
    _media_list_keys = ("reference_images", "reference_audios")

    async def async_prepare_request_media(
        self,
        optional_params: dict,
        *,
        api_key: Optional[str],
        api_base: str,
        headers: dict,
        client: AsyncHTTPHandler,
    ) -> dict:
        key = api_key or get_secret_str("WAVESPEED_API_KEY")
        if not key:
            return optional_params
        upload_url = f"{api_base.rstrip('/')}{self._upload_route}"
        mapping: Dict[str, Optional[str]] = {}
        for value in self._collect_data_urls(optional_params):
            if value in mapping:
                continue
            try:
                name, raw, mime = self._decode_data_url(value)
                resp = await client.post(
                    upload_url,
                    headers={"Authorization": self._bearer(key)},
                    files={"file": (name, raw, mime)},
                )
                resp.raise_for_status()
                mapping[value] = self._require_download_url(resp.json())
            except Exception as e:
                verbose_logger.warning("WaveSpeed media upload failed, dropping reference: %s", e)
                mapping[value] = None
        return self._apply_media_mapping(optional_params, mapping)

    def prepare_request_media(
        self,
        optional_params: dict,
        *,
        api_key: Optional[str],
        api_base: str,
        headers: dict,
        client: HTTPHandler,
    ) -> dict:
        key = api_key or get_secret_str("WAVESPEED_API_KEY")
        if not key:
            return optional_params
        upload_url = f"{api_base.rstrip('/')}{self._upload_route}"
        mapping: Dict[str, Optional[str]] = {}
        for value in self._collect_data_urls(optional_params):
            if value in mapping:
                continue
            try:
                name, raw, mime = self._decode_data_url(value)
                resp = client.post(
                    upload_url,
                    headers={"Authorization": self._bearer(key)},
                    files={"file": (name, raw, mime)},
                )
                resp.raise_for_status()
                mapping[value] = self._require_download_url(resp.json())
            except Exception as e:
                verbose_logger.warning("WaveSpeed media upload failed, dropping reference: %s", e)
                mapping[value] = None
        return self._apply_media_mapping(optional_params, mapping)

    def _collect_data_urls(self, optional_params: dict) -> list:
        out: list = []
        for k in self._media_scalar_keys:
            v = optional_params.get(k)
            if isinstance(v, str) and v.startswith("data:"):
                out.append(v)
        for k in self._media_list_keys:
            for v in optional_params.get(k) or []:
                if isinstance(v, str) and v.startswith("data:"):
                    out.append(v)
        return out

    def _apply_media_mapping(self, optional_params: dict, mapping: Dict[str, Optional[str]]) -> dict:
        for k in self._media_scalar_keys:
            v = optional_params.get(k)
            if isinstance(v, str) and v in mapping:
                hosted = mapping[v]
                if hosted is None:
                    optional_params.pop(k, None)
                else:
                    optional_params[k] = hosted
        for k in self._media_list_keys:
            vals = optional_params.get(k)
            if not isinstance(vals, list):
                continue
            rebuilt = [mapping[v] if isinstance(v, str) and v in mapping else v for v in vals]
            rebuilt = [v for v in rebuilt if v is not None]
            if rebuilt:
                optional_params[k] = rebuilt
            else:
                optional_params.pop(k, None)
        return optional_params

    @staticmethod
    def _bearer(key: str) -> str:
        return key if str(key).lower().startswith("bearer ") else f"Bearer {key}"

    @staticmethod
    def _decode_data_url(value: str) -> Tuple[str, bytes, str]:
        import base64

        header, _, b64 = value.partition(",")
        mime = "application/octet-stream"
        if header.startswith("data:"):
            mime = header[len("data:") :].split(";", 1)[0] or mime
        raw = base64.b64decode(b64)
        ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
            "audio/mpeg": "mp3",
            "audio/mp4": "m4a",
            "audio/wav": "wav",
        }.get(mime, "bin")
        return f"reference.{ext}", raw, mime

    @staticmethod
    def _require_download_url(payload: Any) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        candidates = [
            data.get("download_url") if isinstance(data, dict) else None,
            data.get("url") if isinstance(data, dict) else None,
            payload.get("url") if isinstance(payload, dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return candidate
        raise ValueError("no download_url in WaveSpeed upload response")

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
    def _normalize_resolution(value: Any) -> Optional[str]:
        token = str(value).strip().lower()
        if token in ("480p", "720p", "1080p"):
            return token
        return {"1k": "720p", "2k": "1080p", "4k": "1080p"}.get(token)

    @staticmethod
    def _resolution(size: str) -> str:
        width, height = (int(part) for part in size.lower().split("x", 1))
        shorter = min(width, height)
        if shorter <= 480:
            return "480p"
        if shorter <= 720:
            return "720p"
        return "1080p"
