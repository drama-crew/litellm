import os
import time
from typing import Any, Optional, Tuple, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.videos.main import VideoObject

from .client import LibTVClient
from .common import LibTVError, resolve_libtv_credentials
from .transform import build_generation_params


def _project_name(model: str) -> str:
    return f"litellm-{model}-{int(time.time())}"


def _resolve_mode(optional_params: dict, default_mode: str) -> str:
    return optional_params.get("modeType") or optional_params.get("mode_type") or default_mode


def _reference_payload(ref: Any) -> Optional[Tuple[str, str, Optional[bytes]]]:
    """Normalize a litellm input_reference into ('url', url, None) or ('bytes', filename, data)."""
    if ref is None:
        return None
    if isinstance(ref, str):
        if ref.startswith("http://") or ref.startswith("https://"):
            return ("url", ref, None)
        with open(ref, "rb") as f:
            return ("bytes", os.path.basename(ref) or "reference.png", f.read())
    if isinstance(ref, (bytes, bytearray)):
        return ("bytes", "reference.png", bytes(ref))
    if isinstance(ref, tuple) and len(ref) >= 2:
        body = ref[1]
        data = body.read() if hasattr(body, "read") else (bytes(body) if isinstance(body, (bytes, bytearray)) else None)
        if data is not None:
            return ("bytes", ref[0] or "reference.png", data)
    if hasattr(ref, "read"):
        return ("bytes", getattr(ref, "name", "reference.png"), ref.read())
    raise LibTVError(status_code=400, message=f"unsupported input_reference type: {type(ref).__name__}")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and not (
        len(value) >= 2 and isinstance(value[0], str) and not isinstance(value[1], str)
    ):
        return list(value)
    return [value]


def _collect_reference_groups(optional_params: dict) -> Tuple[list, list, list]:
    images = _as_list(optional_params.get("input_reference")) + _as_list(optional_params.get("image_references"))
    videos = _as_list(optional_params.get("video_references"))
    audios = _as_list(optional_params.get("audio_references"))
    return images, videos, audios


def _default_video_mode(images: list, videos: list, audios: list) -> str:
    if audios:
        return "audio2video"
    if videos:
        return "video2video"
    if images:
        return "image2video"
    return "text2video"


class LibTVLLM(CustomLLM):
    def __init__(self, poll_interval: float = 3.0, poll_max_attempts: int = 200):
        super().__init__()
        self.poll_interval = poll_interval
        self.poll_max_attempts = poll_max_attempts

    def _make_client(
        self,
        api_key: Optional[str],
        optional_params: dict,
        sync_client: Optional[HTTPHandler] = None,
        async_client: Optional[AsyncHTTPHandler] = None,
    ) -> LibTVClient:
        token, webid = resolve_libtv_credentials(token=api_key, webid=optional_params.get("webid"))
        return LibTVClient(
            token=token,
            webid=webid,
            sync_client=sync_client,
            async_client=async_client,
            poll_interval=self.poll_interval,
            poll_max_attempts=self.poll_max_attempts,
        )

    def _build_video_object(self, model: str, result: dict) -> VideoObject:
        urls = result.get("urls") or []
        vo = VideoObject(id=result.get("task_id", ""), object="video", status="completed", model=model)
        vo._hidden_params = {
            "libtv_video_urls": urls,
            "url": urls[0] if urls else None,
            "project_uuid": result.get("project_uuid"),
        }
        return vo

    @staticmethod
    def _apply_video_references(params: dict, mode: str, img_urls: list, vid_urls: list, aud_urls: list) -> None:
        if img_urls:
            params["imageList"] = img_urls
        if vid_urls:
            params["videoList"] = vid_urls
        if aud_urls:
            params["audioList"] = aud_urls
        if mode == "mixed2video":
            params["mixedList"] = (
                [{"url": u, "type": "image"} for u in img_urls]
                + [{"url": u, "type": "video"} for u in vid_urls]
                + [{"url": u, "type": "audio"} for u in aud_urls]
            )

    def _fill_image_response(self, model_response: ImageResponse, result: dict) -> ImageResponse:
        model_response.data = [ImageObject(url=u) for u in (result.get("urls") or [])]
        model_response._hidden_params = {"project_uuid": result.get("project_uuid")}
        return model_response

    def image_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ImageResponse:
        lt = self._make_client(api_key, optional_params, sync_client=client or HTTPHandler())
        spec = lt.resolve_model_spec(model)
        params = build_generation_params(prompt, optional_params, spec, _resolve_mode(optional_params, "text2image"))
        result = lt.generate(model, spec["vendor"], "image", params, _project_name(model))
        return self._fill_image_response(model_response, result)

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        lt = self._make_client(api_key, optional_params, async_client=client or AsyncHTTPHandler())
        spec = await lt.aresolve_model_spec(model)
        params = build_generation_params(prompt, optional_params, spec, _resolve_mode(optional_params, "text2image"))
        result = await lt.agenerate(model, spec["vendor"], "image", params, _project_name(model))
        return self._fill_image_response(model_response, result)

    def video_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> VideoObject:
        lt = self._make_client(api_key, optional_params, sync_client=client or HTTPHandler())
        spec = lt.resolve_model_spec(model)
        images, videos, audios = _collect_reference_groups(optional_params)
        mode = _resolve_mode(optional_params, _default_video_mode(images, videos, audios))
        params = build_generation_params(prompt, optional_params, spec, mode)

        def url_for(ref):
            p = _reference_payload(ref)
            return p[1] if p[0] == "url" else lt.upload_media(p[2], p[1])

        self._apply_video_references(
            params, mode, [url_for(r) for r in images], [url_for(r) for r in videos], [url_for(r) for r in audios]
        )
        result = lt.generate(model, spec["vendor"], "video", params, _project_name(model))
        return self._build_video_object(model, result)

    async def avideo_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> VideoObject:
        lt = self._make_client(api_key, optional_params, async_client=client or AsyncHTTPHandler())
        spec = await lt.aresolve_model_spec(model)
        images, videos, audios = _collect_reference_groups(optional_params)
        mode = _resolve_mode(optional_params, _default_video_mode(images, videos, audios))
        params = build_generation_params(prompt, optional_params, spec, mode)

        async def url_for(ref):
            p = _reference_payload(ref)
            return p[1] if p[0] == "url" else await lt.aupload_media(p[2], p[1])

        self._apply_video_references(
            params,
            mode,
            [await url_for(r) for r in images],
            [await url_for(r) for r in videos],
            [await url_for(r) for r in audios],
        )
        result = await lt.agenerate(model, spec["vendor"], "video", params, _project_name(model))
        return self._build_video_object(model, result)
