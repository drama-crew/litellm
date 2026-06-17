import time
from typing import Any, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.videos.main import VideoObject

from .client import LibTVClient
from .common import resolve_libtv_credentials
from .transform import build_generation_params


def _project_name(model: str) -> str:
    return f"litellm-{model}-{int(time.time())}"


def _resolve_mode(optional_params: dict, default_mode: str) -> str:
    return optional_params.get("modeType") or optional_params.get("mode_type") or default_mode


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
        params = build_generation_params(prompt, optional_params, spec, _resolve_mode(optional_params, "text2video"))
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
        params = build_generation_params(prompt, optional_params, spec, _resolve_mode(optional_params, "text2video"))
        result = await lt.agenerate(model, spec["vendor"], "video", params, _project_name(model))
        return self._build_video_object(model, result)
