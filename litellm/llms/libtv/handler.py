import os
import time
from typing import Any, Optional, Tuple, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

LIBTV_PROVIDER = "libtv"

_REF_DEFAULT_NAME = {"image": "reference.png", "video": "reference.mp4", "audio": "reference.mp3"}

from litellm.exceptions import ContentPolicyViolationError

from .client import LibTVClient
from .common import LibTVContentPolicyError, LibTVError, resolve_libtv_credentials
from .transform import _resolution_from_size, build_generation_params


def _as_content_policy(exc: LibTVContentPolicyError, model: str) -> ContentPolicyViolationError:
    return ContentPolicyViolationError(message=exc.message, model=model, llm_provider=LIBTV_PROVIDER)


def _project_name(model: str) -> str:
    return f"litellm-{model}-{int(time.time())}"


def _video_usage(optional_params: dict) -> Optional[dict]:
    """duration_seconds + video_resolution for cost calc; None when no duration."""
    try:
        duration_seconds = float(optional_params.get("seconds") or optional_params.get("duration"))
    except (TypeError, ValueError):
        return None
    usage: dict = {"duration_seconds": duration_seconds}
    resolution = optional_params.get("resolution") or _resolution_from_size(optional_params.get("size"))
    if resolution:
        usage["video_resolution"] = resolution
    return usage


def _resolve_mode(optional_params: dict, default_mode: str) -> str:
    return optional_params.get("modeType") or optional_params.get("mode_type") or default_mode


def _image_clarity_response_cost(optional_params: dict, quality: Optional[str], image_count: int) -> Optional[float]:
    """Authoritative accrued spend for clarity-tiered libtv image models (e.g. nano-banana-pro).

    Deployment config declares per-tier unit prices as litellm_params keys
    (``output_cost_per_image_1k``/``_2k``/``_4k``), mirroring how libtv video
    deployments declare ``output_cost_per_second_<resolution>`` in model_info.
    Router deployments spread litellm_params into image_generation()'s kwargs;
    any key litellm doesn't recognize as a first-class param is merged into
    optional_params for custom providers (see
    add_provider_specific_params_to_optional_params in litellm/utils.py),
    which is how these tier prices reach this handler. When a model's spec has
    no "quality" setting (and thus build_generation_params never sets
    ``quality``) or the deployment declares no tier keys, this returns None
    and callers must leave litellm's normal cost-calculator matrix in charge.
    """
    if not quality or image_count <= 0:
        return None
    tier_key = f"output_cost_per_image_{quality.strip().lower()}"
    unit_price = optional_params.get(tier_key)
    if unit_price is None:
        return None
    try:
        return float(unit_price) * image_count
    except (TypeError, ValueError):
        return None


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
    # Accept both libtv-native keys and the wavespeed-shaped keys the drama
    # platform sends (reference_images / image / last_image / reference_audios),
    # so the same payload works whether the request lands on libtv or wavespeed.
    images = (
        _as_list(optional_params.get("input_reference"))
        + _as_list(optional_params.get("image_references"))
        + _as_list(optional_params.get("reference_images"))
        + _as_list(optional_params.get("image"))
        + _as_list(optional_params.get("last_image"))
    )
    videos = _as_list(optional_params.get("video_references")) + _as_list(optional_params.get("reference_videos"))
    audios = _as_list(optional_params.get("audio_references")) + _as_list(optional_params.get("reference_audios"))
    return images, videos, audios


def _default_video_mode(images: list, videos: list, audios: list) -> str:
    if audios:
        return "audio2video"
    if videos:
        return "video2video"
    if images:
        return "image2video"
    return "text2video"


def _infer_video_mode(optional_params: dict, images: list, videos: list, audios: list) -> str:
    # wavespeed first/last-frame request: image (first) + last_image (last).
    if optional_params.get("last_image"):
        return "frames2video"
    return _default_video_mode(images, videos, audios)


def _infer_image_mode(images: list) -> str:
    return "image2image" if images else "text2image"


def _auto_compliance_enabled(spec: dict) -> bool:
    # Portrait-capable models (e.g. star-video2) reject raw reference-image URLs that
    # contain a real person; the upstream verify flow must run and convert each image
    # to an ``asset://`` id before generation. The model schema advertises this.
    return bool(((spec.get("properties") or {}).get("autoCompliance") or {}).get("enable"))


class LibTVLLM(CustomLLM):
    def __init__(self, poll_interval: float = 3.0, poll_max_attempts: int = 200, http_get=None, http_put=None):
        super().__init__()
        self.poll_interval = poll_interval
        self.poll_max_attempts = poll_max_attempts
        self._http_get = http_get
        self._http_put = http_put

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
            http_get=self._http_get,
            http_put=self._http_put,
        )

    def _build_video_object(self, model: str, result: dict, optional_params: Optional[dict] = None) -> VideoObject:
        urls = result.get("urls") or []
        url = urls[0] if urls else ""
        # Encode the result url + provider into the video id so the proxy routes
        # subsequent /v1/videos/{id} status and /content calls back to libtv.
        video_id = encode_video_id_with_provider(url, LIBTV_PROVIDER) if url else result.get("task_id", "")
        vo = VideoObject(id=video_id, object="video", status="completed", model=model)
        vo.usage = _video_usage(optional_params or {})
        vo._hidden_params = {
            "libtv_video_urls": urls,
            "url": url or None,
            "project_uuid": result.get("project_uuid"),
        }
        return vo

    @staticmethod
    def _decode_result_url(video_id: str) -> str:
        url = (decode_video_id_with_provider(video_id) or {}).get("video_id") or ""
        if not url.startswith("http"):
            raise LibTVError(status_code=400, message="libtv video id does not carry a result url")
        return url

    def video_status(
        self,
        video_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> VideoObject:
        url = self._decode_result_url(video_id)
        vo = VideoObject(id=video_id, object="video", status="completed")
        vo._hidden_params = {"url": url, "libtv_video_urls": [url]}
        return vo

    async def avideo_status(
        self,
        video_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> VideoObject:
        return self.video_status(video_id, api_key, api_base, optional_params, logging_obj, timeout, None)

    def video_content(
        self,
        video_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> bytes:
        url = self._decode_result_url(video_id)
        http = client or HTTPHandler()
        resp = http.get(url=url)
        if resp.status_code != 200:
            raise LibTVError(status_code=resp.status_code, message="libtv video content download failed")
        return resp.content

    async def avideo_content(
        self,
        video_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> bytes:
        url = self._decode_result_url(video_id)
        http = client or AsyncHTTPHandler()
        resp = await http.get(url=url)
        if resp.status_code != 200:
            raise LibTVError(status_code=resp.status_code, message="libtv video content download failed")
        return resp.content

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

    def _fill_image_response(
        self,
        model_response: ImageResponse,
        result: dict,
        optional_params: Optional[dict] = None,
        quality: Optional[str] = None,
    ) -> ImageResponse:
        urls = result.get("urls") or []
        model_response.data = [ImageObject(url=u) for u in urls]
        hidden_params: dict = {"project_uuid": result.get("project_uuid")}
        if optional_params is not None:
            response_cost = _image_clarity_response_cost(optional_params, quality, len(urls))
            if response_cost is not None:
                hidden_params["response_cost"] = response_cost
        model_response._hidden_params = hidden_params
        return model_response

    def _resolved_image_params(
        self, lt: LibTVClient, prompt: str, spec: dict, images: list, optional_params: dict
    ) -> dict:
        mode = _resolve_mode(optional_params, _infer_image_mode(images))
        params = build_generation_params(prompt, optional_params, spec, mode)
        if images:
            if _auto_compliance_enabled(spec):
                params["autoCompliance"] = 1
                params["imageList"] = lt.resolve_compliant_image_refs([_reference_payload(r) for r in images])
            else:
                params["imageList"] = [
                    lt.ensure_libtv_url(*_reference_payload(r), _REF_DEFAULT_NAME["image"]) for r in images
                ]
        return params

    async def _aresolved_image_params(
        self, lt: LibTVClient, prompt: str, spec: dict, images: list, optional_params: dict
    ) -> dict:
        mode = _resolve_mode(optional_params, _infer_image_mode(images))
        params = build_generation_params(prompt, optional_params, spec, mode)
        if images:
            if _auto_compliance_enabled(spec):
                params["autoCompliance"] = 1
                params["imageList"] = await lt.aresolve_compliant_image_refs([_reference_payload(r) for r in images])
            else:
                params["imageList"] = [
                    await lt.aensure_libtv_url(*_reference_payload(r), _REF_DEFAULT_NAME["image"]) for r in images
                ]
        return params

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
        images, _, _ = _collect_reference_groups(optional_params)
        params = self._resolved_image_params(lt, prompt, spec, images, optional_params)
        result = lt.generate(model, spec["vendor"], "image", params, _project_name(model))
        return self._fill_image_response(model_response, result, optional_params, params.get("quality"))

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
        images, _, _ = _collect_reference_groups(optional_params)
        params = await self._aresolved_image_params(lt, prompt, spec, images, optional_params)
        result = await lt.agenerate(model, spec["vendor"], "image", params, _project_name(model))
        return self._fill_image_response(model_response, result, optional_params, params.get("quality"))

    def image_edit(
        self,
        model: str,
        image: Any,
        prompt: Optional[str],
        model_response: ImageResponse,
        api_key: Optional[str],
        api_base: Optional[str],
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ImageResponse:
        # litellm's OpenAI-shaped /v1/images/edits routes here with the uploaded
        # reference file(s) as `image` (single file or list) rather than
        # optional_params["reference_images"]/["image"] the way image_generation's
        # JSON-body callers do; the reference upload/compliance flow is identical
        # from here on, just fed a differently-shaped reference list.
        lt = self._make_client(api_key, optional_params, sync_client=client or HTTPHandler())
        spec = lt.resolve_model_spec(model)
        images = _as_list(image)
        params = self._resolved_image_params(lt, prompt or "", spec, images, optional_params)
        result = lt.generate(model, spec["vendor"], "image", params, _project_name(model))
        return self._fill_image_response(model_response, result, optional_params, params.get("quality"))

    async def aimage_edit(
        self,
        model: str,
        image: Any,
        prompt: Optional[str],
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
        images = _as_list(image)
        params = await self._aresolved_image_params(lt, prompt or "", spec, images, optional_params)
        result = await lt.agenerate(model, spec["vendor"], "image", params, _project_name(model))
        return self._fill_image_response(model_response, result, optional_params, params.get("quality"))

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

        def url_for(ref, default_name):
            p = _reference_payload(ref)
            return lt.ensure_libtv_url(p[0], p[1], p[2], default_name)

        if images and _auto_compliance_enabled(spec):
            image_refs = lt.resolve_compliant_image_refs([_reference_payload(r) for r in images])
            video_refs = lt.resolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "mixed2video")
            params["autoCompliance"] = 1
            params["mixedList"] = (
                [{"url": r, "type": "image"} for r in image_refs]
                + [{"url": r, "type": "video"} for r in video_refs]
                + [{"url": url_for(r, _REF_DEFAULT_NAME["audio"]), "type": "audio"} for r in audios]
            )
        else:
            mode = _resolve_mode(optional_params, _infer_video_mode(optional_params, images, videos, audios))
            params = build_generation_params(prompt, optional_params, spec, mode)
            self._apply_video_references(
                params,
                mode,
                [url_for(r, _REF_DEFAULT_NAME["image"]) for r in images],
                [url_for(r, _REF_DEFAULT_NAME["video"]) for r in videos],
                [url_for(r, _REF_DEFAULT_NAME["audio"]) for r in audios],
            )
        try:
            result = lt.generate(model, spec["vendor"], "video", params, _project_name(model))
        except LibTVContentPolicyError as e:
            raise _as_content_policy(e, model) from e
        return self._build_video_object(model, result, optional_params)

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

        async def url_for(ref, default_name):
            p = _reference_payload(ref)
            return await lt.aensure_libtv_url(p[0], p[1], p[2], default_name)

        if images and _auto_compliance_enabled(spec):
            image_refs = await lt.aresolve_compliant_image_refs([_reference_payload(r) for r in images])
            video_refs = await lt.aresolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "mixed2video")
            params["autoCompliance"] = 1
            params["mixedList"] = (
                [{"url": r, "type": "image"} for r in image_refs]
                + [{"url": r, "type": "video"} for r in video_refs]
                + [{"url": await url_for(r, _REF_DEFAULT_NAME["audio"]), "type": "audio"} for r in audios]
            )
        else:
            mode = _resolve_mode(optional_params, _infer_video_mode(optional_params, images, videos, audios))
            params = build_generation_params(prompt, optional_params, spec, mode)
            self._apply_video_references(
                params,
                mode,
                [await url_for(r, _REF_DEFAULT_NAME["image"]) for r in images],
                [await url_for(r, _REF_DEFAULT_NAME["video"]) for r in videos],
                [await url_for(r, _REF_DEFAULT_NAME["audio"]) for r in audios],
            )
        try:
            result = await lt.agenerate(model, spec["vendor"], "video", params, _project_name(model))
        except LibTVContentPolicyError as e:
            raise _as_content_policy(e, model) from e
        return self._build_video_object(model, result, optional_params)
