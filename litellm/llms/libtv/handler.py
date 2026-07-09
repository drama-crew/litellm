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

from .client import LibTVClient
from .common import LibTVError, resolve_libtv_credentials
from .transform import _resolution_from_size, build_generation_params

# libtv progress status -> OpenAI-style video status. Non-terminal codes keep the
# client polling; the app treats completed as done and failed as a terminal error.
_LIBTV_STATUS = {0: "queued", 1: "in_progress", 2: "completed", 3: "failed"}


def _decode_task_id(video_id: str) -> str:
    task_id = (decode_video_id_with_provider(video_id) or {}).get("video_id") or ""
    if not task_id:
        raise LibTVError(status_code=400, message="libtv video id does not carry a task id")
    return task_id


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


def _wants_frames2video(optional_params: dict, spec: dict) -> bool:
    # A first/last-frame request (image + last_image) must not be flattened into
    # mixed2video reference-image soup: that loses the first/last ordering entirely.
    # Only take the frames2video path when the model schema actually advertises support
    # for it, so unsupported models keep getting the existing mixed2video behavior
    # instead of a 400.
    if not optional_params.get("last_image"):
        return False
    # An explicit modeType override always wins; only frames2video (or no override)
    # takes this branch, so the mode and the payload shape never disagree.
    if _resolve_mode(optional_params, "frames2video") != "frames2video":
        return False
    mode_items = ((spec.get("properties") or {}).get("modeType") or {}).get("items")
    return isinstance(mode_items, dict) and "frames2video" in mode_items


def _frame_payloads(optional_params: dict) -> list:
    # [first, last] in order; image may be absent (libtv frames2video accepts 1-2
    # frames, and the non-compliance branch already sends a single-image imageList).
    payloads = [_reference_payload(optional_params.get("image")), _reference_payload(optional_params.get("last_image"))]
    return [p for p in payloads if p is not None]


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

    def _build_video_object(self, model: str, created: dict, optional_params: Optional[dict] = None) -> VideoObject:
        op = optional_params or {}
        # Encode the libtv task id + this deployment's status model into the video
        # id so the proxy routes subsequent /v1/videos/{id} status and /content
        # calls back to the SAME libtv account (its token+webid).
        video_id = encode_video_id_with_provider(created["task_id"], LIBTV_PROVIDER, op.get("libtv_status_model"))
        vo = VideoObject(id=video_id, object="video", status="queued", model=model)
        vo.usage = _video_usage(op)
        vo._hidden_params = {"project_uuid": created.get("project_uuid")}
        return vo

    def _video_status(self, video_id: str, state: dict) -> VideoObject:
        status = _LIBTV_STATUS.get(state.get("status"), "in_progress")
        vo = VideoObject(id=video_id, object="video", status=status)
        if status == "completed":
            urls = state.get("urls") or []
            vo._hidden_params = {"url": urls[0] if urls else None, "libtv_video_urls": urls}
        elif status == "failed":
            vo.error = {"message": state.get("failed_reason") or "libtv generation failed"}
        return vo

    def _download(self, http, state: dict) -> bytes:
        if state.get("status") != 2:
            raise LibTVError(status_code=409, message="libtv video still processing")
        urls = state.get("urls") or []
        if not urls:
            raise LibTVError(status_code=502, message="libtv video completed without a result url")
        resp = http.get(url=urls[0])
        if resp.status_code != 200:
            raise LibTVError(status_code=resp.status_code, message="libtv video content download failed")
        return resp.content

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
        lt = self._make_client(api_key, optional_params, sync_client=client or HTTPHandler())
        return self._video_status(video_id, lt.poll_once(_decode_task_id(video_id), "video"))

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
        lt = self._make_client(api_key, optional_params, async_client=client or AsyncHTTPHandler())
        return self._video_status(video_id, await lt.apoll_once(_decode_task_id(video_id), "video"))

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
        http = client or HTTPHandler()
        lt = self._make_client(api_key, optional_params, sync_client=http)
        return self._download(http, lt.poll_once(_decode_task_id(video_id), "video"))

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
        http = client or AsyncHTTPHandler()
        lt = self._make_client(api_key, optional_params, async_client=http)
        state = await lt.apoll_once(_decode_task_id(video_id), "video")
        if state.get("status") != 2:
            raise LibTVError(status_code=409, message="libtv video still processing")
        urls = state.get("urls") or []
        if not urls:
            raise LibTVError(status_code=502, message="libtv video completed without a result url")
        resp = await http.get(url=urls[0])
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

        if images and _auto_compliance_enabled(spec) and _wants_frames2video(optional_params, spec):
            frame_refs = lt.resolve_compliant_image_refs(_frame_payloads(optional_params))
            video_refs = lt.resolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "frames2video")
            params["autoCompliance"] = 1
            params["imageList"] = frame_refs
            if video_refs:
                params["videoList"] = video_refs
            if audios:
                params["audioList"] = [url_for(r, _REF_DEFAULT_NAME["audio"]) for r in audios]
        elif images and _auto_compliance_enabled(spec):
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
        created = lt.create(model, spec["vendor"], "video", params, _project_name(model))
        return self._build_video_object(model, created, optional_params)

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

        if images and _auto_compliance_enabled(spec) and _wants_frames2video(optional_params, spec):
            frame_refs = await lt.aresolve_compliant_image_refs(_frame_payloads(optional_params))
            video_refs = await lt.aresolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "frames2video")
            params["autoCompliance"] = 1
            params["imageList"] = frame_refs
            if video_refs:
                params["videoList"] = video_refs
            if audios:
                params["audioList"] = [await url_for(r, _REF_DEFAULT_NAME["audio"]) for r in audios]
        elif images and _auto_compliance_enabled(spec):
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
        created = await lt.acreate(model, spec["vendor"], "video", params, _project_name(model))
        return self._build_video_object(model, created, optional_params)
