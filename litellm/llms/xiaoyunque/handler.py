import logging
import os
import time
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.llms.libtv.persistence import get_persistence
from litellm.llms.openai.cost_calculation import _video_output_cost_per_second
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

from .client import (
    XiaoyunqueAsyncSleep,
    XiaoyunqueClient,
    XiaoyunqueHTTPGet,
    XiaoyunqueJitter,
    XiaoyunqueSleep,
    decode_composite_task_id,
    encode_composite_task_id,
)
from .common import XiaoyunqueContentPolicyError, XiaoyunqueError, resolve_xiaoyunque_credentials
from .transform import build_video_part_tool_param, resolution_from_size

FB3_PROVIDER = "fb3"
logger = logging.getLogger(__name__)

_REF_DEFAULT_NAME = {"image": "reference.png", "video": "reference.mp4", "audio": "reference.mp3"}

# Keep in sync with the keys _collect_reference_groups reads: the intent guard below must
# recognize exactly the keys that can contribute a reference, so it neither misses a
# caller's reference intent nor false-alarms on a plain text2video request.
_REFERENCE_KEYS = (
    "input_reference",
    "image_references",
    "reference_images",
    "image",
    "last_image",
    "video_references",
    "reference_videos",
    "audio_references",
    "reference_audios",
)

_XIAOYUNQUE_STATUS = {1: "queued", 2: "in_progress", 3: "completed", 4: "failed", 5: "failed"}

_BILLING_WARN_INTERVAL_SECONDS = 300.0
_last_billing_warn: Dict[str, Optional[float]] = {}  # mutable-ok: per-key last-emitted-time cache for rate limiting


def _warn_billing_gap(key: str, message: str) -> None:
    now = time.monotonic()
    last = _last_billing_warn.get(key)
    if last is None or now - last >= _BILLING_WARN_INTERVAL_SECONDS:
        _last_billing_warn[key] = now
        logger.warning(message)


def _raise_normalized_xiaoyunque_error(error: XiaoyunqueError, model: str) -> None:
    response = httpx.Response(
        status_code=error.status_code,
        headers=error.headers,
        request=httpx.Request("POST", "https://fb3.internal"),
    )
    common = {"message": error.message, "model": model, "llm_provider": FB3_PROVIDER}
    if isinstance(error, XiaoyunqueContentPolicyError):
        raise ContentPolicyViolationError(**common, response=response) from error
    if error.status_code == 400:
        raise BadRequestError(**common, response=response) from error
    if error.status_code == 401:
        raise AuthenticationError(**common, response=response) from error
    if error.status_code == 403:
        raise PermissionDeniedError(**common, response=response) from error
    if error.status_code in (408, 504):
        raise Timeout(**common, headers=error.headers, exception_status_code=error.status_code) from error
    if error.status_code == 429:
        raise RateLimitError(**common, response=response) from error
    if error.status_code == 502:
        raise BadGatewayError(**common, response=response) from error
    if error.status_code == 503:
        raise ServiceUnavailableError(**common, response=response) from error
    if error.status_code >= 500:
        raise InternalServerError(**common, response=response) from error
    raise APIError(status_code=error.status_code, **common) from error


def normalize_xiaoyunque_errors(func):
    """Decorator shared by video sync+async custom-provider methods."""

    def _model(args, kwargs) -> str:
        value = kwargs.get("model") or kwargs.get("video_id")
        if value is None and len(args) > 1:
            value = args[1]
        return str(value or "fb3")

    if iscoroutinefunction(func):

        @wraps(func)
        async def _async(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except XiaoyunqueError as error:
                _raise_normalized_xiaoyunque_error(error, _model(args, kwargs))

        return _async

    @wraps(func)
    def _sync(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except XiaoyunqueError as error:
            _raise_normalized_xiaoyunque_error(error, _model(args, kwargs))

    return _sync


def _decode_task_id(video_id: str) -> str:
    task_id = (decode_video_id_with_provider(video_id) or {}).get("video_id") or ""
    if not task_id:
        raise XiaoyunqueError(status_code=400, message="fb3 video id does not carry a task id")
    return task_id


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and not (
        len(value) >= 2 and isinstance(value[0], str) and not isinstance(value[1], str)
    ):
        return list(value)
    return [value]


def _reference_payload(ref: Any) -> Optional[Tuple[str, str, Optional[bytes]]]:
    """Normalize a litellm video reference into ('url', url, None) or ('bytes', filename, data)."""
    if ref is None:
        return None
    if isinstance(ref, str):
        if ref.startswith(("http://", "https://")):
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
    raise XiaoyunqueError(status_code=400, message=f"unsupported video reference type: {type(ref).__name__}")


def _reference_payloads(refs: list) -> List[Tuple[str, str, Optional[bytes]]]:
    # _reference_payload returns None for a bare `None` list entry; unpacking that
    # directly into ensure_uploaded(*payload, ...) would raise a bare TypeError at
    # runtime instead of a clean, filterable result, so drop it here.
    return [p for p in (_reference_payload(r) for r in refs) if p is not None]


def _wants_frames2video(optional_params: dict) -> bool:
    # "mixed2video" and any other modeType the drama backend sends for other providers
    # has no xiaoyunque equivalent and must be silently ignored; only an explicit
    # frames2video signal (or the image+last_image pair it always accompanies on models
    # that require it) switches on generate_type=1.
    if optional_params.get("modeType") == "frames2video":
        return True
    return bool(optional_params.get("image")) and bool(optional_params.get("last_image"))


def _collect_reference_groups(optional_params: dict) -> Tuple[list, list, list, bool]:
    wants_frames = _wants_frames2video(optional_params)
    images = (
        _as_list(optional_params.get("input_reference"))
        + _as_list(optional_params.get("image_references"))
        + _as_list(optional_params.get("reference_images"))
    )
    if wants_frames:
        images = [p for p in (optional_params.get("image"), optional_params.get("last_image")) if p is not None]
    else:
        images = images + _as_list(optional_params.get("image"))
    videos = _as_list(optional_params.get("video_references")) + _as_list(optional_params.get("reference_videos"))
    audios = _as_list(optional_params.get("audio_references")) + _as_list(optional_params.get("reference_audios"))
    return images, videos, audios, wants_frames


def _guard_reference_intent(model: str, optional_params: dict, images: list, videos: list, audios: list) -> None:
    # A caller that sends any reference key is declaring reference intent. If every key
    # we recognize resolves to nothing, silently falling through to text2video would burn
    # provider quota generating the wrong content (libtv production incident, same root
    # cause). Fail loud instead of degrading.
    if images or videos or audios:
        return
    present_keys = [key for key in _REFERENCE_KEYS if key in optional_params]
    if not present_keys:
        return
    raise XiaoyunqueError(
        status_code=400,
        message=(
            f"fb3 video_generation model={model}: reference keys present ({', '.join(present_keys)}) "
            "but resolved to no references; refusing to degrade to text2video"
        ),
    )


def _guard_audio_requires_visual(model: str, images: list, videos: list, audios: list) -> None:
    if audios and not images and not videos:
        raise XiaoyunqueError(
            status_code=400,
            message=f"fb3 video_generation model={model}: audio references require image or video references "
            "(upstream rejects audio-only submissions)",
        )


def _video_usage(optional_params: dict) -> Optional[dict]:
    try:
        duration_seconds = float(optional_params.get("seconds") or optional_params.get("duration"))
    except (TypeError, ValueError):
        return None
    usage: dict = {"duration_seconds": duration_seconds}
    resolution = optional_params.get("resolution") or resolution_from_size(optional_params.get("size"))
    if resolution:
        usage["video_resolution"] = resolution
    return usage


def _video_billing_key(task_id: str) -> str:
    return f"xiaoyunque:{task_id}"


def _video_completion_cost(optional_params: dict, usage: dict) -> Optional[float]:
    model_info = optional_params.get("model_info") or {}
    rate = _video_output_cost_per_second(model_info, usage.get("video_resolution"))
    if rate is None:
        return None
    return rate * usage["duration_seconds"]


class XiaoyunqueLLM(CustomLLM):
    def __init__(
        self,
        http_get: XiaoyunqueHTTPGet | None = None,
        sleep: XiaoyunqueSleep | None = None,
        asleep: XiaoyunqueAsyncSleep | None = None,
        jitter: XiaoyunqueJitter | None = None,
    ):
        super().__init__()
        self._http_get = http_get
        self._sleep = sleep
        self._asleep = asleep
        self._jitter = jitter

    def _make_client(
        self,
        api_key: Optional[str],
        optional_params: dict,
        sync_client: Optional[HTTPHandler] = None,
        async_client: Optional[AsyncHTTPHandler] = None,
    ) -> XiaoyunqueClient:
        require_explicit = optional_params.get("xiaoyunque_require_explicit_credentials") is True
        token = api_key if api_key is not None or not require_explicit else ""
        token = resolve_xiaoyunque_credentials(token=token)
        return XiaoyunqueClient(
            token=token,
            sync_client=sync_client,
            async_client=async_client,
            http_get=self._http_get,
            sleep=self._sleep,
            asleep=self._asleep,
            jitter=self._jitter,
        )

    def _build_video_object(
        self, model: str, thread_id: str, run_id: str, optional_params: Optional[dict] = None
    ) -> VideoObject:
        op = optional_params or {}
        task_id = encode_composite_task_id(thread_id, run_id)
        model_info = op.get("model_info") or {}
        deployment_id = model_info.get("id") if isinstance(model_info, dict) else None
        video_id = encode_video_id_with_provider(
            task_id, FB3_PROVIDER, deployment_id or op.get("xiaoyunque_status_model")
        )
        vo = VideoObject(id=video_id, object="video", status="queued", model=model)
        vo.usage = _video_usage(op)
        return vo

    def _video_status(self, video_id: str, state: dict) -> VideoObject:
        status = _XIAOYUNQUE_STATUS.get(state.get("run_state"), "in_progress")
        vo = VideoObject(id=video_id, object="video", status=status)
        if status == "completed":
            urls = state.get("video_urls") or state.get("image_urls") or []
            vo._hidden_params = {"url": urls[0] if urls else None, "xiaoyunque_video_urls": urls}
        elif status == "failed":
            fail_reason = state.get("fail_reason")
            message = fail_reason.get("message") if isinstance(fail_reason, dict) else None
            vo.error = {"message": message or "fb3 generation failed"}
        return vo

    async def _record_video_task_usage(self, task_id: str, optional_params: dict, params: dict) -> None:
        merged = dict(optional_params)
        if merged.get("seconds") is None and merged.get("duration") is None:
            merged["duration"] = params.get("duration_sec")
        if merged.get("resolution") is None and merged.get("size") is None:
            merged["resolution"] = params.get("resolution")
        usage = _video_usage(merged)
        if usage is None:
            logger.warning(
                "xiaoyunque video billing: create for task %s resolved no duration; task will not be billed", task_id
            )
            return
        persistence = get_persistence()
        if persistence is None:
            return
        try:
            await persistence.store_video_task_usage(
                _video_billing_key(task_id), usage["duration_seconds"], usage.get("video_resolution")
            )
        except Exception:  # noqa: BLE001  # best-effort; a failed write just means the later poll can't bill
            logger.warning("xiaoyunque video billing: failed to record task usage at create", exc_info=True)

    async def _bill_completed_video(self, vo: VideoObject, task_id: str, optional_params: dict) -> None:
        persistence = get_persistence()
        model_info = optional_params.get("model_info") or {}
        deployment_id = model_info.get("id")
        usage = _video_usage(optional_params)
        if usage is None and persistence is not None:
            try:
                usage = await persistence.get_video_task_usage(_video_billing_key(task_id))
            except Exception:  # noqa: BLE001  # any read failure skips the charge rather than guessing
                logger.warning("xiaoyunque video billing: usage lookup failed, skipping charge", exc_info=True)
                return
        if usage is None:
            _warn_billing_gap(
                f"no-usage:{deployment_id}",
                f"xiaoyunque video billing: no usage record for completed task {task_id} "
                f"(model_info id={deployment_id!r}); skipping charge",
            )
            return
        vo.usage = usage
        cost = _video_completion_cost(optional_params, usage)
        if cost is None:
            _warn_billing_gap(
                f"no-price:{deployment_id}:{usage.get('video_resolution')}",
                f"xiaoyunque video billing: no price for video_resolution={usage.get('video_resolution')!r} "
                f"on completed task {task_id} (model_info id={deployment_id!r}); skipping charge -- add "
                "output_cost_per_second[_<tier>] to this deployment's model_info",
            )
            return
        if persistence is None:
            return
        try:
            billed = await persistence.mark_video_billed(_video_billing_key(task_id), usage["duration_seconds"], cost)
        except Exception:  # noqa: BLE001  # fail-safe: a persistence error must skip the charge, never double-bill
            logger.warning("xiaoyunque video billing: persistence check failed, skipping charge", exc_info=True)
            return
        vo._hidden_params = {**vo._hidden_params, "response_cost": cost if billed else 0.0}

    def _resolve_references_sync(
        self, xq: XiaoyunqueClient, model: str, optional_params: dict
    ) -> Tuple[List[str], List[str], List[str], bool]:
        images, videos, audios, wants_frames = _collect_reference_groups(optional_params)
        _guard_reference_intent(model, optional_params, images, videos, audios)
        _guard_audio_requires_visual(model, images, videos, audios)
        image_ids = [xq.ensure_uploaded(*p, _REF_DEFAULT_NAME["image"]) for p in _reference_payloads(images)]
        video_ids = [xq.ensure_uploaded(*p, _REF_DEFAULT_NAME["video"]) for p in _reference_payloads(videos)]
        audio_ids = [xq.ensure_uploaded(*p, _REF_DEFAULT_NAME["audio"]) for p in _reference_payloads(audios)]
        return image_ids, video_ids, audio_ids, wants_frames

    async def _aresolve_references(
        self, xq: XiaoyunqueClient, model: str, optional_params: dict
    ) -> Tuple[List[str], List[str], List[str], bool]:
        images, videos, audios, wants_frames = _collect_reference_groups(optional_params)
        _guard_reference_intent(model, optional_params, images, videos, audios)
        _guard_audio_requires_visual(model, images, videos, audios)
        image_ids = [await xq.aensure_uploaded(*p, _REF_DEFAULT_NAME["image"]) for p in _reference_payloads(images)]
        video_ids = [await xq.aensure_uploaded(*p, _REF_DEFAULT_NAME["video"]) for p in _reference_payloads(videos)]
        audio_ids = [await xq.aensure_uploaded(*p, _REF_DEFAULT_NAME["audio"]) for p in _reference_payloads(audios)]
        return image_ids, video_ids, audio_ids, wants_frames

    @normalize_xiaoyunque_errors
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
        xq = self._make_client(api_key, optional_params, sync_client=client or HTTPHandler())
        image_ids, video_ids, audio_ids, wants_frames = self._resolve_references_sync(xq, model, optional_params)
        params = build_video_part_tool_param(
            prompt,
            model,
            optional_params,
            image_ids,
            video_ids,
            audio_ids,
            generate_type=1 if wants_frames else None,
            warn=logger.warning,
        )
        created = xq.submit_run(
            message=prompt, asset_ids=image_ids + video_ids + audio_ids, video_part_tool_param=params
        )
        return self._build_video_object(model, created["thread_id"], created["run_id"], optional_params)

    @normalize_xiaoyunque_errors
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
        xq = self._make_client(api_key, optional_params, async_client=client or AsyncHTTPHandler())
        image_ids, video_ids, audio_ids, wants_frames = await self._aresolve_references(xq, model, optional_params)
        params = build_video_part_tool_param(
            prompt,
            model,
            optional_params,
            image_ids,
            video_ids,
            audio_ids,
            generate_type=1 if wants_frames else None,
            warn=logger.warning,
        )
        created = await xq.asubmit_run(
            message=prompt, asset_ids=image_ids + video_ids + audio_ids, video_part_tool_param=params
        )
        vo = self._build_video_object(model, created["thread_id"], created["run_id"], optional_params)
        await self._record_video_task_usage(
            encode_composite_task_id(created["thread_id"], created["run_id"]), optional_params, params
        )
        return vo

    @normalize_xiaoyunque_errors
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
        xq = self._make_client(api_key, optional_params, sync_client=client or HTTPHandler())
        thread_id, run_id = decode_composite_task_id(_decode_task_id(video_id))
        return self._video_status(video_id, xq.query_result(thread_id, run_id))

    @normalize_xiaoyunque_errors
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
        xq = self._make_client(api_key, optional_params, async_client=client or AsyncHTTPHandler())
        thread_id, run_id = decode_composite_task_id(_decode_task_id(video_id))
        task_id = encode_composite_task_id(thread_id, run_id)
        vo = self._video_status(video_id, await xq.aquery_result(thread_id, run_id))
        if vo.status == "completed":
            await self._bill_completed_video(vo, task_id, optional_params)
        return vo

    @normalize_xiaoyunque_errors
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
        xq = self._make_client(api_key, optional_params, sync_client=http)
        thread_id, run_id = decode_composite_task_id(_decode_task_id(video_id))
        state = xq.query_result(thread_id, run_id)
        if state.get("run_state") != 3:
            raise XiaoyunqueError(status_code=409, message="fb3 video still processing")
        urls = state.get("video_urls") or state.get("image_urls") or []
        if not urls:
            raise XiaoyunqueError(status_code=502, message="fb3 video completed without a result url")
        resp = http.get(url=urls[0])
        if resp.status_code != 200:
            raise XiaoyunqueError(status_code=resp.status_code, message="fb3 video content download failed")
        return resp.content

    @normalize_xiaoyunque_errors
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
        xq = self._make_client(api_key, optional_params, async_client=http)
        thread_id, run_id = decode_composite_task_id(_decode_task_id(video_id))
        state = await xq.aquery_result(thread_id, run_id)
        if state.get("run_state") != 3:
            raise XiaoyunqueError(status_code=409, message="fb3 video still processing")
        urls = state.get("video_urls") or state.get("image_urls") or []
        if not urls:
            raise XiaoyunqueError(status_code=502, message="fb3 video completed without a result url")
        resp = await http.get(url=urls[0])
        if resp.status_code != 200:
            raise XiaoyunqueError(status_code=resp.status_code, message="fb3 video content download failed")
        return resp.content
