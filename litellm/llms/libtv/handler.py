import asyncio
import logging
import os
import random
import time
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Optional, Tuple, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM
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
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

LIBTV_PROVIDER = "libtv"
logger = logging.getLogger(__name__)

# A pooled keep-alive connection can be closed by the peer during the long
# fresh_asset_retry_wait sleep; reusing it fails the very first http call of the
# retried create instantly, before any request reaches the server. One immediate
# re-attempt dials a fresh connection and is enough to recover.
_CONNECT_PHASE_FAILURES: Tuple[type, ...] = (
    Timeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)
# litellm.Timeout above also wraps genuine mid-request read timeouts (AsyncHTTPHandler
# collapses every httpx.TimeoutException into it), and create's last http call submits
# a paid task: re-dialing after a read timeout could double-bill a task the server
# already accepted. Only an attempt that failed near-instantly (a stale connection
# dies in ~1ms; a real read timeout waited out its full multi-second budget) is safe
# to re-dial.
_STALE_CONNECT_WINDOW_SECONDS = 2.0

_REF_DEFAULT_NAME = {"image": "reference.png", "video": "reference.mp4", "audio": "reference.mp3"}

# Keep in sync with the keys _collect_reference_groups reads: the guard below must
# recognize exactly the keys that can contribute a reference, native libtv or
# wavespeed-shaped, so it neither misses a caller's reference intent nor false-alarms
# on a plain text2video request that never mentioned references at all.
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

from litellm.llms.openai.cost_calculation import _video_output_cost_per_second

from .client import LibTVClient
from .common import LibTVContentPolicyError, LibTVError, resolve_libtv_credentials
from .persistence import get_persistence
from .transform import _resolution_from_size, build_generation_params, build_topaz_upscale_params

_TOPAZ_VENDOR = "topazlabs"

# libtv progress status -> OpenAI-style video status. Non-terminal codes keep the
# client polling; the app treats completed as done and failed as a terminal error.
_LIBTV_STATUS = {0: "queued", 1: "in_progress", 2: "completed", 3: "failed"}


def _raise_normalized_libtv_error(error: LibTVError, model: str) -> None:
    """Convert provider errors once, at the shared custom-provider boundary."""
    response = httpx.Response(
        status_code=error.status_code,
        headers=error.headers,
        request=httpx.Request("POST", "https://api.liblib.tv"),
    )
    common = {"message": error.message, "model": model, "llm_provider": LIBTV_PROVIDER}
    if isinstance(error, LibTVContentPolicyError):
        raise ContentPolicyViolationError(**common, response=response) from error
    if error.status_code == 400:
        raise BadRequestError(**common, response=response) from error
    if error.status_code == 401:
        raise AuthenticationError(**common, response=response) from error
    if error.status_code == 403:
        raise PermissionDeniedError(**common, response=response) from error
    if error.status_code in (408, 504):
        raise Timeout(
            **common,
            headers=error.headers,
            exception_status_code=error.status_code,
        ) from error
    if error.status_code == 429:
        raise RateLimitError(**common, response=response) from error
    if error.status_code == 502:
        raise BadGatewayError(**common, response=response) from error
    if error.status_code == 503:
        raise ServiceUnavailableError(**common, response=response) from error
    if error.status_code >= 500:
        raise InternalServerError(**common, response=response) from error
    raise APIError(status_code=error.status_code, **common) from error


def normalize_libtv_errors(func):
    """Decorator shared by image/video sync+async custom-provider methods."""

    def _model(args, kwargs) -> str:
        value = kwargs.get("model") or kwargs.get("video_id")
        if value is None and len(args) > 1:
            value = args[1]
        return str(value or "libtv")

    if iscoroutinefunction(func):

        @wraps(func)
        async def _async(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except LibTVError as error:
                _raise_normalized_libtv_error(error, _model(args, kwargs))

        return _async

    @wraps(func)
    def _sync(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LibTVError as error:
            _raise_normalized_libtv_error(error, _model(args, kwargs))

    return _sync


def _decode_task_id(video_id: str) -> str:
    task_id = (decode_video_id_with_provider(video_id) or {}).get("video_id") or ""
    if not task_id:
        raise LibTVError(status_code=400, message="libtv video id does not carry a task id")
    return task_id


_PROJECT_NAME_POOL = ("我的项目", "未命名项目", "新建项目", "创意工坊", "日常创作")


def _project_name(model: str) -> str:
    override = os.getenv("LIBTV_PROJECT_NAME")
    if override:
        return override
    return random.choice(_PROJECT_NAME_POOL)


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


def _video_completion_billing(optional_params: dict) -> Tuple[Optional[dict], Optional[float]]:
    """usage + response cost for a completed libtv video task.

    libtv's progress poll never returns the actual generated duration/resolution,
    so this reuses the values the request itself asked for (same source
    _video_usage draws on for create) as the authoritative billing basis. Returns
    (None, None) when no duration can be resolved at all, or (usage, None) when a
    duration is known but the deployment declares no per-second price for it.
    """
    usage = _video_usage(optional_params)
    if usage is None:
        return None, None
    model_info = optional_params.get("model_info") or {}
    rate = _video_output_cost_per_second(model_info, usage.get("video_resolution"))
    if rate is None:
        return usage, None
    return usage, rate * usage["duration_seconds"]


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


def _guard_reference_intent(model: str, optional_params: dict, images: list, videos: list, audios: list) -> None:
    # A caller that sends any reference key is declaring reference intent. If every
    # key we recognize resolves to nothing, silently falling through to the
    # text2video branch would burn provider quota generating the wrong content
    # (production incident: reference_images present but collected empty -> silent
    # text2video). Fail loud instead of degrading.
    if images or videos or audios:
        return
    present_keys = [key for key in _REFERENCE_KEYS if key in optional_params]
    if not present_keys:
        return
    raise LibTVError(
        status_code=400,
        message=(
            f"libtv video_generation model={model}: reference keys present ({', '.join(present_keys)}) "
            "but resolved to no references; refusing to degrade to text2video"
        ),
    )


def _log_reference_collection(
    model: str, optional_params: dict, images: list, videos: list, audios: list, branch: str
) -> None:
    key_presence = " ".join(f"{key}_present={key in optional_params}" for key in _REFERENCE_KEYS)
    logger.info(
        "libtv reference_collection model=%s %s images=%d videos=%d audios=%d branch=%s",
        model,
        key_presence,
        len(images),
        len(videos),
        len(audios),
        branch,
    )


def _is_topaz_upscale(spec: dict) -> bool:
    # topaz-video-upscaler has a single generateType and no modeType in its vendor
    # schema; the mode-inference/modeType-injection machinery below must not touch it.
    return spec.get("vendor") == _TOPAZ_VENDOR


def _topaz_source_videos(optional_params: dict) -> list:
    return _as_list(optional_params.get("video_references")) + _as_list(optional_params.get("input_reference"))


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
    # contain a real person; the upstream verify flow must run and register each image
    # as a compliant asset (real cdn url + assetId) before generation. The model schema
    # advertises this.
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


def _image2video_eligible(optional_params: dict, spec: dict, images: list, videos: list, audios: list) -> bool:
    if videos or audios or optional_params.get("last_image"):
        return False
    mode_items = ((spec.get("properties") or {}).get("modeType") or {}).get("items")
    if not isinstance(mode_items, dict):
        return False
    bounds = mode_items.get("image2video")
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        return False
    lo, hi = bounds
    if not (isinstance(lo, int) and isinstance(hi, int)):
        return False
    cfg_settings = (spec.get("config") or {}).get("settings")
    if isinstance(cfg_settings, dict) and "image2video" not in cfg_settings:
        return False
    return lo <= len(images) <= hi


def _asset_ref_to_string(ref: dict) -> str:
    # frames2video's imageList/videoList vendor contract wants "asset://<id>" strings,
    # not the {url, assetId} objects mixed2video's mixedList/imageList expect: verified
    # in production, an object-shaped imageList fails generation outright on
    # star-video2-fast frames2video. assetId is None for exempt refs (no real person
    # detected), which keeps the pre-compliance behavior of sending the raw cdn url.
    asset_id = ref.get("assetId")
    return f"asset://{asset_id}" if asset_id else ref["url"]


_FRESH_ASSET_RETRY_TOKEN = "请稍后重试"


def _is_fresh_asset_aging_failure(state: dict) -> bool:
    """Whether a frames2video task failed with the vendor's generic retryable
    failure. Live-verified (2026-07-11): seedance/star-video2 frames2video with a
    freshly registered real-person compliant asset ALWAYS fails upstream within
    seconds with this generic reason, while the exact same asset ids succeed once
    the registration is a few minutes old (server runs an internal deep audit with
    no observable readiness signal in third_asset/check; assetId and status=1 are
    returned immediately and never change). Genuine compliance rejections use a
    different message ("请先进行合规校验后重试") and must not be retried."""
    if state.get("status") != 3:
        return False
    reason = state.get("failed_reason") or ""
    return _FRESH_ASSET_RETRY_TOKEN in reason and "合规" not in reason


def _frame_payloads(optional_params: dict) -> list:
    # [first, last] in order; image may be absent (libtv frames2video accepts 1-2
    # frames, and the non-compliance branch already sends a single-image imageList).
    payloads = [_reference_payload(optional_params.get("image")), _reference_payload(optional_params.get("last_image"))]
    return [p for p in payloads if p is not None]


class LibTVLLM(CustomLLM):
    def __init__(
        self,
        poll_interval: float = 3.0,
        poll_max_attempts: int = 200,
        http_get=None,
        http_put=None,
        fresh_asset_retry_attempts: int = 5,
        fresh_asset_retry_wait: float = 60.0,
        fresh_asset_guard_polls: int = 10,
    ):
        super().__init__()
        self.poll_interval = poll_interval
        self.poll_max_attempts = poll_max_attempts
        self._http_get = http_get
        self._http_put = http_put
        self.fresh_asset_retry_attempts = fresh_asset_retry_attempts
        self.fresh_asset_retry_wait = fresh_asset_retry_wait
        self.fresh_asset_guard_polls = fresh_asset_guard_polls

    def _make_client(
        self,
        api_key: Optional[str],
        optional_params: dict,
        sync_client: Optional[HTTPHandler] = None,
        async_client: Optional[AsyncHTTPHandler] = None,
    ) -> LibTVClient:
        require_explicit = optional_params.get("libtv_require_explicit_credentials") is True
        token = api_key if api_key is not None or not require_explicit else ""
        webid = optional_params.get("webid")
        if require_explicit and webid is None:
            webid = ""
        token, webid = resolve_libtv_credentials(token=token, webid=webid)
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
        model_info = op.get("model_info") or {}
        deployment_id = model_info.get("id") if isinstance(model_info, dict) else None
        video_id = encode_video_id_with_provider(
            created["task_id"],
            LIBTV_PROVIDER,
            deployment_id or op.get("libtv_status_model"),
        )
        vo = VideoObject(id=video_id, object="video", status="queued", model=model)
        vo.usage = _video_usage(op)
        vo._hidden_params = {"project_uuid": created.get("project_uuid")}
        return vo

    def _create_with_fresh_asset_retry(
        self, lt: LibTVClient, model: str, vendor: str, params: dict, project_name: str
    ) -> dict:
        """Create a frames2video or image2video task, guard-poll it briefly, and re-create on the
        fresh-asset aging failure (see _is_fresh_asset_aging_failure). Registered
        asset ids stay valid across attempts (the server dedupes by url), so each
        retry reuses the same params; a retry a few minutes later lands after the
        server's internal audit and succeeds. Total worst-case wall time stays
        under the caller's submit read timeout (drama uses 600s)."""
        created = lt.create(model, vendor, "video", params, project_name)
        for _ in range(self.fresh_asset_retry_attempts):
            state = self._guard_poll_sync(lt, created["task_id"])
            if state is None or not _is_fresh_asset_aging_failure(state):
                return created
            time.sleep(self.fresh_asset_retry_wait)
            created = self._create_after_wait(lt, model, vendor, params, project_name)
        return created

    def _create_after_wait(
        self, lt: LibTVClient, model: str, vendor: str, params: dict, project_name: str, monotonic=time.monotonic
    ) -> dict:
        start = monotonic()
        try:
            return lt.create(model, vendor, "video", params, project_name)
        except _CONNECT_PHASE_FAILURES:
            if monotonic() - start >= _STALE_CONNECT_WINDOW_SECONDS:
                raise
            return lt.create(model, vendor, "video", params, project_name)

    async def _acreate_with_fresh_asset_retry(
        self, lt: LibTVClient, model: str, vendor: str, params: dict, project_name: str
    ) -> dict:
        created = await lt.acreate(model, vendor, "video", params, project_name)
        for _ in range(self.fresh_asset_retry_attempts):
            state = await self._guard_poll_async(lt, created["task_id"])
            if state is None or not _is_fresh_asset_aging_failure(state):
                return created
            await asyncio.sleep(self.fresh_asset_retry_wait)
            created = await self._acreate_after_wait(lt, model, vendor, params, project_name)
        return created

    async def _acreate_after_wait(
        self, lt: LibTVClient, model: str, vendor: str, params: dict, project_name: str, monotonic=time.monotonic
    ) -> dict:
        start = monotonic()
        try:
            return await lt.acreate(model, vendor, "video", params, project_name)
        except _CONNECT_PHASE_FAILURES:
            if monotonic() - start >= _STALE_CONNECT_WINDOW_SECONDS:
                raise
            return await lt.acreate(model, vendor, "video", params, project_name)

    def _guard_poll_sync(self, lt: LibTVClient, task_id: str) -> Optional[dict]:
        for _ in range(self.fresh_asset_guard_polls):
            state = lt.poll_once(task_id, "video")
            if state.get("status") in (2, 3):
                return state
            time.sleep(self.poll_interval)
        return None

    async def _guard_poll_async(self, lt: LibTVClient, task_id: str) -> Optional[dict]:
        for _ in range(self.fresh_asset_guard_polls):
            state = await lt.apoll_once(task_id, "video")
            if state.get("status") in (2, 3):
                return state
            await asyncio.sleep(self.poll_interval)
        return None

    def _video_status(self, video_id: str, state: dict) -> VideoObject:
        status = _LIBTV_STATUS.get(state.get("status"), "in_progress")
        vo = VideoObject(id=video_id, object="video", status=status)
        if status == "completed":
            urls = state.get("urls") or []
            vo._hidden_params = {"url": urls[0] if urls else None, "libtv_video_urls": urls}
        elif status == "failed":
            vo.error = {"message": state.get("failed_reason") or "libtv generation failed"}
        return vo

    async def _bill_completed_video(self, vo: VideoObject, task_id: str, optional_params: dict) -> None:
        """Charge for a completed libtv video task exactly once.

        The libtv progress poll is the only point in the async create/poll/download
        flow that knows generation actually finished, so it is also the only correct
        place to accrue spend (create-time cost is always 0.0: no duration is known
        yet). Poll-to-completed fires repeatedly (client retries, repeated status
        checks), so charging here must be idempotent: a persistence-backed
        insert-once marker gates the charge, and any failure to reach that marker
        (no db configured, db error) skips the charge rather than risking a double
        bill.
        """
        usage, cost = _video_completion_billing(optional_params)
        if usage is not None:
            vo.usage = usage
        if cost is None:
            return
        persistence = get_persistence()
        if persistence is None:
            return
        try:
            billed = await persistence.mark_video_billed(f"{LIBTV_PROVIDER}:{task_id}", usage["duration_seconds"], cost)
        except Exception:
            logger.warning("libtv video billing: persistence check failed, skipping charge", exc_info=True)
            return
        vo._hidden_params = {**vo._hidden_params, "response_cost": cost if billed else 0.0}

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

    @normalize_libtv_errors
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

    @normalize_libtv_errors
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
        task_id = _decode_task_id(video_id)
        vo = self._video_status(video_id, await lt.apoll_once(task_id, "video"))
        if vo.status == "completed":
            await self._bill_completed_video(vo, task_id, optional_params)
        return vo

    @normalize_libtv_errors
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

    @normalize_libtv_errors
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
                params["imageList"] = [
                    {**ref, "mediaType": "image"}
                    for ref in lt.resolve_compliant_image_refs([_reference_payload(r) for r in images])
                ]
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
                params["imageList"] = [
                    {**ref, "mediaType": "image"}
                    for ref in await lt.aresolve_compliant_image_refs([_reference_payload(r) for r in images])
                ]
            else:
                params["imageList"] = [
                    await lt.aensure_libtv_url(*_reference_payload(r), _REF_DEFAULT_NAME["image"]) for r in images
                ]
        return params

    @normalize_libtv_errors
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

    @normalize_libtv_errors
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

    @normalize_libtv_errors
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

    @normalize_libtv_errors
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

    @normalize_libtv_errors
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
        if _is_topaz_upscale(spec):
            source_videos = _topaz_source_videos(optional_params)
            if not source_videos:
                raise LibTVError(
                    status_code=400,
                    message=f"libtv video_generation model={model}: topaz-video-upscaler requires "
                    "video_references or input_reference",
                )
            params = build_topaz_upscale_params(prompt, optional_params)
            params["videoList"] = [
                lt.ensure_libtv_url(*_reference_payload(r), _REF_DEFAULT_NAME["video"]) for r in source_videos
            ]
            created = lt.create(model, spec["vendor"], "video", params, _project_name(model))
            return self._build_video_object(model, created, {**optional_params, "resolution": params["resolution"]})
        images, videos, audios = _collect_reference_groups(optional_params)
        _guard_reference_intent(model, optional_params, images, videos, audios)
        auto_compliance = _auto_compliance_enabled(spec)
        wants_frames = bool(images) and auto_compliance and _wants_frames2video(optional_params, spec)
        image2video_eligible = (
            bool(images)
            and auto_compliance
            and not wants_frames
            and _image2video_eligible(optional_params, spec, images, videos, audios)
        )
        default_mode = "image2video" if image2video_eligible else "mixed2video"
        wants_image2video = image2video_eligible and _resolve_mode(optional_params, default_mode) == "image2video"
        _log_reference_collection(
            model,
            optional_params,
            images,
            videos,
            audios,
            branch=(
                "frames2video"
                if wants_frames
                else "image2video"
                if wants_image2video
                else "mixed2video"
                if images and auto_compliance
                else "else-mode"
            ),
        )

        def url_for(ref, default_name):
            p = _reference_payload(ref)
            return lt.ensure_libtv_url(p[0], p[1], p[2], default_name)

        if wants_frames:
            frame_refs = lt.resolve_compliant_image_refs(_frame_payloads(optional_params))
            video_refs = lt.resolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "frames2video")
            params["autoCompliance"] = 1
            params["imageList"] = [_asset_ref_to_string(ref) for ref in frame_refs]
            if video_refs:
                params["videoList"] = [_asset_ref_to_string(ref) for ref in video_refs]
            if audios:
                params["audioList"] = [url_for(r, _REF_DEFAULT_NAME["audio"]) for r in audios]
        elif wants_image2video:
            image_refs = lt.resolve_compliant_image_refs([_reference_payload(r) for r in images])
            params = build_generation_params(prompt, optional_params, spec, "image2video")
            params["autoCompliance"] = 1
            params["imageList"] = [_asset_ref_to_string(ref) for ref in image_refs]
        elif images and auto_compliance:
            image_refs = lt.resolve_compliant_image_refs([_reference_payload(r) for r in images])
            video_refs = lt.resolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "mixed2video")
            params["autoCompliance"] = 1
            image_entries = [{**ref, "mediaType": "image"} for ref in image_refs]
            params["mixedList"] = (
                image_entries
                + [{**ref, "mediaType": "video"} for ref in video_refs]
                + [{"url": url_for(r, _REF_DEFAULT_NAME["audio"]), "mediaType": "audio"} for r in audios]
            )
            params["imageList"] = image_entries
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
        created = (
            self._create_with_fresh_asset_retry(lt, model, spec["vendor"], params, _project_name(model))
            if wants_frames or wants_image2video
            else lt.create(model, spec["vendor"], "video", params, _project_name(model))
        )
        return self._build_video_object(model, created, optional_params)

    @normalize_libtv_errors
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
        if _is_topaz_upscale(spec):
            source_videos = _topaz_source_videos(optional_params)
            if not source_videos:
                raise LibTVError(
                    status_code=400,
                    message=f"libtv video_generation model={model}: topaz-video-upscaler requires "
                    "video_references or input_reference",
                )
            params = build_topaz_upscale_params(prompt, optional_params)
            params["videoList"] = [
                await lt.aensure_libtv_url(*_reference_payload(r), _REF_DEFAULT_NAME["video"]) for r in source_videos
            ]
            created = await lt.acreate(model, spec["vendor"], "video", params, _project_name(model))
            return self._build_video_object(model, created, {**optional_params, "resolution": params["resolution"]})
        images, videos, audios = _collect_reference_groups(optional_params)
        _guard_reference_intent(model, optional_params, images, videos, audios)
        auto_compliance = _auto_compliance_enabled(spec)
        wants_frames = bool(images) and auto_compliance and _wants_frames2video(optional_params, spec)
        image2video_eligible = (
            bool(images)
            and auto_compliance
            and not wants_frames
            and _image2video_eligible(optional_params, spec, images, videos, audios)
        )
        default_mode = "image2video" if image2video_eligible else "mixed2video"
        wants_image2video = image2video_eligible and _resolve_mode(optional_params, default_mode) == "image2video"
        _log_reference_collection(
            model,
            optional_params,
            images,
            videos,
            audios,
            branch=(
                "frames2video"
                if wants_frames
                else "image2video"
                if wants_image2video
                else "mixed2video"
                if images and auto_compliance
                else "else-mode"
            ),
        )

        async def url_for(ref, default_name):
            p = _reference_payload(ref)
            return await lt.aensure_libtv_url(p[0], p[1], p[2], default_name)

        if wants_frames:
            frame_refs = await lt.aresolve_compliant_image_refs(_frame_payloads(optional_params))
            video_refs = await lt.aresolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "frames2video")
            params["autoCompliance"] = 1
            params["imageList"] = [_asset_ref_to_string(ref) for ref in frame_refs]
            if video_refs:
                params["videoList"] = [_asset_ref_to_string(ref) for ref in video_refs]
            if audios:
                params["audioList"] = [await url_for(r, _REF_DEFAULT_NAME["audio"]) for r in audios]
        elif wants_image2video:
            image_refs = await lt.aresolve_compliant_image_refs([_reference_payload(r) for r in images])
            params = build_generation_params(prompt, optional_params, spec, "image2video")
            params["autoCompliance"] = 1
            params["imageList"] = [_asset_ref_to_string(ref) for ref in image_refs]
        elif images and auto_compliance:
            image_refs = await lt.aresolve_compliant_image_refs([_reference_payload(r) for r in images])
            video_refs = await lt.aresolve_compliant_video_refs([_reference_payload(r) for r in videos])
            params = build_generation_params(prompt, optional_params, spec, "mixed2video")
            params["autoCompliance"] = 1
            image_entries = [{**ref, "mediaType": "image"} for ref in image_refs]
            params["mixedList"] = (
                image_entries
                + [{**ref, "mediaType": "video"} for ref in video_refs]
                + [{"url": await url_for(r, _REF_DEFAULT_NAME["audio"]), "mediaType": "audio"} for r in audios]
            )
            params["imageList"] = image_entries
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
        created = (
            await self._acreate_with_fresh_asset_retry(lt, model, spec["vendor"], params, _project_name(model))
            if wants_frames or wants_image2video
            else await lt.acreate(model, spec["vendor"], "video", params, _project_name(model))
        )
        return self._build_video_object(model, created, optional_params)
