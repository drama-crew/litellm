from __future__ import annotations

import logging
import math
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.libtv.persistence import get_persistence
from litellm.llms.libtv.transfer import get_transfer_redis
from litellm.llms.libtv.video_generate import (
    VideoGenerateError,
    VideoGenerateSettings,
    enqueue_video_generate,  # pyright: ignore[reportUnknownVariableType]  # legacy engine has untyped Redis ports
    fetch_video_generate_status,  # pyright: ignore[reportUnknownVariableType]  # legacy engine returns an untyped dict
    validate_video_generate_url,
)
from litellm.types.utils import all_litellm_params
from litellm.types.videos.main import VideoObject

CAUSYN_MODEL = "causyn-1.0"
CAUSYN_VIDEO_ID_PREFIX = "causyn_"
CAUSYN_RESOLUTION = "768x512"
CAUSYN_RATIO = "3:2"
CAUSYN_DEADLINE_SECONDS = 1800.0
_INTERNAL_VIDEO_FLAG = "DRAMA_INTERNAL_VIDEO_ENABLED"
_INTERNAL_VIDEO_FLAG_ALIAS = "OH_DRAMA_INTERNAL_VIDEO_ENABLED"
_INTERNAL_VIDEO_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_TASK_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CAUSYN_USER_PARAMS = frozenset(
    {
        "seconds",
        "resolution",
        "size",
        "aspect_ratio",
        "reference_images",
        "generate_audio",
        "seed",
        # LiteLLM exposes this standard request field, but it is not sent to the
        # worker request payload.
        "user",
    }
)
_CAUSYN_FRAMEWORK_PARAMS = frozenset(all_litellm_params)
logger = logging.getLogger(__name__)


class ContentResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes: ...


RequestTimeout = float | httpx.Timeout | None
ContentGet = Callable[[str, RequestTimeout, bool], Awaitable[ContentResponse]]


class VideoTaskPersistence(Protocol):
    async def store_video_task_usage(
        self,
        billing_key: str,
        duration_seconds: float,
        video_resolution: str | None,
    ) -> None: ...

    async def get_video_task_usage(self, billing_key: str) -> dict[str, object] | None: ...

    async def mark_video_billed(
        self,
        billing_key: str,
        duration_seconds: float,
        response_cost: float,
    ) -> bool: ...


PersistenceFactory = Callable[[], VideoTaskPersistence | None]


async def _default_content_get(
    url: str,
    timeout: RequestTimeout,
    follow_redirects: bool,
) -> ContentResponse:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects) as client:
        return await client.get(url)


def _new_task_id() -> str:
    return uuid.uuid4().hex


def _default_redis_factory() -> object:
    redis: object = get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL"))
    if redis is None:
        raise VideoGenerateError("misconfigured", "no redis URL configured for video-generate")
    return redis


def _default_persistence_factory() -> VideoTaskPersistence | None:
    return get_persistence()


def _enabled() -> bool:
    # Keep this precedence byte-for-byte aligned with the platform admission
    # gate.  Presence matters: an explicitly empty canonical value must not
    # silently fall through to a stale OH_ alias in this separate process.
    if _INTERNAL_VIDEO_FLAG in os.environ:
        raw = os.environ[_INTERNAL_VIDEO_FLAG]
    elif _INTERNAL_VIDEO_FLAG_ALIAS in os.environ:
        raw = os.environ[_INTERNAL_VIDEO_FLAG_ALIAS]
    else:
        return False
    return raw.strip().lower() in _INTERNAL_VIDEO_TRUE_VALUES


def _bad_request(message: str) -> CustomLLMError:
    return CustomLLMError(status_code=400, message=message)


def _service_error(message: str = "causyn video service unavailable") -> CustomLLMError:
    return CustomLLMError(status_code=503, message=message)


def _decode_task_id(video_id: str) -> str:
    if not video_id.startswith(CAUSYN_VIDEO_ID_PREFIX):
        raise _bad_request("invalid causyn video id")
    task_id = video_id[len(CAUSYN_VIDEO_ID_PREFIX) :]
    if _TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise _bad_request("invalid causyn video id")
    return task_id


def _duration(optional_params: dict[str, object]) -> int:
    raw = optional_params.get("seconds")
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdigit():
        raise _bad_request("seconds must be an integer string from 3 through 8")
    value = int(raw)
    if value < 3 or value > 8:
        raise _bad_request("seconds must be from 3 through 8")
    return value


def _references(optional_params: dict[str, object]) -> tuple[dict[str, str], ...]:
    raw = optional_params.get("reference_images")
    try:
        urls = TypeAdapter(list[str]).validate_python(raw, strict=True)
    except ValidationError:
        raise _bad_request("reference_images must contain from 1 through 4 URLs") from None
    if not 1 <= len(urls) <= 4:
        raise _bad_request("reference_images must contain from 1 through 4 URLs")
    if any(not url for url in urls):
        raise _bad_request("each reference image must be a non-empty URL string")
    return tuple({"role": "reference", "media_type": "image", "url": url} for url in urls)


def _resolution(optional_params: dict[str, object]) -> str:
    resolution = optional_params.get("resolution")
    size = optional_params.get("size")
    for name, value in (("resolution", resolution), ("size", size)):
        if value is not None and not isinstance(value, str):
            raise _bad_request(f"{name} must be a string")
    if resolution is not None and size is not None and resolution != size:
        raise _bad_request("resolution and size must match")
    canonical = resolution if resolution is not None else size
    if canonical != CAUSYN_RESOLUTION:
        raise _bad_request(f"resolution must be {CAUSYN_RESOLUTION}")
    return CAUSYN_RESOLUTION


def _request(model: str, prompt: object, optional_params: dict[str, object]) -> tuple[dict[str, object], int]:
    if model != CAUSYN_MODEL:
        raise _bad_request("unsupported causyn model")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _bad_request("prompt must be a non-empty string")
    unsupported = sorted(
        key
        for key, value in optional_params.items()
        if key not in _CAUSYN_USER_PARAMS and key not in _CAUSYN_FRAMEWORK_PARAMS and value is not None
    )
    if unsupported:
        raise _bad_request(f"unsupported causyn video parameter: {', '.join(unsupported)}")
    resolution = _resolution(optional_params)
    ratio = optional_params.get("aspect_ratio")
    if ratio is not None and ratio != CAUSYN_RATIO:
        raise _bad_request(f"aspect_ratio must be {CAUSYN_RATIO}")
    generate_audio = optional_params.get("generate_audio")
    if generate_audio is not None and not isinstance(generate_audio, bool):
        raise _bad_request("generate_audio must be a boolean")
    for unsupported in ("input_reference", "reference_audios", "reference_videos", "image", "last_image"):
        if optional_params.get(unsupported) is not None:
            raise _bad_request(f"{unsupported} is not supported")
    seed = optional_params.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise _bad_request("seed must be an integer")
    duration = _duration(optional_params)
    request: dict[str, object] = {
        "prompt": prompt,
        "duration_seconds": duration,
        "resolution": resolution,
        "ratio": CAUSYN_RATIO,
        # Materialize the platform default in the worker contract so the
        # native-audio choice cannot disappear between /v1/videos and Redis.
        "generate_audio": True if generate_audio is None else generate_audio,
        "references": list(_references(optional_params)),
    }
    if seed is not None:
        request["seed"] = seed
    return request, duration


class _WorkerError(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    code: str | None = None
    kind: str | None = None


class _WorkerResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    validation_version: Literal["video-v1"]
    staging_key: str
    staging_url: str | None = None
    etag: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    content_type: Literal["video/mp4"]
    duration_seconds: float = Field(ge=3, le=8)
    width: Literal[768]
    height: Literal[512]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _StatusEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    status: Literal["queued", "claimed", "succeeded", "failed"] | None = None
    error: _WorkerError | None = None
    result: _WorkerResult | None = None


class _StoredUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    duration_seconds: float = Field(ge=3, le=8)
    video_resolution: Literal["768x512"]


class _Pricing(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    output_cost_per_second_768x512: float | None = Field(default=None, gt=0)
    output_cost_per_second: float | None = Field(default=None, gt=0)


def _public_worker_error(error: _WorkerError | None) -> dict[str, str]:
    raw_code = error.code if error is not None else None
    code = raw_code if isinstance(raw_code, str) and _ERROR_CODE_PATTERN.fullmatch(raw_code) else "worker_failed"
    raw_kind = error.kind if error is not None else None
    kind = raw_kind if raw_kind in {"permanent", "transient", "unknown"} else "unknown"
    return {"code": code, "message": "video generation failed", "kind": kind}


def _usage(duration: float) -> dict[str, float | str]:
    return {"duration_seconds": duration, "video_resolution": CAUSYN_RESOLUTION}


def _billing_key(task_id: str) -> str:
    return f"causyn:{task_id}"


def _staging_key(task_id: str) -> str:
    return f"staging/video-tasks/{task_id}.mp4"


def _completion_cost(optional_params: dict[str, object], duration_seconds: float) -> float | None:
    raw_model_info = optional_params.get("model_info")
    try:
        pricing = _Pricing.model_validate(raw_model_info)
    except ValidationError:
        return None
    rate = pricing.output_cost_per_second_768x512 or pricing.output_cost_per_second
    if rate is None or not math.isfinite(rate) or rate <= 0:
        return None
    return rate * duration_seconds


def _set_response_cost(response: VideoObject, response_cost: float) -> None:
    response._hidden_params = {"response_cost": response_cost}  # pyright: ignore[reportPrivateUsage]  # LiteLLM cost API


class CausynVideoHandler(CustomLLM):
    def __init__(
        self,
        *,
        redis_factory: Callable[[], object] = _default_redis_factory,
        settings_factory: Callable[[], VideoGenerateSettings] = VideoGenerateSettings.from_environment,
        content_get: ContentGet = _default_content_get,
        task_id_factory: Callable[[], str] = _new_task_id,
        clock: Callable[[], float] = time.time,
        persistence_factory: PersistenceFactory = _default_persistence_factory,
    ) -> None:
        super().__init__()
        self._redis_factory = redis_factory
        self._settings_factory = settings_factory
        self._content_get = content_get
        self._task_id_factory = task_id_factory
        self._clock = clock
        self._persistence_factory = persistence_factory

    async def _record_usage(self, task_id: str, duration_seconds: float) -> None:
        persistence = self._persistence_factory()
        if persistence is None:
            return
        try:
            await persistence.store_video_task_usage(
                _billing_key(task_id),
                duration_seconds,
                CAUSYN_RESOLUTION,
            )
        except Exception:  # noqa: BLE001  # a failed record must not turn an accepted generation into an error
            logger.warning("causyn video billing: failed to record task usage at create", exc_info=True)

    async def _bill_completed_video(
        self,
        response: VideoObject,
        task_id: str,
        optional_params: dict[str, object],
    ) -> None:
        _set_response_cost(response, 0.0)
        persistence = self._persistence_factory()
        if persistence is None:
            return
        try:
            raw_usage = await persistence.get_video_task_usage(_billing_key(task_id))
        except Exception:  # noqa: BLE001  # a failed lookup must skip charging rather than guess
            logger.warning("causyn video billing: usage lookup failed, skipping charge", exc_info=True)
            return
        try:
            usage = _StoredUsage.model_validate(raw_usage)
        except ValidationError:
            logger.warning("causyn video billing: no valid usage record for completed task")
            return
        response.usage = _usage(usage.duration_seconds)
        cost = _completion_cost(optional_params, usage.duration_seconds)
        if cost is None:
            logger.warning("causyn video billing: no valid price for completed task")
            return
        try:
            billed = await persistence.mark_video_billed(
                _billing_key(task_id),
                usage.duration_seconds,
                cost,
            )
        except Exception:  # noqa: BLE001  # a failed idempotency check must never risk charging twice
            logger.warning("causyn video billing: persistence check failed, skipping charge", exc_info=True)
            return
        _set_response_cost(response, cost if billed else 0.0)

    def video_generation(
        self,
        model: str,
        prompt: str,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        logging_obj: object,
        timeout: RequestTimeout = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        raise NotImplementedError("causyn video generation is async-only")

    async def avideo_generation(
        self,
        model: str,
        prompt: str,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        logging_obj: object,
        timeout: RequestTimeout = None,
        client: AsyncHTTPHandler | None = None,
    ) -> VideoObject:
        if not _enabled():
            raise CustomLLMError(status_code=403, message="causyn video generation is disabled")
        request, duration = _request(model, prompt, optional_params)
        task_id = self._task_id_factory()
        if _TASK_ID_PATTERN.fullmatch(task_id) is None:
            raise _service_error()
        payload = {
            "task_id": task_id,
            "model": CAUSYN_MODEL,
            "deadline_ts": self._clock() + CAUSYN_DEADLINE_SECONDS,
            "request": request,
        }
        try:
            settings = self._settings_factory()
            await enqueue_video_generate(payload, redis_factory=self._redis_factory, settings=settings)
        except VideoGenerateError as error:
            status_code = 400 if error.code in {"invalid_params", "invalid_url"} else 503
            message = "invalid causyn video request" if status_code == 400 else "causyn video service unavailable"
            raise CustomLLMError(status_code=status_code, message=message) from None
        except Exception:  # noqa: BLE001  # replace unknown driver details with a stable public error
            raise _service_error() from None
        await self._record_usage(task_id, float(duration))
        response = VideoObject(
            id=f"{CAUSYN_VIDEO_ID_PREFIX}{task_id}",
            object="video",
            status="queued",
            created_at=int(self._clock()),
            seconds=str(duration),
            size=CAUSYN_RESOLUTION,
            model=CAUSYN_MODEL,
            usage=_usage(duration),
        )
        _set_response_cost(response, 0.0)
        return response

    def video_status(
        self,
        video_id: str,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        logging_obj: object,
        timeout: RequestTimeout = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        raise NotImplementedError("causyn video status is async-only")

    async def _status(self, video_id: str) -> _StatusEnvelope:
        task_id = _decode_task_id(video_id)
        try:
            raw_body: object = await fetch_video_generate_status(  # pyright: ignore[reportUnknownVariableType]  # validated below
                task_id, redis=self._redis_factory()
            )
        except VideoGenerateError:
            raise _service_error() from None
        except Exception:  # noqa: BLE001  # replace unknown driver details with a stable public error
            raise _service_error() from None
        try:
            body = _StatusEnvelope.model_validate(raw_body)
        except ValidationError:
            raise _service_error() from None
        if body.status is None:
            raise CustomLLMError(status_code=404, message="causyn video was not found")
        return body

    async def avideo_status(
        self,
        video_id: str,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        logging_obj: object,
        timeout: RequestTimeout = None,
        client: AsyncHTTPHandler | None = None,
    ) -> VideoObject:
        body = await self._status(video_id)
        status = body.status
        if status == "queued":
            return VideoObject(id=video_id, object="video", status="queued", model=CAUSYN_MODEL)
        if status == "claimed":
            return VideoObject(id=video_id, object="video", status="in_progress", model=CAUSYN_MODEL)
        if status != "succeeded":
            return VideoObject(
                id=video_id,
                object="video",
                status="failed",
                model=CAUSYN_MODEL,
                error=_public_worker_error(body.error),
            )
        result = body.result
        if result is None:
            raise _service_error("causyn video result is unavailable")
        task_id = _decode_task_id(video_id)
        if result.staging_key != _staging_key(task_id):
            raise _service_error("causyn video result is unavailable")
        duration = result.duration_seconds
        object_store_result = result.model_dump(exclude={"staging_url"}, exclude_none=True)
        response = VideoObject(
            id=video_id,
            object="video",
            status="completed",
            completed_at=int(self._clock()),
            seconds=str(duration),
            size=CAUSYN_RESOLUTION,
            model=CAUSYN_MODEL,
            usage=_usage(duration),
            object_store_result=object_store_result,
        )
        await self._bill_completed_video(response, task_id, optional_params)
        return response

    def video_content(
        self,
        video_id: str,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        logging_obj: object,
        timeout: RequestTimeout = None,
        client: HTTPHandler | None = None,
    ) -> bytes:
        raise NotImplementedError("causyn video content is async-only")

    async def avideo_content(
        self,
        video_id: str,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        logging_obj: object,
        timeout: RequestTimeout = None,
        client: AsyncHTTPHandler | None = None,
    ) -> bytes:
        body = await self._status(video_id)
        if body.status != "succeeded":
            raise CustomLLMError(status_code=409, message="causyn video is not ready")
        result = body.result
        staging_url = result.staging_url if result is not None else None
        task_id = _decode_task_id(video_id)
        if result is None or result.staging_key != _staging_key(task_id) or not isinstance(staging_url, str):
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable")
        try:
            settings = self._settings_factory()
            validate_video_generate_url(staging_url, settings.target_hosts, settings, "staging download")
        except VideoGenerateError:
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable") from None
        try:
            response = await self._content_get(staging_url, timeout, False)
        except Exception:  # noqa: BLE001  # do not expose signed URLs or HTTP client details
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable") from None
        if response.status_code != 200:
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable")
        return response.content


causyn_video_handler = CausynVideoHandler()

__all__ = ["CausynVideoHandler", "causyn_video_handler"]
