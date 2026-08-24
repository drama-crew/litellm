"""causyn — queue-backed video provider for the intranet GPU worker fleet.

Every other video provider in this repo makes an outbound HTTP call to a
vendor API. This one cannot: the worker sits on an intranet machine that
accepts no inbound connections and only ever polls the platform. litellm
supports exactly this shape through ``litellm.custom_provider_map`` --
``litellm/videos/main.py``'s ``_custom_video_generation`` / ``_custom_video_status``
/ ``_custom_video_content`` dispatch to a handler without going anywhere near
``base_llm_http_handler``, and none of the three signatures implies a round
trip. So "submit" here means XADD onto a Redis stream, and "poll" means
reading that task's status back.

Design: docs/superpowers/specs/2026-08-23-causyn-litellm-provider-design.md

Three deliberate non-responsibilities, all resolved by keeping object-store
work on the platform side (which owns the credentials and already has a
correct, recently-debugged signing implementation):

* the staging upload URL is NOT signed here -- the platform's worker-runner
  injects it when the worker claims the task;
* the finished object is NOT read from OSS here -- the platform signs a GET
  at content-read time through an authenticated internal service endpoint, and
  ``avideo_content`` merely fetches that URL over plain HTTPS;
* consequently this module needs no object-store SDK and no OSS credentials.

The enqueue/poll engine itself is reused from ``litellm.llms.libtv.video_generate``
rather than reimplemented: that module already carries the fail-closed URL
allowlists, capacity admission, dedupe and status translation, all hardened
over a review pass. This handler is a standard-interface face on it, not a
second copy.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.libtv.billing_outbox import CausynBillingEvent, enqueue_causyn_billing
from litellm.llms.libtv.persistence import get_persistence
from litellm.llms.libtv.transfer import get_transfer_redis
from litellm.llms.libtv.video_generate import (
    VideoGenerateError,
    VideoGenerateSettings,
    enqueue_video_generate,  # pyright: ignore[reportUnknownVariableType]  # legacy engine has untyped Redis ports
    fetch_video_generate_task_metadata,
    fetch_video_generate_status,  # pyright: ignore[reportUnknownVariableType]  # legacy engine returns an untyped dict
    validate_video_generate_url,
)
from litellm.litellm_core_utils.asyncify import run_async_function  # pyright: ignore[reportUnknownVariableType]
from litellm.types.utils import all_litellm_params
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

CAUSYN_MODEL = "causyn-1.0"
PROVIDER = "causyn"
CAUSYN_VIDEO_ID_PREFIX = "causyn_"
LEGACY_VIDEO_ID_PREFIX = CAUSYN_VIDEO_ID_PREFIX
CAUSYN_RESOLUTION = "768x512"
CAUSYN_BILLING_METADATA_VERSION = "causyn-video-billing-v1"
CAUSYN_RATIO = "3:2"
CAUSYN_DEADLINE_SECONDS = 1800.0
_INTERNAL_VIDEO_FLAG = "DRAMA_INTERNAL_VIDEO_ENABLED"
_INTERNAL_VIDEO_FLAG_ALIAS = "OH_DRAMA_INTERNAL_VIDEO_ENABLED"
_INTERNAL_VIDEO_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_CAUSYN_PLATFORM_URL_ENV = "DRAMA_CAUSYN_PLATFORM_URL"
_CAUSYN_SERVICE_API_KEY_ENV = "DRAMA_CAUSYN_SERVICE_API_KEY"
_CAUSYN_REFRESH_PATH = "/api/service/causyn/tasks"
_TASK_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_STATUS_TO_OPENAI = {
    "queued": "queued",
    "claimed": "in_progress",
    "succeeded": "completed",
    "failed": "failed",
}
_CAUSYN_USER_PARAMS = frozenset(
    {
        "seconds",
        "resolution",
        "size",
        "aspect_ratio",
        "reference_images",
        "references",
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
RefreshUrl = Callable[[str, RequestTimeout], Awaitable[str]]


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
BillingEnqueue = Callable[[object, CausynBillingEvent], Awaitable[bool]]


async def _default_content_get(
    url: str,
    timeout: RequestTimeout,
    follow_redirects: bool,
) -> ContentResponse:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects) as client:
        return await client.get(url)


def _normalize_causyn_platform_origin(value: str) -> str | None:
    """Return a safe origin for the fixed platform refresh route.

    This is deliberately stricter than a generic URL parser: this setting is
    an origin, not a caller-controlled URL.  Rejecting authority delimiters,
    alternate IP spellings, and non-origin components prevents the value from
    changing either the host or the fixed refresh path.
    """
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return None
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return None
    if not parsed.netloc or "%" in parsed.netloc:
        return None
    try:
        hostname = parsed.hostname
    except ValueError:
        return None
    if not hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is not None and not 1 <= port <= 65535:
        return None

    authority = parsed.netloc
    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            return None
        literal = authority[1:closing]
        try:
            normalized_host = f"[{ipaddress.IPv6Address(literal).compressed.lower()}]"
        except ValueError:
            return None
        suffix = authority[closing + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return None
        if suffix == ":":
            return None
    else:
        if "[" in authority or "]" in authority or authority.count(":") > 1:
            return None
        if ":" in authority and not authority.rsplit(":", 1)[1].isdigit():
            return None
        host_part = authority.rsplit(":", 1)[0] if ":" in authority else authority
        if not host_part or host_part.lower() != hostname.lower():
            return None
        if not re.fullmatch(r"[A-Za-z0-9.-]+", hostname):
            return None
        if hostname.startswith(".") or hostname.endswith(".") or ".." in hostname:
            return None
        if all(char.isdigit() or char == "." for char in hostname):
            try:
                ipaddress.IPv4Address(hostname)
            except ValueError:
                return None
        elif hostname.lower().startswith("0x"):
            # Reject WHATWG-style hexadecimal IPv4 numbers instead of treating
            # them as a DNS label that another HTTP client may reinterpret.
            return None
        normalized_host = hostname.lower()

    normalized_port = f":{port}" if port is not None else ""
    return f"{parsed.scheme.lower()}://{normalized_host}{normalized_port}"


async def _default_refresh_staging_url(task_id: str, timeout: RequestTimeout) -> str:
    """Ask the platform to mint one short-lived URL for this task."""
    platform_url = _normalize_causyn_platform_origin(os.getenv(_CAUSYN_PLATFORM_URL_ENV, ""))
    service_key = os.getenv(_CAUSYN_SERVICE_API_KEY_ENV, "").strip()
    if not platform_url or not service_key:
        raise VideoGenerateError("misconfigured", "Causyn staging refresh is not configured")
    endpoint = f"{platform_url}{_CAUSYN_REFRESH_PATH}/{task_id}/staging-url"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(endpoint, headers={"X-Service-API-Key": service_key})
    if response.status_code != 200:
        raise VideoGenerateError("refresh_failed", "Causyn staging URL refresh failed")
    try:
        body = response.json()
    except ValueError as exc:
        raise VideoGenerateError("refresh_failed", "Causyn staging URL refresh failed") from exc
    expected_key = _staging_key(task_id)
    if not isinstance(body, dict) or body.get("staging_key") != expected_key or not isinstance(body.get("url"), str):
        raise VideoGenerateError("refresh_failed", "Causyn staging URL refresh failed")
    return body["url"]


def _new_task_id() -> str:
    return uuid.uuid4().hex


def _default_redis_factory() -> object:
    redis: object = get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL"))
    if redis is None:
        raise VideoGenerateError("misconfigured", "no redis URL configured for video-generate")
    return redis


# Kept as a module seam for the standard custom-provider tests and callers
# that replace the queue client without constructing a bespoke handler.
_redis_factory = _default_redis_factory


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
    # Keep the original opaque form readable for tasks issued before provider
    # encoding was introduced. The shared decoder intentionally recognizes that
    # legacy prefix too, so inspect it first to avoid treating the whole legacy
    # id as the task id.
    if video_id.startswith(CAUSYN_VIDEO_ID_PREFIX):
        task_id = video_id[len(CAUSYN_VIDEO_ID_PREFIX) :]
    else:
        decoded = decode_video_id_with_provider(video_id)
        if decoded.get("custom_llm_provider") != PROVIDER:
            raise _bad_request("invalid causyn video id")
        task_id = decoded.get("video_id") or ""
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
    shaped = optional_params.get("references")
    if shaped is not None:
        try:
            references = TypeAdapter(list[dict[str, str]]).validate_python(shaped, strict=True)
        except ValidationError:
            raise _bad_request("references must contain from 1 through 4 objects") from None
        if not 1 <= len(references) <= 4:
            raise _bad_request("references must contain from 1 through 4 objects")
        return tuple(references)
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
    missing = object()
    resolution = optional_params.get("resolution", missing)
    size = optional_params.get("size", missing)
    for name, value in (("resolution", resolution), ("size", size)):
        if value is not missing and not isinstance(value, str):
            raise _bad_request(f"{name} must be a string")
    if resolution is not missing and size is not missing and resolution != size:
        raise _bad_request("resolution and size must match")
    canonical = resolution if resolution is not missing else size
    if canonical != CAUSYN_RESOLUTION:
        raise _bad_request(f"resolution must be {CAUSYN_RESOLUTION}")
    return CAUSYN_RESOLUTION


def _request(model: str, prompt: object, optional_params: dict[str, object]) -> tuple[dict[str, object], int]:
    if model != CAUSYN_MODEL:
        raise _bad_request("unsupported causyn model")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _bad_request("prompt must be a non-empty string")
    unsupported = sorted(
        key for key in optional_params if key not in _CAUSYN_USER_PARAMS and key not in _CAUSYN_FRAMEWORK_PARAMS
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


class _Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: Literal[CAUSYN_MODEL]
    id: str = Field(min_length=1)
    output_cost_per_second_768x512: float | None = Field(default=None, gt=0)
    output_cost_per_second: float | None = Field(default=None, gt=0)

    @field_validator("output_cost_per_second_768x512", "output_cost_per_second")
    @classmethod
    def _requires_finite_rate(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("pricing rate must be finite")
        return value

    @field_validator("id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("pricing id must be non-empty")
        return value

    @model_validator(mode="after")
    def _requires_rate(self) -> "_Pricing":
        if self.output_cost_per_second_768x512 is None and self.output_cost_per_second is None:
            raise ValueError("pricing must include an output rate")
        return self


class _DurableAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    api_key: str | None
    team_id: str | None
    user_id: str | None
    organization_id: str | None

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, value: str | None) -> str | None:
        if value is not None and _hashed_api_key(value) is None:
            raise ValueError("api_key must be a hash")
        return value

    @field_validator("team_id", "user_id", "organization_id")
    @classmethod
    def _validate_owner_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("owner identifiers must be non-empty")
        return value


class _DurableTaskMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[CAUSYN_BILLING_METADATA_VERSION]
    duration_seconds: float = Field(ge=3, le=8)
    video_resolution: Literal[CAUSYN_RESOLUTION]
    pricing: _Pricing
    attribution: _DurableAttribution


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


def _completion_cost(pricing: _Pricing, duration_seconds: float) -> float:
    rate = pricing.output_cost_per_second_768x512 or pricing.output_cost_per_second
    assert rate is not None
    return rate * duration_seconds


def _metadata_sources(optional_params: dict[str, object], logging_obj: object) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []

    def add(value: object) -> None:
        if isinstance(value, dict):
            sources.append(value)

    details = getattr(logging_obj, "model_call_details", None)
    if isinstance(details, dict):
        add(details.get("metadata"))
        params = details.get("litellm_params")
        if isinstance(params, dict):
            add(params.get("metadata"))
    params = getattr(logging_obj, "litellm_params", None)
    if isinstance(params, dict):
        add(params.get("metadata"))
    # Proxy preprocessing is the authority for authenticated attribution. The
    # request metadata is only a compatibility fallback for direct callers and
    # must not overwrite values injected into the logging object.
    add(optional_params.get("metadata"))
    add(optional_params.get("litellm_metadata"))
    return sources


def _hashed_api_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) or re.fullmatch(r"hashed-jwt-[0-9a-fA-F]{64}", value):
        return value
    return None


def _billing_attribution(optional_params: dict[str, object], logging_obj: object) -> dict[str, str]:
    """Extract proxy-auth metadata, whose token value is already hashed."""
    auth = optional_params.get("user_api_key_dict")
    values: dict[str, object] = {}
    for source in _metadata_sources(optional_params, logging_obj):
        for key, value in source.items():
            values.setdefault(key, value)

    def first(*keys: str) -> object:
        for key in keys:
            if values.get(key) is not None:
                return values[key]
        return None

    def auth_value(key: str) -> object:
        if isinstance(auth, dict):
            return auth.get(key)
        return getattr(auth, key, None)

    api_key = _hashed_api_key(first("user_api_key_hash", "user_api_key"))
    if api_key is None:
        api_key = _hashed_api_key(auth_value("api_key"))
    attribution: dict[str, str] = {}
    if api_key is not None:
        attribution["api_key"] = api_key
    for field, keys, auth_attr in (
        ("team_id", ("user_api_key_team_id", "team_id"), "team_id"),
        ("user_id", ("user_api_key_user_id", "user_id"), "user_id"),
        ("organization_id", ("user_api_key_org_id", "org_id", "organization_id"), "org_id"),
    ):
        value = first(*keys)
        if value is None:
            value = auth_value(auth_attr)
        if isinstance(value, str):
            attribution[field] = value
    return attribution


def _pricing_identity(optional_params: dict[str, object], logging_obj: object) -> dict[str, object]:
    raw: object = None
    details = getattr(logging_obj, "model_call_details", None)
    if isinstance(details, dict):
        raw = details.get("model_info")
        params = details.get("litellm_params")
        if raw is None and isinstance(params, dict):
            raw = params.get("model_info")
    if raw is None:
        for source in _metadata_sources(optional_params, logging_obj):
            candidate = source.get("model_info")
            if isinstance(candidate, dict):
                raw = candidate
                break
    if raw is None:
        raw = optional_params.get("model_info")
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return {"model": CAUSYN_MODEL}
    pricing: dict[str, object] = {"model": CAUSYN_MODEL}
    model_id = raw.get("id")
    if isinstance(model_id, str) and model_id:
        pricing["id"] = model_id
    for key in ("output_cost_per_second_768x512", "output_cost_per_second"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            pricing[key] = float(value)
    return pricing


def _durable_attribution(optional_params: dict[str, object], logging_obj: object) -> dict[str, str | None]:
    attribution = _billing_attribution(optional_params, logging_obj)
    return {
        "api_key": attribution.get("api_key"),
        "team_id": attribution.get("team_id"),
        "user_id": attribution.get("user_id"),
        "organization_id": attribution.get("organization_id"),
    }


def _set_response_cost(response: VideoObject, response_cost: float) -> None:
    response._hidden_params = {"response_cost": response_cost}  # pyright: ignore[reportPrivateUsage]  # LiteLLM cost API


class CausynVideoHandler(CustomLLM):
    def __init__(
        self,
        *,
        redis_factory: Callable[[], object] | None = None,
        settings_factory: Callable[[], VideoGenerateSettings] = VideoGenerateSettings.from_environment,
        content_get: ContentGet = _default_content_get,
        refresh_staging_url: RefreshUrl = _default_refresh_staging_url,
        task_id_factory: Callable[[], str] = _new_task_id,
        clock: Callable[[], float] = time.time,
        persistence_factory: PersistenceFactory = _default_persistence_factory,
        billing_enqueue: BillingEnqueue | None = None,
    ) -> None:
        super().__init__()
        self._redis_factory = redis_factory or _redis_factory
        self._settings_factory = settings_factory
        self._content_get = content_get
        self._refresh_staging_url = refresh_staging_url
        self._task_id_factory = task_id_factory
        self._clock = clock
        self._persistence_factory = persistence_factory
        self._billing_enqueue = billing_enqueue if billing_enqueue is not None else enqueue_causyn_billing

    async def _bill_completed_video(
        self,
        response: VideoObject,
        task_id: str,
        optional_params: dict[str, object],
        result: _WorkerResult,
        logging_obj: object,
    ) -> None:
        # The worker result is the authoritative usage fact. The Redis metadata
        # key only carries immutable pricing/attribution captured at enqueue;
        # neither depends on the best-effort usage DB side channel.
        try:
            task_metadata = await fetch_video_generate_task_metadata(
                task_id, redis=self._redis_factory()
            )
        except Exception as exc:  # noqa: BLE001
            raise _service_error() from exc
        try:
            durable = _DurableTaskMetadata.model_validate(task_metadata)
        except ValidationError as exc:
            logger.warning("causyn video billing: task metadata is missing or invalid")
            raise _service_error() from exc
        if not math.isclose(durable.duration_seconds, result.duration_seconds, rel_tol=0.0, abs_tol=1e-6):
            logger.warning("causyn video billing: task metadata duration does not match worker result")
            raise _service_error()
        if durable.video_resolution != f"{result.width}x{result.height}":
            logger.warning("causyn video billing: task metadata resolution does not match worker result")
            raise _service_error()
        response.usage = _usage(result.duration_seconds)
        _set_response_cost(response, 0.0)
        cost = _completion_cost(durable.pricing, result.duration_seconds)
        attribution = durable.attribution.model_dump()
        try:
            event = CausynBillingEvent(
                provider_task_id=task_id,
                response_cost=cost,
                team_id=attribution.get("team_id"),
                user_id=attribution.get("user_id"),
                organization_id=attribution.get("organization_id"),
                api_key=attribution.get("api_key"),
                model=CAUSYN_MODEL,
            )
            enqueued = await self._billing_enqueue(self._redis_factory(), event)
            if not enqueued:
                raise RuntimeError("durable outbox did not accept the billing event")
        except Exception as exc:  # noqa: BLE001  # status must remain retryable until durable delivery succeeds
            raise _service_error() from exc
        _set_response_cost(response, cost)

    def _completed_response(self, video_id: str, task_id: str, result: _WorkerResult) -> VideoObject:
        if result.staging_key != _staging_key(task_id):
            raise _service_error("causyn video result is unavailable")
        duration = result.duration_seconds
        object_store_result = result.model_dump(exclude={"staging_url"}, exclude_none=True)
        return VideoObject(
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
        task_metadata = {
            "version": CAUSYN_BILLING_METADATA_VERSION,
            "duration_seconds": float(duration),
            "video_resolution": CAUSYN_RESOLUTION,
            "pricing": _pricing_identity(optional_params, logging_obj),
            "attribution": _durable_attribution(optional_params, logging_obj),
        }
        try:
            durable_metadata = _DurableTaskMetadata.model_validate(task_metadata)
        except ValidationError as exc:
            logger.warning("causyn video billing: deployment pricing metadata is missing or invalid")
            raise _service_error() from exc
        serialized_metadata = durable_metadata.model_dump()
        serialized_metadata["pricing"] = durable_metadata.pricing.model_dump(exclude_none=True)
        payload = {
            "task_id": task_id,
            "model": CAUSYN_MODEL,
            "deadline_ts": self._clock() + CAUSYN_DEADLINE_SECONDS,
            "request": request,
            "task_metadata": serialized_metadata,
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
        response = VideoObject(
            # The public endpoint decodes this deployment id before routing a
            # status/content request.  Keep the hyphenated form stable because
            # it is also the deployment id used by the router and auth layer.
            id=encode_video_id_with_provider(task_id, PROVIDER, "causyn-1-0"),
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
        return cast(
            VideoObject,
            run_async_function(
                self.avideo_status,
                video_id=video_id,
                api_key=api_key,
                api_base=api_base,
                optional_params=optional_params,
                logging_obj=logging_obj,
                timeout=timeout,
                client=None,
            ),
        )

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
        response = self._completed_response(video_id, task_id, result)
        await self._bill_completed_video(response, task_id, optional_params, result, logging_obj)
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
        return cast(
            bytes,
            run_async_function(
                self.avideo_content,
                video_id=video_id,
                api_key=api_key,
                api_base=api_base,
                optional_params=optional_params,
                logging_obj=logging_obj,
                timeout=timeout,
                client=None,
            ),
        )

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
        task_id = _decode_task_id(video_id)
        if result is None or result.staging_key != _staging_key(task_id):
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable")
        response = self._completed_response(video_id, task_id, result)
        await self._bill_completed_video(response, task_id, optional_params, result, logging_obj)
        try:
            staging_url = await self._refresh_staging_url(task_id, timeout)
        except Exception:  # noqa: BLE001  # do not expose internal service details
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable") from None
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
