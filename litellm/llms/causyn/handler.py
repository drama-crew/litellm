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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, cast
from urllib.parse import SplitResult, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.llms.causyn.context_ir_store import BillingIdentity, ContextIRTask, task_key, PREFIX as CONTEXT_IR_PREFIX
from litellm.llms.causyn.h3_prompt import RewriteError
from litellm.llms.causyn.vdn_geometry import pixel_budget, resolve_geometry
from litellm.llms.causyn.video_prompt import VideoPromptInput, VideoSubmission, submit_video_prompt
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
from litellm.llms.causyn.topaz import TopazAdvance, TopazIndeterminateError, TopazRedis, TopazVideoAdapter

CAUSYN_MODEL = "causyn-1.0"
CAUSYN_H3_MODEL = "causyn-1.1"
PROVIDER = "causyn"
CAUSYN_VIDEO_ID_PREFIX = "causyn_"
LEGACY_VIDEO_ID_PREFIX = CAUSYN_VIDEO_ID_PREFIX
CAUSYN_RESOLUTION = "768x512"
CAUSYN_2K_RESOLUTION = "2k"
CAUSYN_RESOLUTIONS = frozenset({CAUSYN_RESOLUTION, CAUSYN_2K_RESOLUTION})
CAUSYN_H3_RESOLUTION = "768p"
CAUSYN_H3_RATIOS = frozenset({"16:9", "9:16", "1:1", "4:3", "3:4"})
CAUSYN_BILLING_METADATA_VERSION = "causyn-video-billing-v1"
CAUSYN_BILLING_METADATA_VERSION_V2 = "causyn-video-billing-v2"
CAUSYN_BILLING_METADATA_VERSION_V3 = "causyn-video-billing-v3"
CAUSYN_BILLING_METADATA_VERSION_V4 = "causyn-video-billing-v4"
CAUSYN_RATIO = "3:2"
CAUSYN_DEADLINE_SECONDS = 1800.0
CAUSYN_H3_DEADLINE_SECONDS = 1800.0
_INTERNAL_VIDEO_FLAG = "DRAMA_INTERNAL_VIDEO_ENABLED"
_INTERNAL_VIDEO_FLAG_ALIAS = "OH_DRAMA_INTERNAL_VIDEO_ENABLED"
_CAUSYN_H3_FLAG = "DRAMA_CAUSYN_1_1_ENABLED"
_CAUSYN_H3_FLAG_ALIAS = "OH_DRAMA_CAUSYN_1_1_ENABLED"
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
        "image",
        "last_image",
        "generate_audio",
        "seed",
        # LiteLLM exposes this standard request field, but it is not sent to the
        # worker request payload.
        "user",
    }
)
_CAUSYN_FRAMEWORK_PARAMS = frozenset(all_litellm_params)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ModelSpec:
    model: str
    deployment_id: str
    flag: str
    flag_alias: str
    resolutions: frozenset[str]
    ratios: frozenset[str]
    duration_min: int
    duration_max: int
    deadline_seconds: float
    native_audio_required: bool
    supports_topaz: bool


_MODEL_SPECS: Mapping[str, _ModelSpec] = MappingProxyType(
    {
        CAUSYN_MODEL: _ModelSpec(
            model=CAUSYN_MODEL,
            deployment_id="causyn-1-0",
            flag=_INTERNAL_VIDEO_FLAG,
            flag_alias=_INTERNAL_VIDEO_FLAG_ALIAS,
            resolutions=CAUSYN_RESOLUTIONS,
            ratios=frozenset({CAUSYN_RATIO}),
            duration_min=3,
            duration_max=8,
            deadline_seconds=CAUSYN_DEADLINE_SECONDS,
            native_audio_required=False,
            supports_topaz=True,
        ),
        CAUSYN_H3_MODEL: _ModelSpec(
            model=CAUSYN_H3_MODEL,
            deployment_id="causyn-1-1",
            flag=_CAUSYN_H3_FLAG,
            flag_alias=_CAUSYN_H3_FLAG_ALIAS,
            resolutions=frozenset({CAUSYN_H3_RESOLUTION}),
            ratios=CAUSYN_H3_RATIOS,
            duration_min=4,
            duration_max=15,
            # One admitted 15s request measured 420.368s P100 and uses a 600s
            # H3 execution cap. This deadline starts earlier, at queue entry,
            # so it stays at 30 minutes to cover waiting behind prior jobs at
            # the measured production concurrency of one. Keep it independent
            # from the historical 1.0 pipeline and the downstream call cap.
            deadline_seconds=CAUSYN_H3_DEADLINE_SECONDS,
            native_audio_required=True,
            supports_topaz=False,
        ),
    }
)
_MODEL_SPECS_BY_DEPLOYMENT: Mapping[str, _ModelSpec] = MappingProxyType(
    {spec.deployment_id: spec for spec in _MODEL_SPECS.values()}
)


class ContentResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes: ...


RequestTimeout: TypeAlias = float | httpx.Timeout | None
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


def _parse_causyn_platform_origin(value: str) -> tuple[SplitResult, str, int | None] | None:
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return None
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return None
    if not parsed.netloc or "%" in parsed.netloc or not hostname:
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    return parsed, hostname, port


def _normalize_causyn_ipv6_authority(authority: str) -> str | None:
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
    return None if suffix == ":" else normalized_host


def _normalize_causyn_hostname_authority(authority: str, hostname: str) -> str | None:
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
        return None
    return hostname.lower()


def _normalize_causyn_platform_authority(authority: str, hostname: str) -> str | None:
    if authority.startswith("["):
        return _normalize_causyn_ipv6_authority(authority)
    return _normalize_causyn_hostname_authority(authority, hostname)


def _normalize_causyn_platform_origin(value: str) -> str | None:
    """Return a safe origin for the fixed platform refresh route.

    This is deliberately stricter than a generic URL parser: this setting is
    an origin, not a caller-controlled URL.  Rejecting authority delimiters,
    alternate IP spellings, and non-origin components prevents the value from
    changing either the host or the fixed refresh path.
    """
    parsed_origin = _parse_causyn_platform_origin(value)
    if parsed_origin is None:
        return None
    parsed, hostname, port = parsed_origin
    normalized_host = _normalize_causyn_platform_authority(parsed.netloc, hostname)
    if normalized_host is None:
        return None
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


def _default_redis_factory() -> TopazRedis:
    redis = get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL"))
    if redis is None:
        raise VideoGenerateError("misconfigured", "no redis URL configured for video-generate")
    return redis


# Kept as a module seam for the standard custom-provider tests and callers
# that replace the queue client without constructing a bespoke handler.
_redis_factory = _default_redis_factory


def _default_persistence_factory() -> VideoTaskPersistence | None:
    return get_persistence()


def _model_spec(model: str) -> _ModelSpec:
    normalized = model.split("/", 1)[-1]
    spec = _MODEL_SPECS.get(normalized)
    if spec is None:
        raise _bad_request("unsupported causyn model")
    return spec


def _enabled(model: str = CAUSYN_MODEL) -> bool:
    spec = _MODEL_SPECS.get(model)
    if spec is None:
        return False
    if spec.flag in os.environ:
        raw = os.environ[spec.flag]
    elif spec.flag_alias in os.environ:
        raw = os.environ[spec.flag_alias]
    else:
        return False
    return raw.strip().lower() in _INTERNAL_VIDEO_TRUE_VALUES


def _bad_request(message: str) -> CustomLLMError:
    return CustomLLMError(status_code=400, message=message)


# Ordered durations are whole seconds, so a genuine mis-render differs by >=1s.
# Half a second separates that from the pipeline's one-frame overshoot without
# hardcoding its frame rate.
_DURATION_RECONCILE_TOLERANCE_SECONDS = 0.5


def _service_error(message: str = "causyn video service unavailable") -> CustomLLMError:
    return CustomLLMError(status_code=503, message=message)


def _decode_video_identity(video_id: str) -> tuple[str, _ModelSpec]:
    # Keep the original opaque form readable for tasks issued before provider
    # encoding was introduced. The shared decoder intentionally recognizes that
    # legacy prefix too, so inspect it first to avoid treating the whole legacy
    # id as the task id.
    if video_id.startswith(CAUSYN_VIDEO_ID_PREFIX):
        task_id = video_id[len(CAUSYN_VIDEO_ID_PREFIX) :]
        spec = _MODEL_SPECS[CAUSYN_MODEL]
    else:
        decoded = decode_video_id_with_provider(video_id)
        if decoded.get("custom_llm_provider") != PROVIDER:
            raise _bad_request("invalid causyn video id")
        task_id = decoded.get("video_id") or ""
        decoded_spec = _MODEL_SPECS_BY_DEPLOYMENT.get(decoded.get("model_id") or "")
        if decoded_spec is None:
            raise _bad_request("invalid causyn video id")
        spec = decoded_spec
    if _TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise _bad_request("invalid causyn video id")
    return task_id, spec


def _decode_task_id(video_id: str) -> str:
    return _decode_video_identity(video_id)[0]


def _duration(optional_params: dict[str, object], spec: _ModelSpec) -> int:
    raw = optional_params.get("seconds")
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdigit():
        raise _bad_request(f"seconds must be an integer string from {spec.duration_min} through {spec.duration_max}")
    value = int(raw)
    if value < spec.duration_min or value > spec.duration_max:
        raise _bad_request(f"seconds must be from {spec.duration_min} through {spec.duration_max}")
    return value


def _legacy_references(
    optional_params: dict[str, object],
) -> tuple[dict[str, str], ...]:
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


def _h3_shaped_references(value: object) -> tuple[dict[str, str], ...]:
    try:
        references = TypeAdapter(list[dict[str, str]]).validate_python(value, strict=True)
    except ValidationError:
        raise _bad_request("references must contain first_frame and optional last_frame") from None
    allowed = {"role", "media_type", "url"}
    if any(set(reference) != allowed for reference in references):
        raise _bad_request("references must contain first_frame and optional last_frame")
    by_role: dict[str, dict[str, str]] = {}
    for reference in references:
        role = reference["role"]
        if (
            role not in {"first_frame", "last_frame"}
            or role in by_role
            or reference["media_type"] != "image"
            or not reference["url"]
        ):
            raise _bad_request("references must contain first_frame and optional last_frame")
        by_role[role] = reference
    if "last_frame" in by_role and "first_frame" not in by_role:
        raise _bad_request("last_image requires image")
    return tuple(by_role[role] for role in ("first_frame", "last_frame") if role in by_role)


def _h3_references(optional_params: dict[str, object]) -> tuple[dict[str, str], ...]:
    reference_images = optional_params.get("reference_images")
    if reference_images not in (None, []):
        raise _bad_request("reference_images is not supported")
    shaped = optional_params.get("references")
    image = optional_params.get("image")
    last_image = optional_params.get("last_image")
    if shaped is not None:
        if image is not None or last_image is not None:
            raise _bad_request("references cannot be combined with image or last_image")
        return _h3_shaped_references(shaped)
    for name, value in (("image", image), ("last_image", last_image)):
        if value is not None and (not isinstance(value, str) or not value):
            raise _bad_request(f"{name} must be a non-empty URL string")
    if last_image is not None and image is None:
        raise _bad_request("last_image requires image")
    references: list[dict[str, str]] = []
    if isinstance(image, str):
        references.append({"role": "first_frame", "media_type": "image", "url": image})
    if isinstance(last_image, str):
        references.append({"role": "last_frame", "media_type": "image", "url": last_image})
    return tuple(references)


def _resolution(optional_params: dict[str, object], spec: _ModelSpec) -> str:
    missing = object()
    resolution = optional_params.get("resolution", missing)
    size = optional_params.get("size", missing)
    for name, value in (("resolution", resolution), ("size", size)):
        if value is not missing and not isinstance(value, str):
            raise _bad_request(f"{name} must be a string")
    if resolution is not missing and size is not missing and resolution != size:
        raise _bad_request("resolution and size must match")
    canonical = resolution if resolution is not missing else size
    if canonical not in spec.resolutions:
        offered = ", ".join(sorted(spec.resolutions))
        raise _bad_request(f"resolution must be one of {offered}")
    return str(canonical)


def _h3_source_resolution(ratio: str) -> str:
    width_ratio, height_ratio = (int(value) for value in ratio.split(":"))
    short_edge = 768
    if width_ratio >= height_ratio:
        height = short_edge
        width = int(short_edge * width_ratio / height_ratio) // 32 * 32
    else:
        width = short_edge
        height = int(short_edge * height_ratio / width_ratio) // 32 * 32
    return f"{width}x{height}"


def _source_resolution(spec: _ModelSpec, ratio: str) -> str:
    if spec.model == CAUSYN_H3_MODEL:
        return _h3_source_resolution(ratio)
    return CAUSYN_RESOLUTION


def _vdn_source_resolution(ratio: str, duration: int) -> str:
    if ratio == "adaptive":
        return "adaptive"
    frames = duration * 24 + (5 - duration * 24) % 17
    geometry = resolve_geometry(ratio=ratio, frames=frames)
    return f"{geometry.output_width}x{geometry.output_height}"


def _request(
    model: str, prompt: object, optional_params: dict[str, object]
) -> tuple[dict[str, object], int, str, str, _ModelSpec]:
    spec = _model_spec(model)
    if not isinstance(prompt, str) or not prompt.strip():
        raise _bad_request("prompt must be a non-empty string")
    unsupported = sorted(
        key for key in optional_params if key not in _CAUSYN_USER_PARAMS and key not in _CAUSYN_FRAMEWORK_PARAMS
    )
    if unsupported:
        raise _bad_request(f"unsupported causyn video parameter: {', '.join(unsupported)}")
    requested_resolution = _resolution(optional_params, spec)
    references = (
        _h3_references(optional_params) if spec.model == CAUSYN_H3_MODEL else _legacy_references(optional_params)
    )
    ratio = optional_params.get("aspect_ratio")
    if spec.model == CAUSYN_H3_MODEL and references:
        if ratio is not None and (not isinstance(ratio, str) or ratio not in {*spec.ratios, "adaptive", "21:9"}):
            raise _bad_request("unsupported aspect_ratio for keyframe generation")
        ratio = "adaptive"
    elif not isinstance(ratio, str) or ratio not in spec.ratios:
        offered = ", ".join(sorted(spec.ratios))
        raise _bad_request(f"aspect_ratio must be one of {offered}")
    generate_audio = optional_params.get("generate_audio")
    if generate_audio is not None and not isinstance(generate_audio, bool):
        raise _bad_request("generate_audio must be a boolean")
    if spec.native_audio_required and generate_audio is False:
        raise _bad_request("generate_audio must be true")
    for unsupported_param in (
        "input_reference",
        "reference_audios",
        "reference_videos",
    ):
        if optional_params.get(unsupported_param) is not None:
            raise _bad_request(f"{unsupported_param} is not supported")
    if spec.model == CAUSYN_MODEL:
        for unsupported_param in ("image", "last_image"):
            if optional_params.get(unsupported_param) is not None:
                raise _bad_request(f"{unsupported_param} is not supported")
    seed = optional_params.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise _bad_request("seed must be an integer")
    duration = _duration(optional_params, spec)
    source_resolution = (
        _vdn_source_resolution(ratio, duration) if spec.model == CAUSYN_H3_MODEL else _source_resolution(spec, ratio)
    )
    request: dict[str, object] = {
        "prompt": prompt,
        "duration_seconds": duration,
        "resolution": CAUSYN_H3_RESOLUTION if spec.model == CAUSYN_H3_MODEL else CAUSYN_RESOLUTION,
        "ratio": ratio,
        "generate_audio": True if generate_audio is None else generate_audio,
        "references": list(references),
    }
    if seed is not None:
        request["seed"] = seed
    return request, duration, requested_resolution, source_resolution, spec


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
    # No upper bound: this is what the device produced, not what was ordered.
    # The pipeline renders duration*24 + 1 frames, so the top of the orderable
    # range (8s) always comes back as 8.041667 -- the le=8 copied from the order
    # bounds therefore rejected 100% of 8-second jobs. A rejection here raises a
    # 503 on the status poll, which the platform treats as terminal, so a video
    # already sitting in the object store was reported to the user as a failure.
    # The lower bound stays: a shorter-than-ordered result is a garbled report,
    # and nothing about frame quantisation shortens a render.
    duration_seconds: float = Field(ge=3)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _StatusEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    status: Literal["queued", "claimed", "succeeded", "failed"] | None = None
    error: _WorkerError | None = None
    result: _WorkerResult | None = None


class _Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: Literal["causyn-1.0", "causyn-1.1"]
    id: str = Field(min_length=1)
    output_cost_per_second_768x512: float | None = Field(default=None, gt=0)
    output_cost_per_second_768p: float | None = Field(default=None, gt=0)
    output_cost_per_second_2k: float | None = Field(default=None, gt=0)
    output_cost_per_second: float | None = Field(default=None, gt=0)

    @field_validator(
        "output_cost_per_second_768x512",
        "output_cost_per_second_768p",
        "output_cost_per_second_2k",
        "output_cost_per_second",
    )
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
    def _requires_rate(self) -> _Pricing:
        if (
            self.output_cost_per_second_768x512 is None
            and self.output_cost_per_second_768p is None
            and self.output_cost_per_second_2k is None
            and self.output_cost_per_second is None
        ):
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

    version: Literal["causyn-video-billing-v1"]
    duration_seconds: float = Field(ge=3, le=8)
    video_resolution: Literal["768x512"]
    pricing: _Pricing
    attribution: _DurableAttribution

    @model_validator(mode="after")
    def _requires_native_price(self) -> _DurableTaskMetadata:
        if self.pricing.model != CAUSYN_MODEL:
            raise ValueError("pricing model must match task model")
        if self.pricing.output_cost_per_second_768x512 is None and self.pricing.output_cost_per_second is None:
            raise ValueError("768x512 pricing rate is required")
        return self


class _DurableTaskMetadataV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["causyn-video-billing-v2"]
    duration_seconds: float = Field(ge=3, le=8)
    source_resolution: Literal["768x512"]
    requested_resolution: Literal["768x512", "2k"]
    pricing: _Pricing
    attribution: _DurableAttribution

    @model_validator(mode="after")
    def _requires_resolution_price(self) -> _DurableTaskMetadataV2:
        if self.pricing.model != CAUSYN_MODEL:
            raise ValueError("pricing model must match task model")
        if self.requested_resolution == "2k" and self.pricing.output_cost_per_second_2k is None:
            raise ValueError("2k pricing rate is required")
        if self.requested_resolution == "768x512":
            if self.pricing.output_cost_per_second_768x512 is None and self.pricing.output_cost_per_second is None:
                raise ValueError("768x512 pricing rate is required")
        return self


class _DurableTaskMetadataV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["causyn-video-billing-v3"]
    model: Literal["causyn-1.1"]
    duration_seconds: float = Field(ge=4, le=15)
    source_resolution: str = Field(pattern=r"^[1-9][0-9]*x[1-9][0-9]*$")
    requested_resolution: Literal["768p"]
    ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
    pricing: _Pricing
    attribution: _DurableAttribution

    context_ir_task_id: str | None = None
    prompt_rewrite_model: str | None = None
    prompt_rewrite_system_sha256: str | None = None

    @model_validator(mode="after")
    def _requires_h3_contract(self) -> _DurableTaskMetadataV3:
        if self.pricing.model != self.model:
            raise ValueError("pricing model must match task model")
        if self.pricing.output_cost_per_second_768p is None:
            raise ValueError("768p pricing rate is required")
        if self.source_resolution != _h3_source_resolution(self.ratio):
            raise ValueError("source resolution must match ratio")
        return self


class _DurableTaskMetadataV4(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["causyn-video-billing-v4"]
    geometry_profile: Literal["vdn-adaptive-v1"] = "vdn-adaptive-v1"
    model: Literal["causyn-1.1"]
    duration_seconds: float = Field(ge=4, le=15)
    source_resolution: str = Field(pattern=r"^(adaptive|[1-9][0-9]*x[1-9][0-9]*)$")
    requested_resolution: Literal["768p"]
    ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "adaptive"]
    pricing: _Pricing
    attribution: _DurableAttribution
    context_ir_task_id: str | None = None
    prompt_rewrite_model: str | None = None
    prompt_rewrite_system_sha256: str | None = None

    @model_validator(mode="after")
    def _requires_h3_contract(self) -> _DurableTaskMetadataV4:
        if self.pricing.model != self.model or self.pricing.output_cost_per_second_768p is None:
            raise ValueError("matching 768p pricing is required")
        if not self.duration_seconds.is_integer():
            raise ValueError("ordered duration must be whole seconds")
        if self.source_resolution != _vdn_source_resolution(self.ratio, int(self.duration_seconds)):
            raise ValueError("source resolution must match the admitted geometry profile")
        return self


_TaskMetadata = _DurableTaskMetadata | _DurableTaskMetadataV2 | _DurableTaskMetadataV3 | _DurableTaskMetadataV4
_TASK_METADATA_ADAPTER: TypeAdapter[_TaskMetadata] = TypeAdapter(_TaskMetadata)


def _metadata_model(metadata: _TaskMetadata) -> str:
    if isinstance(metadata, (_DurableTaskMetadataV3, _DurableTaskMetadataV4)):
        return metadata.model
    return CAUSYN_MODEL


def _metadata_resolution(metadata: _TaskMetadata) -> tuple[str, str]:
    if isinstance(metadata, _DurableTaskMetadata):
        return metadata.video_resolution, metadata.video_resolution
    return metadata.requested_resolution, metadata.source_resolution


def _result_geometry_matches(metadata: _TaskMetadata, result: _WorkerResult) -> bool:
    _, source_resolution = _metadata_resolution(metadata)
    if not isinstance(metadata, _DurableTaskMetadataV4) or metadata.ratio != "adaptive":
        return source_resolution == f"{result.width}x{result.height}"
    width, height = result.width, result.height
    frames = int(metadata.duration_seconds) * 24
    frames += (5 - frames) % 17
    budget = pixel_budget(frames)
    return (
        all(256 <= value <= 1536 and value % 2 == 0 for value in (width, height))
        and 0.399 <= width / height <= 2.506
        and math.ceil(width / 32) * math.ceil(height / 32) * 1024 <= budget
        and width * height >= 0.9 * min(768 * 768, budget)
    )


def _public_worker_error(error: _WorkerError | None) -> dict[str, str]:
    raw_code = error.code if error is not None else None
    code = raw_code if isinstance(raw_code, str) and _ERROR_CODE_PATTERN.fullmatch(raw_code) else "worker_failed"
    raw_kind = error.kind if error is not None else None
    kind = raw_kind if raw_kind in {"permanent", "transient", "unknown"} else "unknown"
    return {"code": code, "message": "video generation failed", "kind": kind}


def _usage(duration: float, resolution: str = CAUSYN_RESOLUTION) -> dict[str, float | str]:
    return {"duration_seconds": duration, "video_resolution": resolution}


def _billing_key(task_id: str) -> str:
    return f"causyn:{task_id}"


def _staging_key(task_id: str) -> str:
    return f"staging/video-tasks/{task_id}.mp4"


def _completion_cost(pricing: _Pricing, duration_seconds: float, resolution: str = CAUSYN_RESOLUTION) -> float:
    if resolution == CAUSYN_2K_RESOLUTION:
        rate = pricing.output_cost_per_second_2k
    elif resolution == CAUSYN_H3_RESOLUTION:
        rate = pricing.output_cost_per_second_768p
    else:
        rate = pricing.output_cost_per_second_768x512 or pricing.output_cost_per_second
    if rate is None:
        raise ValueError("pricing rate is unavailable")
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


def _first_billing_value(values: dict[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if values.get(key) is not None:
            return values[key]
    return None


def _billing_auth_value(auth: object, key: str) -> object:
    if isinstance(auth, dict):
        return auth.get(key)
    return getattr(auth, key, None)


def _billing_identifier(values: dict[str, object], auth: object, keys: tuple[str, ...], auth_attr: str) -> str | None:
    value = _first_billing_value(values, keys)
    if value is None:
        value = _billing_auth_value(auth, auth_attr)
    return value if isinstance(value, str) else None


def _billing_attribution(optional_params: dict[str, object], logging_obj: object) -> dict[str, str]:
    """Extract proxy-auth metadata, whose token value is already hashed."""
    auth = optional_params.get("user_api_key_dict")
    values: dict[str, object] = {}
    for source in _metadata_sources(optional_params, logging_obj):
        for key, value in source.items():
            values.setdefault(key, value)

    api_key = _hashed_api_key(_first_billing_value(values, ("user_api_key_hash", "user_api_key")))
    if api_key is None:
        api_key = _hashed_api_key(_billing_auth_value(auth, "api_key"))
    attribution: dict[str, str] = {}
    if api_key is not None:
        attribution["api_key"] = api_key
    for field, keys, auth_attr in (
        ("team_id", ("user_api_key_team_id", "team_id"), "team_id"),
        ("user_id", ("user_api_key_user_id", "user_id"), "user_id"),
        ("organization_id", ("user_api_key_org_id", "org_id", "organization_id"), "org_id"),
    ):
        value = _billing_identifier(values, auth, keys, auth_attr)
        if value is not None:
            attribution[field] = value
    return attribution


def _model_info_from_logging(logging_obj: object) -> object:
    raw: object = None
    details = getattr(logging_obj, "model_call_details", None)
    if isinstance(details, dict):
        raw = details.get("model_info")
        params = details.get("litellm_params")
        if raw is None and isinstance(params, dict):
            raw = params.get("model_info")
    return raw


def _pricing_model_info(optional_params: dict[str, object], logging_obj: object) -> object:
    raw = _model_info_from_logging(logging_obj)
    if raw is None:
        for source in _metadata_sources(optional_params, logging_obj):
            candidate = source.get("model_info")
            if isinstance(candidate, dict):
                raw = candidate
                break
    if raw is None:
        raw = optional_params.get("model_info")
    return raw


def _pricing_identity(
    optional_params: dict[str, object], logging_obj: object, model: str = CAUSYN_MODEL
) -> dict[str, object]:
    raw = _pricing_model_info(optional_params, logging_obj)
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return {"model": model}
    pricing: dict[str, object] = {"model": model}
    model_id = raw.get("id")
    if isinstance(model_id, str) and model_id:
        pricing["id"] = model_id
    for key in (
        "output_cost_per_second_768x512",
        "output_cost_per_second_768p",
        "output_cost_per_second_2k",
        "output_cost_per_second",
    ):
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
        redis_factory: Callable[[], TopazRedis] | None = None,
        settings_factory: Callable[[], VideoGenerateSettings] = VideoGenerateSettings.from_environment,
        content_get: ContentGet = _default_content_get,
        refresh_staging_url: RefreshUrl = _default_refresh_staging_url,
        task_id_factory: Callable[[], str] = _new_task_id,
        clock: Callable[[], float] = time.time,
        persistence_factory: PersistenceFactory = _default_persistence_factory,
        billing_enqueue: BillingEnqueue | None = None,
        topaz_adapter: TopazVideoAdapter | None = None,
        prompt_submit: Callable[[VideoSubmission, BillingIdentity], Awaitable[None]] = submit_video_prompt,
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
        self._topaz_adapter = topaz_adapter
        self._prompt_submit = prompt_submit

    async def _bill_completed_video(
        self,
        response: VideoObject,
        task_id: str,
        optional_params: dict[str, object],
        result: _WorkerResult,
        logging_obj: object,
        spec: _ModelSpec,
    ) -> None:
        # The worker result is the authoritative usage fact. The Redis metadata
        # key only carries immutable pricing/attribution captured at enqueue;
        # neither depends on the best-effort usage DB side channel.
        try:
            task_metadata = await fetch_video_generate_task_metadata(task_id, redis=self._redis_factory())
        except Exception as exc:  # noqa: BLE001
            raise _service_error() from exc
        try:
            durable = _TASK_METADATA_ADAPTER.validate_python(task_metadata)
        except ValidationError as exc:
            logger.warning("causyn video billing: task metadata is missing or invalid")
            raise _service_error() from exc
        if _metadata_model(durable) != spec.model:
            raise _service_error()
        requested_resolution, _ = _metadata_resolution(durable)
        # Ordered durations are whole seconds (3..8), so a worker that rendered
        # something other than what was ordered is off by at least a second.
        # Anything smaller is frame quantisation, not a mismatch: the pipeline
        # renders duration*24 + 1 frames at 24 fps, so every result comes back
        # ~0.0417s long by construction. The previous abs_tol=1e-6 demanded
        # equality and so rejected EVERY completed video -- and because a
        # rejection here surfaces as a 503 on the status poll, which the
        # platform treats as terminal, each one became a "生成失败" for a video
        # that was already sitting in the object store. 45 of them before this
        # was found.
        if abs(durable.duration_seconds - result.duration_seconds) > _DURATION_RECONCILE_TOLERANCE_SECONDS:
            logger.warning("causyn video billing: task metadata duration does not match worker result")
            raise _service_error()
        if not _result_geometry_matches(durable, result):
            logger.warning("causyn video billing: task metadata resolution does not match worker result")
            raise _service_error()
        response.usage = _usage(result.duration_seconds, requested_resolution)
        _set_response_cost(response, 0.0)
        # Never bill above the quote. The two differ by that extra frame on every
        # single render, so billing the worker's figure charged 5.0417s against a
        # 5s quote every time.
        try:
            cost = _completion_cost(
                durable.pricing, min(durable.duration_seconds, result.duration_seconds), requested_resolution
            )
        except ValueError as exc:
            raise _service_error() from exc
        attribution = durable.attribution.model_dump()
        try:
            event = CausynBillingEvent(
                provider_task_id=task_id,
                response_cost=cost,
                team_id=attribution.get("team_id"),
                user_id=attribution.get("user_id"),
                organization_id=attribution.get("organization_id"),
                api_key=attribution.get("api_key"),
                model=spec.model,
            )
            enqueued = await self._billing_enqueue(self._redis_factory(), event)
            if not enqueued:
                raise RuntimeError("durable outbox did not accept the billing event")
        except Exception as exc:  # noqa: BLE001  # status must remain retryable until durable delivery succeeds
            raise _service_error() from exc
        _set_response_cost(response, cost)

    def _completed_response(
        self,
        video_id: str,
        task_id: str,
        result: _WorkerResult,
        spec: _ModelSpec,
        resolution: str | None = None,
        object_store_result: bool = True,
    ) -> VideoObject:
        if result.staging_key != _staging_key(task_id):
            raise _service_error("causyn video result is unavailable")
        duration = result.duration_seconds
        if resolution is None:
            resolution = CAUSYN_H3_RESOLUTION if spec.model == CAUSYN_H3_MODEL else CAUSYN_RESOLUTION
        result_payload = result.model_dump(exclude={"staging_url"}, exclude_none=True)
        return VideoObject(
            id=video_id,
            object="video",
            status="completed",
            completed_at=int(self._clock()),
            seconds=str(duration),
            size=resolution,
            model=spec.model,
            usage=_usage(duration, resolution),
            object_store_result=result_payload if object_store_result else None,
        )

    async def _task_metadata(self, task_id: str) -> _TaskMetadata:
        try:
            raw = await fetch_video_generate_task_metadata(task_id, redis=self._redis_factory())
        except Exception as exc:
            raise _service_error() from exc
        try:
            return _TASK_METADATA_ADAPTER.validate_python(raw)
        except ValidationError as exc:
            raise _service_error() from exc

    def _get_topaz_adapter(self) -> TopazVideoAdapter:
        if self._topaz_adapter is None:
            try:
                self._topaz_adapter = TopazVideoAdapter.from_environment(
                    redis_factory=self._redis_factory, now=self._clock
                )
            except Exception as exc:
                raise _service_error() from exc
        return self._topaz_adapter

    async def _advance_topaz(self, task_id: str, timeout: RequestTimeout) -> TopazAdvance:
        settings = self._settings_factory()

        async def source_url(task: str) -> str:
            return await self._refresh_staging_url(task, timeout)

        def validate_source(url: str) -> None:
            validate_video_generate_url(url, settings.target_hosts, settings, "staging source")

        try:
            return await self._get_topaz_adapter().advance(task_id, source_url, validate_source=validate_source)
        except TopazIndeterminateError as exc:
            raise _service_error("causyn 2K processing indeterminate") from exc
        except Exception as exc:
            raise _service_error() from exc

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
        spec = _model_spec(model)
        if not _enabled(spec.model):
            raise CustomLLMError(status_code=403, message="causyn video generation is disabled")
        request, duration, requested_resolution, source_resolution, spec = _request(model, prompt, optional_params)
        task_id = self._task_id_factory()
        if _TASK_ID_PATTERN.fullmatch(task_id) is None:
            raise _service_error()
        task_metadata: dict[str, object] = {
            "version": CAUSYN_BILLING_METADATA_VERSION_V2,
            "duration_seconds": float(duration),
            "source_resolution": source_resolution,
            "requested_resolution": requested_resolution,
            "pricing": _pricing_identity(optional_params, logging_obj, spec.model),
            "attribution": _durable_attribution(optional_params, logging_obj),
        }
        metadata_type: type[_DurableTaskMetadataV2] | type[_DurableTaskMetadataV4]
        if spec.model == CAUSYN_H3_MODEL:
            task_metadata.update(
                {
                    "version": CAUSYN_BILLING_METADATA_VERSION_V4,
                    "model": spec.model,
                    "ratio": request["ratio"],
                }
            )
            metadata_type = _DurableTaskMetadataV4
        else:
            metadata_type = _DurableTaskMetadataV2
        try:
            metadata_model: _TaskMetadata = metadata_type.model_validate(task_metadata)
        except ValidationError as exc:
            logger.warning("causyn video billing: deployment pricing metadata is missing or invalid")
            raise _service_error() from exc
        if spec.model == CAUSYN_H3_MODEL:
            prompt_input = VideoPromptInput.model_validate(request)
            try:
                rewrite_settings = self._settings_factory()
                for reference in prompt_input.references:
                    validate_video_generate_url(
                        reference.url, rewrite_settings.source_hosts, rewrite_settings, reference.role
                    )
            except VideoGenerateError as exc:
                raise _bad_request("invalid causyn video reference") from exc
        durable_metadata = metadata_model
        serialized_metadata = durable_metadata.model_dump()
        serialized_metadata["pricing"] = durable_metadata.pricing.model_dump(exclude_none=True)
        payload = {
            "task_id": task_id,
            "model": spec.model,
            "deadline_ts": self._clock() + spec.deadline_seconds,
            "request": request,
            "task_metadata": serialized_metadata,
        }
        try:
            settings = self._settings_factory()
            if spec.model == CAUSYN_H3_MODEL:
                await self._prompt_submit(
                    VideoSubmission.model_validate(payload),
                    BillingIdentity.model_validate(metadata_model.attribution.model_dump()),
                )
            else:
                await enqueue_video_generate(payload, redis_factory=self._redis_factory, settings=settings)
        except RewriteError as exc:
            raise CustomLLMError(status_code=exc.status_code, message=str(exc)) from None
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
            id=encode_video_id_with_provider(task_id, PROVIDER, spec.deployment_id),
            object="video",
            status="queued",
            created_at=int(self._clock()),
            seconds=str(duration),
            size=requested_resolution,
            model=spec.model,
            usage=_usage(duration, requested_resolution),
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
        task_id, spec = _decode_video_identity(video_id)
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
        except ValidationError as exc:
            # The public error stays deliberately opaque, but swallowing the
            # cause entirely made this class of bug near-undiagnosable: an
            # out-of-range duration produced 3272 identical "service
            # unavailable" lines in fifteen minutes with nothing naming the
            # field. Log the reason; keep `from None` so no driver detail
            # reaches the caller.
            logger.warning(
                "causyn status envelope failed validation for task %s: %s",
                task_id,
                exc,
            )
            raise _service_error() from None
        if body.status is None:
            if spec.model == CAUSYN_H3_MODEL:
                return await self._rewrite_status(task_id)
            raise CustomLLMError(status_code=404, message="causyn video was not found")
        return body

    async def _rewrite_status(self, task_id: str) -> _StatusEnvelope:
        try:
            raw = await self._redis_factory().get(task_key(CONTEXT_IR_PREFIX + task_id))
            if raw is not None:
                task = ContextIRTask.model_validate_json(raw)
                if task.video_payload is not None:
                    return _StatusEnvelope.model_validate(
                        {
                            "status": "failed" if task.status in {"failed", "cancelled"} else "queued",
                            "error": {"code": "prompt_rewrite_failed", "message": task.error} if task.error else None,
                        }
                    )
        except Exception as exc:
            raise _service_error() from exc
        raise CustomLLMError(status_code=404, message="causyn video was not found")

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
        task_id, spec = _decode_video_identity(video_id)
        body = await self._status(video_id)
        status = body.status
        if status == "queued":
            return VideoObject(id=video_id, object="video", status="queued", model=spec.model)
        if status == "claimed":
            return VideoObject(id=video_id, object="video", status="in_progress", model=spec.model)
        if status != "succeeded":
            return VideoObject(
                id=video_id,
                object="video",
                status="failed",
                model=spec.model,
                error=_public_worker_error(body.error),
            )
        result = body.result
        if result is None:
            raise _service_error("causyn video result is unavailable")
        base_response = self._completed_response(video_id, task_id, result, spec)
        metadata = await self._task_metadata(task_id)
        if _metadata_model(metadata) != spec.model:
            raise _service_error()
        requested_resolution, _source_resolution = _metadata_resolution(metadata)
        if spec.supports_topaz and requested_resolution == CAUSYN_2K_RESOLUTION:
            topaz = await self._advance_topaz(task_id, timeout)
            if topaz.status == "in_progress":
                return VideoObject(
                    id=video_id,
                    object="video",
                    status="in_progress",
                    model=spec.model,
                )
            if topaz.status == "failed":
                return VideoObject(
                    id=video_id,
                    object="video",
                    status="failed",
                    model=spec.model,
                    error={"message": "causyn 2K processing failed", "kind": "provider"},
                )
            response = self._completed_response(
                video_id,
                task_id,
                result,
                spec,
                resolution=CAUSYN_2K_RESOLUTION,
                object_store_result=False,
            )
        else:
            response = base_response
        await self._bill_completed_video(response, task_id, optional_params, result, logging_obj, spec)
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
        task_id, spec = _decode_video_identity(video_id)
        body = await self._status(video_id)
        if body.status != "succeeded":
            raise CustomLLMError(status_code=409, message="causyn video is not ready")
        result = body.result
        if result is None:
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable")
        if result.staging_key != _staging_key(task_id):
            raise CustomLLMError(status_code=502, message="causyn video content is unavailable")
        metadata = await self._task_metadata(task_id)
        if _metadata_model(metadata) != spec.model:
            raise _service_error()
        requested_resolution, _source_resolution = _metadata_resolution(metadata)
        if spec.supports_topaz and requested_resolution == CAUSYN_2K_RESOLUTION:
            return await self._topaz_content(
                video_id, task_id, api_key, api_base, optional_params, logging_obj, timeout, client
            )
        response = self._completed_response(video_id, task_id, result, spec)
        await self._bill_completed_video(response, task_id, optional_params, result, logging_obj, spec)
        return await self._staging_content(task_id, timeout, response)

    async def _topaz_content(
        self,
        video_id: str,
        task_id: str,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        logging_obj: object,
        timeout: RequestTimeout,
        client: AsyncHTTPHandler | None,
    ) -> bytes:
        status_response = await self.avideo_status(
            video_id, api_key, api_base, optional_params, logging_obj, timeout, client
        )
        if status_response.status != "completed":
            raise CustomLLMError(status_code=409, message="causyn video is not ready")
        try:
            return await self._get_topaz_adapter().content(task_id)
        except TopazIndeterminateError as exc:
            raise _service_error("causyn 2K processing indeterminate") from exc
        except Exception as exc:
            raise _service_error() from exc

    async def _staging_content(self, task_id: str, timeout: RequestTimeout, response: VideoObject) -> bytes:
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
