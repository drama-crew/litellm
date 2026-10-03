from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
import logging
import re
import time
import unicodedata
import weakref
from contextvars import ContextVar
from collections import OrderedDict
from dataclasses import dataclass, replace
from email.utils import parsedate_to_datetime
from functools import lru_cache
from importlib.resources import files
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from litellm.llms.causyn import ref2va_plan
from litellm.llms.causyn.ref_media_facts import (
    AudioFacts,
    VideoFacts,
    analyze_audio,
    analyze_video,
    fmt_ts,
    keyframe_budget,
    render_audio_facts,
    render_video_facts,
)
from litellm.proxy.video_endpoints.minimax_h3_models import (
    AudioItem,
    ContentItem,
    ImageItem,
    MiniMaxH3Content,
    Ratio,
    TextItem,
    VideoItem,
)

BASE_REWRITE_DEFAULT_MODEL = "qwen/qwen3.8-flash"
MODEL = BASE_REWRITE_DEFAULT_MODEL
BASE_REWRITE_MODEL_ENV = "CAUSYN_H3_REWRITE_MODEL"
# Default Ref2VA rewrite model (evaluated: omni needs fewer repairs, writes fuller descriptions and is the only one
# that accepts audio at the same cost). Non-ref2va modes use CAUSYN_H3_REWRITE_MODEL (default MODEL).
REF2VA_DEFAULT_MODEL = "qwen/qwen3.8-omni-flash"
# Ref2VA rewrite model allow-list with each model's input modalities. Audio is sent only to `omni`.
REF2VA_MODEL_CAPS: dict[str, frozenset[str]] = {
    "qwen/qwen3.8-flash": frozenset({"text", "image", "video"}),
    "qwen/qwen3.8-omni-flash": frozenset({"text", "image", "audio", "video"}),
    "qwen/qwen3.8-max-0902": frozenset({"text", "image", "video"}),
}
# Per-model reasoning request and completion budget. max-0902 rejects `reasoning.enabled=false`.
REF2VA_MODEL_REASONING: dict[str, tuple[dict[str, JsonValue], int]] = {
    "qwen/qwen3.8-flash": ({"enabled": False}, 8192),
    "qwen/qwen3.8-omni-flash": ({"enabled": False}, 8192),
    "qwen/qwen3.8-max-0902": ({"enabled": True, "effort": "low"}, 12000),
}
MAX_RAW_VIDEO_BYTES = 20 * 1024 * 1024
REF2VA_MODEL_ENV = "CAUSYN_H3_REF2VA_REWRITE_MODEL"
SEND_VIDEO_ENV = "CAUSYN_H3_REWRITE_SEND_VIDEO"
PERCEPTION_BUDGET_S = 20.0
CUT_TOLERANCE_S = 0.35
DESCRIPTION_WORDS = (200, 750)
AUDIO_PART_BUDGET_BYTES = 12 * 1024 * 1024
MIN_REPAIR_BUDGET_S = 10.0
CRITIC_ENV = "CAUSYN_H3_REWRITE_CRITIC"
CRITIC_MODEL_ENV = "CAUSYN_H3_REWRITE_CRITIC_MODEL"
CRITIC_TIMEOUT_S = 20.0
CRITIC_MAX_DEFECTS = 8
CRITIC_DEFECT_CHARS = 300
ATTEMPT_BUDGET_S = 90.0
PUBLIC_MODEL = "causyn-h3-context-ir"
AUTH_MODEL = "causyn-1.1"
# causyn-1.1 reference-video limits. The Ref2VA engine enforces an exact memory/row admission
# guard (--max-ref-video-total-seconds 15) and returns its own 400 for over-budget combinations.
CAUSYN_VIDEO_REF_MAX_SECONDS = 15.0
CAUSYN_VIDEO_REF_TOTAL_MAX_SECONDS = 15.0
CAUSYN_REFERENCE_IMAGE_MAX_WITH_VIDEO = 4
PRICE_CREDITS = 4.0
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REFERENCE_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(?:access[-_ ]?key(?:[-_ ]?(?:id|secret))?|secret|token|api[-_ ]?key|password)\s*[=:]\s*\S+"),
    re.compile(r"[A-Za-z0-9_\-]{32,}"),
)
_DETAIL_LIMIT = 160
# Monotonic deadline of the rewrite attempt in flight; lets the context-ir service call
# size its poll budget to what the attempt timeout still allows.
ATTEMPT_DEADLINE: ContextVar[float | None] = ContextVar("causyn_attempt_deadline", default=None)


def redact_provider_detail(value: object) -> str | None:
    """Short, secret-free provider message for logs and stored task errors."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text[:_DETAIL_LIMIT] or None


def provider_error_detail(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except (ValueError, UnicodeDecodeError):
        return None
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    return redact_provider_detail(message)


class RewriteError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        *,
        retryable: bool | None = None,
        retry_after: str | None = None,
        upstream_status: int | None = None,
        detail: str | None = None,
        poll: bool = False,
        violations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        # Validator findings (each starts with a fixed phrase) for the one-shot repair message; never public.
        self.violations = violations
        # True for the base-mode structural checks added later (shot timestamps, last-frame alignment): every
        # older check had already passed when it is set, so an answer failing only these is still usable.
        self.new_checks_only = False
        self.status_code = status_code
        self.retryable = status_code == 429 if retryable is None else retryable
        self.retry_after = retry_after
        self.upstream_status = upstream_status
        # True: "task still running, check again" - not a failure, so no backoff.
        self.poll = poll
        self.detail = redact_provider_detail(detail)

    def describe_upstream(self) -> str | None:
        """Upstream status plus short redacted provider detail, for diagnosis (not public)."""
        extras = []
        if self.upstream_status is not None:
            extras.append(f"upstream {self.upstream_status}")
        if self.detail:
            extras.append(self.detail)
        return ": ".join(extras) or None

    def describe(self) -> str:
        detail = self.describe_upstream()
        return f"{self} ({detail})" if detail else str(self)

    def retry_delay(
        self,
        attempt: int,
        *,
        cap: float = 30.0,
        jitter: float = 1.0,
        retry_after_cap: float = 60.0,
        poll_fast: bool = False,
    ) -> float:
        if self.poll and poll_fast:
            return 1.0
        backoff = min(cap, 2.0**attempt) + random.uniform(0.0, jitter)
        if self.retry_after is None:
            return backoff
        try:
            seconds = float(self.retry_after)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(self.retry_after).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                return backoff
        return max(backoff, min(retry_after_cap, seconds))


class TruncatedRewriteError(RewriteError):
    """The provider cut the answer off at max_tokens. Permanent unless the one repair call can still produce it."""

    usage: "RewriteUsage"


@dataclass(frozen=True)
class RefFacts:
    """Deterministic perception of the reference media, aligned with the per-type `<Video N>` / `<Audio N>` order."""

    videos: tuple[VideoFacts | None, ...] = ()
    audios: tuple[AudioFacts | None, ...] = ()
    audio_raw: tuple[bytes, ...] = ()


_log = logging.getLogger(__name__)
_warned_models: set[str] = set()


def ref2va_model() -> str:
    """Selected Ref2VA rewrite model. An env value outside the allow-list fails closed (never a silent fallback)."""
    chosen = os.getenv(REF2VA_MODEL_ENV, "").strip()
    if not chosen:
        return REF2VA_DEFAULT_MODEL
    if chosen not in REF2VA_MODEL_CAPS:
        if chosen not in _warned_models:
            _warned_models.add(chosen)
            _log.warning("%s is not an allow-listed Ref2VA rewrite model: %r", REF2VA_MODEL_ENV, chosen[:80])
        raise RewriteError("H3 prompt rewrite model is not configured", 503, retryable=False)
    return chosen


def base_model() -> str:
    """Selected rewrite model for t2va/i2va/fl2va/l2va. An env value outside the allow-list fails closed."""
    chosen = os.getenv(BASE_REWRITE_MODEL_ENV, "").strip()
    if not chosen:
        return BASE_REWRITE_DEFAULT_MODEL
    if chosen not in REF2VA_MODEL_CAPS:
        if BASE_REWRITE_MODEL_ENV + chosen not in _warned_models:
            _warned_models.add(BASE_REWRITE_MODEL_ENV + chosen)
            _log.warning("%s is not an allow-listed rewrite model: %r", BASE_REWRITE_MODEL_ENV, chosen[:80])
        raise RewriteError("H3 prompt rewrite model is not configured", 503, retryable=False)
    return chosen


def send_video_enabled() -> bool:
    return os.getenv(SEND_VIDEO_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _audio_format(raw: bytes) -> Literal["wav", "mp3"] | None:
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    if raw[:3] == b"ID3":
        return "mp3"
    # MPEG-1/2 Layer III frame header: sync, version not reserved, layer bits 01. AAC ADTS has layer 00.
    if len(raw) > 1 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0 and (raw[1] >> 3) & 3 != 1 and (raw[1] >> 1) & 3 == 1:
        return "mp3"
    return None


_MEMO_SIZE = 32
FACTS_MEMO_MAX_BYTES = 64 * 1024 * 1024
_FACTS_MEMO: OrderedDict[str, RefFacts] = OrderedDict()
_FIRST_ANSWERS: OrderedDict[str, tuple[str, RewriteUsage]] = OrderedDict()
_PLAN_NOTES: OrderedDict[str, tuple[str, RewriteUsage]] = OrderedDict()


def _remember(store: OrderedDict, key: str, value: object) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > _MEMO_SIZE:
        store.popitem(last=False)


def _facts_bytes(facts: RefFacts) -> int:
    return sum(len(frame.jpeg) for video in facts.videos if video is not None for frame in video.keyframes)


def _remember_facts(key: str, facts: RefFacts) -> None:
    """Keep the facts (never the audio bytes) bounded by entry count and by total keyframe bytes."""
    _remember(_FACTS_MEMO, key, facts)
    while len(_FACTS_MEMO) > 1 and sum(_facts_bytes(f) for f in _FACTS_MEMO.values()) > FACTS_MEMO_MAX_BYTES:
        _FACTS_MEMO.popitem(last=False)


def media_key(spec: ContextIRRequest, videos: tuple[bytes, ...], audios: tuple[bytes, ...]) -> str:
    """Per-request key: text, duration, ratio, each item's type/role in order, image URLs and fetched media digests."""
    digest = hashlib.sha256()

    def feed(*parts: str | bytes) -> None:
        for part in parts:
            data = part.encode() if isinstance(part, str) else part
            digest.update(len(data).to_bytes(8, "big") + data)

    feed(spec.model, spec.prompt, str(spec.duration), spec.ratio)
    video_index = audio_index = 0
    for item in spec.content:
        if isinstance(item, TextItem):
            continue
        feed(type(item).__name__, item.role)
        if isinstance(item, ImageItem):
            feed(item.image_url.url)
        elif isinstance(item, VideoItem):
            feed(hashlib.sha256(videos[video_index]).digest() if video_index < len(videos) else b"")
            video_index += 1
        else:
            feed(hashlib.sha256(audios[audio_index]).digest() if audio_index < len(audios) else b"")
            audio_index += 1
    return digest.hexdigest()


_PERCEPTION_SLOTS = 2
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()
_warned_import = False


def _perception_semaphore() -> asyncio.Semaphore:
    """Process-wide cap on concurrent perception, created lazily per event loop (tests use one loop each)."""
    loop = asyncio.get_running_loop()
    semaphore = _semaphores.get(loop)
    if semaphore is None:
        semaphore = _semaphores[loop] = asyncio.Semaphore(_PERCEPTION_SLOTS)
    return semaphore


async def perceive_media(spec: ContextIRRequest, raw: object, key: str | None = None) -> RefFacts:
    """Perception for one request, memoised across durable-task retries (LRU of 32 entries and 64 MB of keyframes).

    A retry still re-fetches the media (the bytes are not cached) but skips the up to 20 s of CPU work. The cache never
    holds audio bytes: they are re-attached from the current request. Any failure degrades to missing facts. The work
    runs as a shielded task, so if the attempt is cancelled mid-way the finished facts still land in the memo.
    """
    videos_raw: tuple[bytes, ...] = getattr(raw, "videos", ())
    audios_raw: tuple[bytes, ...] = getattr(raw, "audios", ())
    if key is None:
        key = await asyncio.to_thread(media_key, spec, videos_raw, audios_raw)
    cached = _FACTS_MEMO.get(key)
    if cached is None:

        async def run() -> RefFacts:
            async with _perception_semaphore():
                facts = await _perceive(videos_raw, audios_raw)
            _remember_facts(key, replace(facts, audio_raw=()))
            return facts

        facts = await asyncio.shield(asyncio.ensure_future(run()))
    else:
        _FACTS_MEMO.move_to_end(key)
        facts = cached
    return replace(facts, audio_raw=audios_raw)


async def _perceive(videos_raw: tuple[bytes, ...], audios_raw: tuple[bytes, ...]) -> RefFacts:
    def work() -> RefFacts:
        global _warned_import
        try:
            import av  # noqa: F401
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            if not _warned_import:
                _warned_import = True
                _log.warning("Ref2VA perception is unavailable: a required imaging library is missing")
            return RefFacts(videos=(None,) * len(videos_raw), audios=(None,) * len(audios_raw), audio_raw=audios_raw)
        deadline = time.monotonic() + PERCEPTION_BUDGET_S
        budget = keyframe_budget(len(videos_raw))
        videos = tuple(analyze_video(data, max_keyframes=budget, deadline=deadline) for data in videos_raw)
        if any(video.cuts is None and not video.keyframes for video in videos):
            _log.warning("Ref2VA perception produced no shot cuts and no keyframes for a reference video")
        audios: list[AudioFacts | None] = []
        for data in audios_raw:
            try:
                audios.append(analyze_audio(data))
            except Exception:  # noqa: BLE001  # undecodable audio only loses its facts
                audios.append(None)
        return RefFacts(videos=videos, audios=tuple(audios), audio_raw=audios_raw)

    try:
        return await asyncio.to_thread(work)
    except Exception:  # noqa: BLE001
        _log.warning("Ref2VA perception failed and was skipped")
        return RefFacts(videos=(None,) * len(videos_raw), audios=(None,) * len(audios_raw), audio_raw=audios_raw)


class ContextIRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Literal["MiniMax-H3", "causyn-1.1"]
    content: tuple[ContentItem, ...] = Field(min_length=1, max_length=16)
    duration: int = Field(ge=4, le=15, strict=True)
    ratio: Ratio = "adaptive"
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        MiniMaxH3Content(
            model="MiniMax-H3", content=list(self.content), duration=self.duration, resolution="768P", ratio=self.ratio
        )
        return self

    @property
    def prompt(self) -> str:
        return next(item.text for item in self.content if isinstance(item, TextItem))

    @property
    def ordered_media(self) -> tuple[ImageItem | VideoItem | AudioItem, ...]:
        media = tuple(item for item in self.content if not isinstance(item, TextItem))
        return tuple(sorted(media, key=lambda item: {"first_frame": 0, "last_frame": 1}.get(item.role, 2)))

    @property
    def mode(self) -> Literal["t2va", "i2va", "l2va", "fl2va", "ref2va"]:
        roles = frozenset(item.role for item in self.ordered_media)
        if any(role.startswith("reference_") for role in roles):
            return "ref2va"
        if roles == frozenset({"first_frame", "last_frame"}):
            return "fl2va"
        if roles == frozenset({"last_frame"}):
            return "l2va"
        return "i2va" if roles else "t2va"

    @property
    def effective_ratio(self) -> Ratio:
        return "adaptive" if self.mode in {"i2va", "l2va", "fl2va"} else self.ratio

    def user_content(
        self, facts: RefFacts | None = None, model: str = MODEL, send_video: bool = False
    ) -> list[dict[str, JsonValue]]:
        task = {"t2va": "t2av", "i2va": "i2av", "l2va": "l2av", "fl2va": "fl2av", "ref2va": "Ref2VA"}[self.mode]
        ref2va = self.mode == "ref2va"
        parts: list[dict[str, JsonValue]] = []
        audio_budget = AUDIO_PART_BUDGET_BYTES
        for position, item in enumerate(self.ordered_media, 1):
            index = sum(type(previous) is type(item) for previous in self.ordered_media[:position])
            if ref2va and isinstance(item, VideoItem):
                parts.extend(self._perceived_video_parts(item, index, facts, send_video))
            elif ref2va and isinstance(item, AudioItem):
                audio_parts, audio_budget = self._perceived_audio_parts(index, facts, model, audio_budget)
                parts.extend(audio_parts)
            else:
                parts.extend(self._media_parts(item, index))
        parts.append(
            {
                "type": "text",
                "text": f"task: {task}\nresolution: {self.effective_ratio}\nduration: {self.duration}s\noriginal_prompt: {self.prompt}",
            }
        )
        return parts

    @staticmethod
    def _perceived_video_parts(
        item: VideoItem, index: int, facts: RefFacts | None, send_video: bool
    ) -> list[dict[str, JsonValue]]:
        label = f"<Video {index}>"
        parts: list[dict[str, JsonValue]] = [{"type": "text", "text": f"{label} reference video:\n"}]
        video = facts.videos[index - 1] if facts is not None and index <= len(facts.videos) else None
        frames = tuple(sorted(video.keyframes, key=lambda frame: frame.t)) if video is not None else ()
        if video is not None:
            parts.append({"type": "text", "text": render_video_facts(label, video) + "\n"})
        for frame in frames:
            parts.append({"type": "text", "text": f"{label} frame at {fmt_ts(frame.t)} (Shot {frame.shot})"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii")},
                }
            )
        # The raw video is opt-in only (never a fallback for failed perception) and bounded in size.
        if send_video and len(item.video_url.url) * 3 // 4 <= MAX_RAW_VIDEO_BYTES:
            parts.append({"type": "video_url", "video_url": {"url": item.video_url.url}})
        return parts

    @staticmethod
    def _perceived_audio_parts(
        index: int, facts: RefFacts | None, model: str, budget: int
    ) -> tuple[list[dict[str, JsonValue]], int]:
        label = f"<Audio {index}>"
        parts: list[dict[str, JsonValue]] = [{"type": "text", "text": f"{label} reference audio:\n"}]
        audio = facts.audios[index - 1] if facts is not None and index <= len(facts.audios) else None
        if audio is not None:
            parts.append({"type": "text", "text": render_audio_facts(label, audio)})
        raw = facts.audio_raw[index - 1] if facts is not None and index <= len(facts.audio_raw) else b""
        fmt = _audio_format(raw)
        if "audio" in REF2VA_MODEL_CAPS.get(model, frozenset()) and fmt is not None and len(raw) <= budget:
            parts.append(
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode("ascii"), "format": fmt}}
            )
            budget -= len(raw)
        return parts, budget

    @staticmethod
    def _media_parts(item: ImageItem | VideoItem | AudioItem, index: int) -> tuple[dict[str, JsonValue], ...]:
        if isinstance(item, ImageItem):
            return (
                {
                    "type": "text",
                    "text": (
                        f"Picture {index} — exact first frame at 0.00 seconds:\n"
                        if item.role == "first_frame"
                        else f"\nPicture {index} — exact final frame at the end of the target video:\n"
                        if item.role == "last_frame"
                        else f"<Picture {index}> reference image:\n"
                    ),
                },
                {"type": "image_url", "image_url": {"url": item.image_url.url}},
            )
        if isinstance(item, VideoItem):
            return (
                {"type": "text", "text": f"<Video {index}> reference video:\n"},
                {"type": "video_url", "video_url": {"url": item.video_url.url}},
            )
        return ({"type": "text", "text": f"<Audio {index}> reference audio:\n"},)


def causyn_reference_limit_violation(spec: ContextIRRequest) -> str | None:
    references = [item for item in spec.content if isinstance(item, ImageItem) and item.role == "reference_image"]
    videos = [item for item in spec.content if isinstance(item, VideoItem)]
    audios = [item for item in spec.content if isinstance(item, AudioItem)]
    if len(references) + len(videos) + len(audios) > 12:
        return "total reference count exceeds 12"
    if videos and len(references) > CAUSYN_REFERENCE_IMAGE_MAX_WITH_VIDEO:
        return f"at most {CAUSYN_REFERENCE_IMAGE_MAX_WITH_VIDEO} reference images are allowed when reference video is present"
    if audios and not references and not videos:
        return "reference audio requires at least one reference image or video"
    return None


class RewriteUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost: float | None = Field(default=None, ge=0)


class RewriteResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str
    usage: RewriteUsage
    model: str = MODEL
    system_sha256: str
    # Soft validator findings the accepted answer still has (ref2va only); diagnostic, never public.
    soft_violations: tuple[str, ...] = ()


class _Message(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _Message
    finish_reason: Literal["stop"]

    @field_validator("finish_reason", mode="before")
    @classmethod
    def reject_interruption(cls, value: JsonValue) -> JsonValue:
        if value == "error":
            raise RewriteError("H3 prompt rewrite provider interrupted generation", retryable=True)
        return value


class _Completion(BaseModel):
    model: str
    choices: tuple[_Choice, ...] = Field(min_length=1, max_length=1)
    usage: RewriteUsage

    @field_validator("model")
    @classmethod
    def allow_listed_model(cls, value: str, info: ValidationInfo) -> str:
        requested = (info.context or {}).get("requested")
        base = value.split(":", 1)[0] if requested and value.startswith(requested + ":") else value
        if base not in REF2VA_MODEL_CAPS or (requested is not None and base != requested):
            raise ValueError("model is not allow-listed")
        return value


@lru_cache(maxsize=1)
def system_prompt() -> str:
    return files("litellm.llms.causyn").joinpath("prompts/h3-system.txt").read_text(encoding="utf-8")


_MEDIA_LABEL = re.compile(r"<(Picture|Video|Audio) (\d+)>")
_LINE_LABEL = re.compile(r"^<(?:Subject|Picture|Video|Audio) \d+>")
_TIMESTAMP = re.compile(r"\bAt (\d{2}):(\d{2})\.(\d{3})\b")
_SHOT_START = re.compile(r"\[Shot (\d+)\]\s*[Aa]t (\d{2}):(\d{2})(?:\.(\d{3}))?(?!\d)")
_V_LABELS = "The rewritten H3 prompt must use exactly the provided media labels"
_V_DEFINE = "The rewritten H3 prompt must define every provided media label in subject_definitions"
_V_SHOTS = "The rewritten H3 prompt has invalid shot numbering"
_V_TIMES = "The rewritten H3 prompt has invalid shot timestamps"
_V_LENGTH = "The rewritten H3 prompt must contain 1 to 7000 characters"
_V_WORDS = "The rewritten H3 prompt detailed_description must have 200 to 750 words"
_V_RETENTION = "The rewritten H3 prompt retention_analysis lines must start with a defined label"
_V_REPLACE = (
    "The rewritten H3 prompt replacement subject must take its appearance from the reference image, "
    "not the source performer"
)
_CLOTHING = r"(?:attire|clothing|clothes|outfit|wardrobe|costume|hoodie|sweatshirt|jacket)"
# Every alternative is anchored on a clothing word (or 衣/服/装 after 原视频/源视频).
_SOURCE_ATTIRE = re.compile(
    rf"(?:original|source)(?: video)?(?:'s|’s)? {_CLOTHING}|"
    rf"to match the (?:original|source)(?: video)?(?:'s|’s)? {_CLOTHING}|"
    rf"same {_CLOTHING} as the (?:original|source)|"
    r"(?:原视频|源视频|原片)[^。，,.]{0,6}[衣服装]",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:not|never|without|unlike|instead of|rather than)\b|n't|不|而非", re.IGNORECASE)
_KEEP_SOURCE_CLOTHING = re.compile(
    rf"\b(?:keep|retain|preserve|maintain)\b[^.。]{{0,40}}?\b(?:original|source)\b[^.。]{{0,25}}?\b{_CLOTHING}\b|"
    rf"\b(?:original|source)\b[^.。]{{0,25}}?\b{_CLOTHING}\b[^.。]{{0,25}}?\b(?:kept|retained|preserved|unchanged)\b|"
    r"(?:保持|保留|不变)[^。，,.]{0,6}[衣服装穿]|[衣服装穿][^。，,.]{0,6}(?:保持|保留|不变)",
    re.IGNORECASE,
)
_V_CUTS = "The rewritten H3 prompt does not mirror the source video's shot cuts"


def _provided_labels(spec: ContextIRRequest) -> tuple[str, ...]:
    return tuple(
        f"<{type(item).__name__.removesuffix('Item').replace('Image', 'Picture')} "
        f"{sum(type(previous) is type(item) for previous in spec.ordered_media[:position])}>"
        for position, item in enumerate(spec.ordered_media, 1)
    )


def _seconds(minutes: str, seconds: str, millis: str | None) -> float:
    return int(minutes) * 60 + int(seconds) + int(millis or 0) / 1000


def _section(prompt: str, fields: tuple[str, ...], name: str) -> str:
    start = prompt.find(name + ":") + len(name) + 1
    later = [position for field in fields if (position := prompt.find(field + ":")) > start]
    return prompt[start : min(later) if later else len(prompt)]


def ref2va_violations(
    prompt: str, spec: ContextIRRequest, facts: RefFacts | None
) -> tuple[list[str], list[str]]:
    """(hard, soft) findings beyond the section layout; each starts with a fixed phrase, details follow a colon.

    Hard: unknown labels, shot numbering, timestamps, source-cut alignment (they break generation).
    Soft: length, label definitions, retention completeness and form (they only lower prompt quality).
    """
    hard: list[str] = []
    soft: list[str] = []
    sections = {name: _section(prompt, REFERENCE_FIELDS, name) for name in REFERENCE_FIELDS}
    provided = frozenset(_provided_labels(spec))
    used = frozenset(f"<{kind} {number}>" for kind, number in _MEDIA_LABEL.findall(prompt))
    if used - provided:
        hard.append(f"{_V_LABELS}: {', '.join(sorted(used - provided))}")
    undefined = sorted(label for label in provided if label not in sections["subject_definitions"])
    if undefined:
        soft.append(f"{_V_DEFINE}: {', '.join(undefined)}")
    description = sections["detailed_description"]
    shots = tuple(int(match.group(1)) for match in re.finditer(r"\[Shot (\d+)\]", description))
    if not shots or shots[0] != 1 or tuple(dict.fromkeys(shots)) != tuple(range(1, max(shots) + 1)):
        hard.append(f"{_V_SHOTS}: number [Shot N] sequentially from 1 in detailed_description")
    times = tuple(_seconds(*match.groups()) for match in _TIMESTAMP.finditer(description))
    if any(later <= earlier for earlier, later in zip(times, times[1:])) or any(t >= spec.duration for t in times):
        # Without a reference video there are no source timings to protect, so this is only a quality finding.
        (hard if any(isinstance(item, VideoItem) for item in spec.content) else soft).append(f"{_V_TIMES}: 'At MM:SS.mmm' must strictly increase and stay below {spec.duration}.000")
    words = len(description.split())
    if not DESCRIPTION_WORDS[0] <= words <= DESCRIPTION_WORDS[1]:
        soft.append(f"{_V_WORDS}: it has {words}, aim for 350-500")
    subjects = frozenset(
        match.group(0)
        for line in sections["subject_definitions"].splitlines()
        if (match := re.match(r"^<Subject \d+>", line))
    )
    defined = provided | frozenset(
        match.group(0) for line in sections["subject_definitions"].splitlines() if (match := _LINE_LABEL.match(line))
    )
    retention = sections["retention_analysis"]
    bad: list[str] = []
    line_labels: set[str] = set()
    for line in (raw.strip() for raw in retention.splitlines()):
        if not line:
            continue
        match = _LINE_LABEL.match(line)
        if match and match.group(0) in defined:
            line_labels.add(match.group(0))
        elif match or not line_labels:  # an undefined label, or text before any label line; wrapped text is fine
            bad.append(line[:40])
    unmentioned = sorted([label for label in subjects if label not in line_labels] + [
        label for label in provided if label not in retention
    ])
    if bad:
        soft.append(f"{_V_RETENTION}: {bad[0]!r}")
    elif unmentioned:
        soft.append(f"{_V_RETENTION}: add a retention line for {', '.join(unmentioned)}")
    hard.extend(_cut_violations(spec, facts, sections, description))
    soft.extend(_replacement_violations(spec, sections))
    return hard, soft


def _replacement_subject_text(subject_definitions: str) -> str:
    """Heuristic scope: the first `<Subject N>` definition (with its wrapped lines) that is derived from a `<Picture N>`.

    The replacement is the subject built from the reference image; environment and other subjects are not checked.
    """
    block: list[str] | None = None
    for line in subject_definitions.splitlines():
        if re.match(r"^<Subject \d+>", line):
            if block and re.search(r"<Picture \d+>", " ".join(block)):
                break
            block = [line]
        elif block is not None and not line.lstrip().startswith("<"):
            block.append(line)
    return " ".join(block) if block and re.search(r"<Picture \d+>", " ".join(block)) else ""


def _source_attire_mentioned(text: str) -> bool:
    return any(
        not _NEGATION.search(text[max(0, match.start() - 40) : match.start()]) for match in _SOURCE_ATTIRE.finditer(text)
    )


def _replacement_violations(spec: ContextIRRequest, sections: dict[str, str]) -> list[str]:
    """Conservative phrase checks for a person swap in a video edit; no semantics, soft only.

    Skipped when the request is not a video edit with a replacement, or when the user explicitly asks to keep the
    source clothing (English or Chinese).
    """
    if not (
        any(isinstance(item, VideoItem) for item in spec.content)
        and any(isinstance(item, ImageItem) for item in spec.content)
    ):
        return []
    summary = sections["summary"].lower()
    if "video editing" not in summary or not re.search(r"replac|swap", summary):
        return []
    if _KEEP_SOURCE_CLOTHING.search(spec.prompt):
        return []
    weak = any(
        re.match(r"<Video \d+>", line.strip()) and "weak_reference" in line
        for line in sections["retention_analysis"].splitlines()
    )
    attire = _source_attire_mentioned(_replacement_subject_text(sections["subject_definitions"]))
    if not weak and not attire:
        return []
    detail = "source <Video N> must be fully_preserved" if weak else "do not match the source performer's attire"
    return [f"{_V_REPLACE}: {detail}"]


def _cut_violations(
    spec: ContextIRRequest, facts: RefFacts | None, sections: dict[str, str], description: str
) -> list[str]:
    videos = [item for item in spec.content if isinstance(item, VideoItem)]
    if facts is None or len(videos) != 1 or not facts.videos or facts.videos[0] is None:
        return []
    cuts = tuple(cut for cut in (facts.videos[0].cuts or ()) if cut < spec.duration)
    if not cuts or "video editing" not in sections["summary"].lower():
        return []
    starts = tuple(
        _seconds(*match.groups()[1:]) for match in _SHOT_START.finditer(description) if int(match.group(1)) >= 2
    )
    missing = [cut for cut in cuts if not any(abs(start - cut) <= CUT_TOLERANCE_S for start in starts)]
    if not missing:
        return []
    cut_list = ", ".join(fmt_ts(cut) for cut in missing)
    return [f"{_V_CUTS}: start a [Shot k] with 'At MM:SS.mmm' at each source cut, missing {cut_list}"]


_FROM_SHOT = re.compile(r"\(from (?:\[Shot (\d+)\]|Shot (\d+))\)")
_V_ALIGN = "The rewritten H3 prompt has invalid last-frame alignment"


def _base_violations(prompt: str, spec: ContextIRRequest, description: str) -> list[str]:
    """Hard structural findings for t2va/i2va/fl2va/l2va: later-shot timestamps and last-frame alignment."""
    found: list[str] = []
    numbers = tuple(int(m.group(1)) for m in re.finditer(r"\[Shot (\d+)\]", description))
    starts: dict[int, float] = {}
    for match in _SHOT_START.finditer(description):
        starts.setdefault(int(match.group(1)), _seconds(*match.groups()[1:]))
    later = sorted({n for n in numbers if n >= 2})
    times = [starts.get(n) for n in later]
    if any(t is None for t in times):
        found.append(f"{_V_TIMES}: every later [Shot k] needs 'At MM:SS.mmm', missing for Shot {later[times.index(None)]}")
    elif any(b <= a for a, b in zip(times, times[1:])) or any(not 0 < t < spec.duration for t in times):
        found.append(f"{_V_TIMES}: 'At MM:SS.mmm' must strictly increase and stay between 0 and {spec.duration}.000")
    if spec.mode in {"fl2va", "l2va"} and numbers:
        aligned = _FROM_SHOT.findall(prompt.splitlines()[0])
        if aligned and int(aligned[-1][0] or aligned[-1][1]) != max(numbers):
            found.append(f"{_V_ALIGN}: the last frame must come from the final shot, Shot {max(numbers)}")
    return found


_CURLY_QUOTED = re.compile(r"\u201c([^\u201d]{2,300})\u201d|\u300c([^\u300d]{2,300})\u300d|\u300e([^\u300f]{2,300})\u300f")
# A straight pair counts only when opened/closed at a boundary (not inside a word: 12" pizza and a 14" pan).
_STRAIGHT_QUOTED = re.compile(r'(?<![A-Za-z0-9_"])"([^"]{2,300})"(?![A-Za-z0-9_])')
_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.DOTALL)
_DIALOGUE_MARKERS = re.compile(r"</?(?:cutoff|scenetrans)\s*/?>")
_LANG_TAG = re.compile(r"^\s*\[[^\]\n]+\]")
_TRAILING_PUNCT = ".!?\u3002\uff01\uff1f\u2026,\uff0c "
_QUOTE_MAP = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u3002": "."}
)
_SPEECH_MAX_WORDS = 12
_V_QUOTED = "The rewritten H3 prompt must keep the user's quoted text verbatim: "
_V_SPEECH = "The rewritten H3 prompt must not add dialogue the user did not write: "
_V_VISIBLE = "The rewritten H3 prompt must not add visible text the user did not write: "


def _fold(text: str) -> str:
    """Comparison form: NFKC (full/half-width, ellipsis), one quote style, casefolded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP).casefold().split())


def _norm(text: str) -> str:
    """`_fold` without trailing punctuation marks."""
    return _fold(text).rstrip(_TRAILING_PUNCT)


def _quoted_spans(text: str, newlines: bool = True) -> list[str]:
    """Raw spans inside curly / CJK quotes and boundary-delimited straight quotes (none when `"` is unbalanced)."""
    spans = [next(g for g in m.groups() if g is not None) for m in _CURLY_QUOTED.finditer(text)]
    if text.count('"') % 2 == 0:
        spans += [m.group(1) for m in _STRAIGHT_QUOTED.finditer(text)]
    return [span for span in spans if newlines or "\n" not in span]


def literal_violations(prompt: str, spec: ContextIRRequest) -> list[str]:
    """Soft, deterministic fidelity findings: quoted lines kept, no invented speech or on-screen text.

    Not applied to requests with a reference video (its spoken content is unknown here).
    """
    if any(isinstance(item, VideoItem) for item in spec.content):
        return []
    found: list[str] = []
    user = _fold(spec.prompt)
    rewrite = _fold(prompt)
    quotes = [(raw, _norm(raw)) for raw in _quoted_spans(spec.prompt)]
    missing = [raw.strip() for raw, norm in dict(quotes).items() if norm and norm not in rewrite]
    if missing:
        found.append(_V_QUOTED + "; ".join(missing))
    if not any(isinstance(item, AudioItem) for item in spec.content):
        spans = [norm for _, norm in quotes if norm]
        invented: list[str] = []
        for match in _DIALOGUE.finditer(prompt):
            for piece in _DIALOGUE_MARKERS.split(match.group(1)):
                shown = _LANG_TAG.sub("", piece, count=1).strip()
                text = _norm(shown)
                if not text:
                    continue
                if spans:  # the user marked their lines: a short line from the prompt, or part of a marked line
                    ok = any(text in span for span in spans) or (
                        len(text.split()) <= _SPEECH_MAX_WORDS and text in user
                    )
                else:
                    ok = text in user
                if not ok:
                    invented.append(shown[:120])
        if invented:
            found.append(_V_SPEECH + "; ".join(dict.fromkeys(invented)))
    if not spec.ordered_media:
        spans = [span.strip() for span in _quoted_spans(_DIALOGUE.sub(" ", prompt), newlines=False)]
        extra = [t for t in dict.fromkeys(spans) if _norm(t) and _norm(t) not in user]
        if extra:
            found.append(_V_VISIBLE + "; ".join(extra))
    return found


def validate_prompt(
    prompt: str, spec: ContextIRRequest, facts: RefFacts | None = None, soft: list[str] | None = None
) -> None:
    """Raise on any violation. With `soft` given, ref2va soft-only findings are appended to it instead of raising."""
    if not prompt or len(prompt) > 7000:
        # "it has N characters" is added only for the too-long case (the repair message then asks to shorten).
        detail = (f"{_V_LENGTH}: it has {len(prompt)} characters",) if prompt else ()
        raise RewriteError(_V_LENGTH, violations=detail)
    fields = REFERENCE_FIELDS if spec.mode == "ref2va" else BASE_FIELDS
    positions = tuple(prompt.find(field + ":") for field in fields)
    if any(position < 0 for position in positions) or positions != tuple(sorted(positions)):
        raise RewriteError("The rewritten H3 prompt has invalid fields")
    if any(prompt.count(field + ":") != 1 for field in fields):
        raise RewriteError("The rewritten H3 prompt has duplicate fields")
    if spec.mode in {"t2va", "ref2va"} and not prompt.startswith(fields[0] + ":"):
        raise RewriteError("The rewritten H3 prompt has an invalid prefix")
    if spec.mode == "i2va" and not prompt.startswith(
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    ):
        raise RewriteError("The rewritten H3 prompt has invalid first-frame alignment")
    if spec.mode in {"fl2va", "l2va"} and (
        not prompt.startswith("How the reference pictures align with the target video")
        or f"aligns with the {spec.duration:.2f}-second mark" not in prompt.splitlines()[0]
    ):
        raise RewriteError("The rewritten H3 prompt has invalid last-frame alignment")
    if spec.mode != "ref2va":
        description = prompt[positions[0] : positions[1]]
        shots = tuple(int(match.group(1)) for match in re.finditer(r"\[Shot (\d+)\]", description))
        if not shots or shots[0] != 1 or tuple(dict.fromkeys(shots)) != tuple(range(1, max(shots) + 1)):
            raise RewriteError("The rewritten H3 prompt has invalid shot numbering")
    if prompt.count("<d>") != prompt.count("</d>") or re.search(r"<d>(?!\[[^\]\n]+\])", prompt):
        raise RewriteError("The rewritten H3 prompt has invalid dialogue tags")
    if spec.mode != "ref2va":
        base_hard = _base_violations(prompt, spec, prompt[positions[0] : positions[1]])
        if base_hard:
            failure = RewriteError(base_hard[0].split(": ", 1)[0], violations=tuple(base_hard))
            failure.new_checks_only = True
            raise failure
    if spec.mode == "ref2va":
        hard, found_soft = ref2va_violations(prompt, spec, facts)
        if hard:
            raise RewriteError(hard[0].split(": ", 1)[0], violations=tuple(hard + found_soft))
        if soft is not None:
            soft.extend(found_soft)
        elif found_soft:
            raise RewriteError(found_soft[0].split(": ", 1)[0], violations=tuple(found_soft))


REPLACEMENT_APPEARANCE_RULE = (
    "When the request replaces a person or subject in a source video with a subject from a reference image, the "
    "replacement's entire visible appearance — face, hair, body, clothing, footwear and accessories — comes from the "
    "reference image. Keep the source performer's clothing only if the user explicitly asks for it. Never describe the "
    "replaced performer's clothing, colours or accessories on the new subject. The source video supplies only motion, "
    "timing, camera and environment. In a [video editing] task the source `<Video N>` keeps its structural role: mark "
    "it fully_preserved (or partially_preserved when the user changes the structure), never weak_reference."
)


def ref2va_addendum(has_video: bool, has_audio: bool, has_image: bool = False) -> str:
    """Perception rules appended to the Ref2VA system text; the video-only parts apply only with a reference video."""
    parts = []
    if has_video or has_audio:
        parts.append(
            "The user message for this Ref2VA request carries measured media facts (durations, detected shot cuts, "
            "timestamped keyframes, audio loudness). Treat them as ground truth."
        )
    if has_video:
        parts.append(
            "When the summary is a video edit of a single source video, mirror its shots: write one `[Shot N]` per "
            "detected source shot, starting at the detected cut times written as `At MM:SS.mmm`. "
            "Describe each shot's composition, camera and timed actions taken from the keyframes. "
            "Define the environment as its own `<Subject N>`. "
            "`detailed_description` should be 350–500 words."
        )
        if has_image:
            parts.append(REPLACEMENT_APPEARANCE_RULE)
    else:
        parts.append("`detailed_description` uses `[Shot N]` markers numbered from 1 and has 200–750 words.")
    parts.append(
        "Write retention lines in the form `<Label> (appears in [Shot ...]): marker - ...`. "
        "Every provided media label appears in `subject_definitions`, and no other media labels are used."
    )
    return "\n" + " ".join(parts)


TRUNCATION_REPAIR_MESSAGE = (
    "Your previous answer was cut off before it finished. Rewrite the complete answer from the beginning, "
    "concisely: keep detailed_description within 350–500 words and do not repeat sentences."
)


TRUNCATION_REPAIR_MESSAGE_BASE = (
    "Your previous answer was cut off before it finished. Rewrite the complete answer from the beginning, "
    "concisely, and do not repeat sentences."
)


KEEP_DETAIL_SENTENCE = (
    "Keep everything that was already correct, including the level of detail and length of the descriptive section; "
    "change only what is needed to fix the listed points."
)


SHORTEN_SENTENCE = (
    "Shorten the descriptive sections so the whole prompt stays well under 7000 characters; "
    "keep every required section, label and spoken line."
)


def _too_long(violation: str) -> bool:
    """The over-7000-characters violation, or the word-count violation in its over-maximum form."""
    if violation.startswith(_V_LENGTH):
        return ": it has " in violation
    if violation.startswith(_V_WORDS):
        found = re.search(r"it has (\d+)", violation)
        return found is not None and int(found.group(1)) > DESCRIPTION_WORDS[1]
    return False


def _repair_message(violations: tuple[str, ...], labels: tuple[str, ...], base: bool = False) -> str:
    listed = "\n".join(f"- {violation}" for violation in violations)
    closing = SHORTEN_SENTENCE if any(_too_long(v) for v in violations) else KEEP_DETAIL_SENTENCE
    if base:
        return (
            "Your previous answer has these problems:\n"
            f"{listed}\n"
            "Fix exactly these items and keep everything else unchanged. "
            "Return the complete prompt again in the same three-field format.\n"
            f"{closing}"
        )
    return (
        "Your previous answer broke these output rules:\n"
        f"{listed}\n"
        f"The provided media labels are exactly: {', '.join(labels) or 'none'}.\n"
        "Write the complete prompt again in the same six-section format and fix every point.\n"
        f"{closing}"
    )


CRITIC_INSTRUCTION = """You audit a rewritten video-generation prompt against the user's original request.
List ONLY concrete fidelity defects of the rewrite, each one short:
- a subject, object, attribute (color, material, count, size, clothing), spatial relation, viewpoint, camera instruction
  (movement type and direction), style, action or setting that the user explicitly requested but the rewrite dropped,
  changed or contradicted;
- a spoken line or visible text that the user wrote but the rewrite changed, translated, shortened or omitted;
- dialogue, lyrics or visible on-screen text that the rewrite invented although the user did not ask for it;
- a new main subject or story event that replaces or competes with what the user asked for.
Added scenic detail, lighting, sound design, camera framing or motion that is compatible with the request is NOT a defect.
Attached images are authoritative for appearance: details taken from them are not defects.
Return ONLY JSON: {"defects": ["...", ...]} (empty list when the rewrite is faithful)."""
CRITIC_AUDIO_NOTE = (
    "Reference audio is attached to this request but not shown to you: spoken content may come from it, "
    "so do not flag dialogue for that reason."
)


def critic_enabled() -> bool:
    return os.getenv(CRITIC_ENV, "").strip() != "0"


def critic_model() -> str | None:
    """Allow-listed critic model; an unknown value skips the critic (it is optional) with a single warning."""
    chosen = os.getenv(CRITIC_MODEL_ENV, "").strip() or REF2VA_DEFAULT_MODEL
    if chosen in REF2VA_MODEL_CAPS:
        return chosen
    if CRITIC_MODEL_ENV + chosen not in _warned_models:
        _warned_models.add(CRITIC_MODEL_ENV + chosen)
        _log.warning("%s is not an allow-listed model, the fidelity critic is skipped: %r", CRITIC_MODEL_ENV, chosen[:80])
    return None


def plan_enabled() -> bool:
    return os.getenv(ref2va_plan.PLAN_ENV, "").strip() != "0"


def plan_model() -> str | None:
    """Allow-listed model for the observe and plan calls; an unknown value skips the plan stage with a single warning."""
    chosen = os.getenv(ref2va_plan.PLAN_MODEL_ENV, "").strip() or REF2VA_DEFAULT_MODEL
    if chosen in REF2VA_MODEL_CAPS:
        return chosen
    if ref2va_plan.PLAN_MODEL_ENV + chosen not in _warned_models:
        _warned_models.add(ref2va_plan.PLAN_MODEL_ENV + chosen)
        _log.warning("%s is not an allow-listed model, the Ref2VA plan stage is skipped: %r", ref2va_plan.PLAN_MODEL_ENV, chosen[:80])
    return None


def parse_defects(text: str) -> tuple[str, ...] | None:
    """Defects from the first JSON object in a critic reply; None when the reply is unusable."""
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        defects = value.get("defects") if isinstance(value, dict) else None
        if isinstance(defects, list) and all(isinstance(d, str) for d in defects):
            cleaned = (" ".join(d.split())[:CRITIC_DEFECT_CHARS] for d in defects)
            return tuple(d for d in cleaned if d)[:CRITIC_MAX_DEFECTS]
        return None
    return None


def _remaining(started: float) -> float:
    """Seconds left for provider calls: the attempt budget since `started`, capped by the attempt deadline."""
    remaining = ATTEMPT_BUDGET_S - (time.monotonic() - started)
    deadline = ATTEMPT_DEADLINE.get()
    return remaining if deadline is None else min(remaining, deadline - time.monotonic())


def _sum_usage(first: RewriteUsage, second: RewriteUsage) -> RewriteUsage:
    cost = None if first.cost is None and second.cost is None else (first.cost or 0.0) + (second.cost or 0.0)
    return RewriteUsage(
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        completion_tokens=first.completion_tokens + second.completion_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
        cost=cost,
    )


def _reply_usage(response: httpx.Response) -> RewriteUsage:
    try:
        return RewriteUsage.model_validate(response.json().get("usage") or {})
    except Exception:  # noqa: BLE001
        return RewriteUsage()


def _response_detail(response: httpx.Response) -> str:
    """Short diagnostic for an unusable provider reply: finish_reason, returned model id, empty content. No body text."""
    try:
        body = response.json()
        choice = body["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        return (
            f"finish_reason={choice.get('finish_reason')} model={body.get('model')} "
            f"content_empty={not (isinstance(content, str) and content.strip())}"
        )
    except Exception:  # noqa: BLE001
        return f"unreadable reply, {len(response.content)} bytes"


class H3PromptRewriter:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, JsonValue]],
        budget: float,
        ref2va: bool = False,
        settings: tuple[dict[str, JsonValue], int] | None = None,
    ) -> _Completion:
        # Reasoning and the completion budget come from the per-model table; `settings` overrides it (critic).
        reasoning, max_tokens = settings or REF2VA_MODEL_REASONING[model]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0,
            "reasoning": reasoning,
            "provider": {"allow_fallbacks": False, "require_parameters": True},
        }
        try:
            async with asyncio.timeout(budget):
                response = await self.client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=min(85, budget),
                    follow_redirects=False,
                )
        except (httpx.TransportError, TimeoutError) as exc:
            raise RewriteError(
                "H3 prompt rewrite provider interrupted", retryable=True, detail=type(exc).__name__
            ) from exc
        if response.status_code != 200:
            raise RewriteError(
                "H3 prompt rewrite provider unavailable",
                429 if response.status_code == 429 else 502,
                retryable=response.status_code in RETRYABLE_STATUS_CODES,
                retry_after=response.headers.get("retry-after"),
                upstream_status=response.status_code,
                detail=provider_error_detail(response),
            )
        try:
            return _Completion.model_validate(response.json(), context={"requested": model})
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RewriteError(
                "H3 prompt rewrite provider returned an incomplete response",
                retryable=True,
                detail=_response_detail(response),
            ) from exc
        except ValidationError as exc:
            # A wrong/unlisted model id is a provider routing glitch (retry); a truncated or malformed answer
            # (e.g. finish_reason=length) is permanent, as before.
            routing = any(error["loc"][:1] == ("model",) for error in exc.errors())
            truncated = any(error["loc"][:3] == ("choices", 0, "finish_reason") for error in exc.errors())
            failure = (TruncatedRewriteError if truncated else RewriteError)(
                "H3 prompt rewrite provider returned an invalid response",
                502,
                retryable=routing,
                detail=_response_detail(response),
            )
            if truncated:
                failure.usage = _reply_usage(response)  # type: ignore[attr-defined]
            raise failure from exc

    async def fidelity_defects(
        self, spec: ContextIRRequest, prompt: str, budget: float
    ) -> tuple[tuple[str, ...], RewriteUsage]:
        """Defects a cheap critic call finds in `prompt`, plus the call's usage. Strictly fail-open: any error -> none.

        `spec` must be the prepared request, so the critic sees the same fetched data-URL images as the rewrite call."""
        usage = RewriteUsage()
        try:
            model = critic_model()
            if model is None:
                return (), usage
            parts: list[dict[str, JsonValue]] = [{"type": "text", "text": CRITIC_INSTRUCTION}]
            for item in spec.ordered_media:
                if isinstance(item, ImageItem):
                    parts.append({"type": "text", "text": f"Attached image ({item.role}):"})
                    parts.append({"type": "image_url", "image_url": {"url": item.image_url.url}})
            if any(isinstance(item, AudioItem) for item in spec.content):
                parts.append({"type": "text", "text": CRITIC_AUDIO_NOTE})
            parts.append({"type": "text", "text": f"USER REQUEST:\n{spec.prompt}\n\nREWRITE:\n{prompt}"})
            reasoning = REF2VA_MODEL_REASONING[model][0]
            settings = (reasoning, 6000 if reasoning.get("enabled") else 1500)
            try:
                completed = await self._complete(
                    model, [{"role": "user", "content": parts}], min(CRITIC_TIMEOUT_S, budget), settings=settings
                )
            except TruncatedRewriteError as cut:
                _log.info("H3 rewrite fidelity critic skipped: reply cut off")
                return (), cut.usage
            except RewriteError as failure:
                _log.info("H3 rewrite fidelity critic skipped: %s", failure.describe_upstream() or "provider error")
                return (), usage
            usage = completed.usage
            defects = parse_defects(completed.choices[0].message.content)
            if defects is None:
                _log.info("H3 rewrite fidelity critic skipped: unusable reply")
                return (), usage
            return defects, usage
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  # the critic is optional: never fail the rewrite
            _log.info("H3 rewrite fidelity critic skipped: %s", type(exc).__name__)
            return (), usage

    async def ref2va_plan_notes(
        self, spec: ContextIRRequest, key: str, started: float
    ) -> tuple[str | None, RewriteUsage]:
        """Directing-plan notes for an image-only Ref2VA request plus the stage's usage. Strictly fail-open: any error -> none.

        `spec` must be the prepared request (images as data URLs). Observe each picture, then plan the whole shot."""
        usage = RewriteUsage()
        try:
            if not plan_enabled():
                return None, usage
            if any(not isinstance(item, ImageItem) for item in spec.ordered_media):
                return None, usage
            model = plan_model()
            if model is None:
                return None, usage
            memo_key = key + "plan" + model
            memoized = _PLAN_NOTES.get(memo_key)
            if memoized is not None:
                _PLAN_NOTES.move_to_end(memo_key)
                return memoized
            budget = min(
                ref2va_plan.PLAN_STAGE_MAX_S,
                _remaining(started) - MIN_REPAIR_BUDGET_S - ref2va_plan.PLAN_STAGE_RESERVE_S,
            )
            if budget < ref2va_plan.PLAN_STAGE_MIN_S:
                _log.info("H3 Ref2VA plan stage skipped: no time budget")
                return None, usage
            began = time.monotonic()
            urls = [item.image_url.url for item in spec.ordered_media if isinstance(item, ImageItem)]
            reasoning = REF2VA_MODEL_REASONING[model][0]
            spent: list[RewriteUsage] = []
            gate = asyncio.Semaphore(ref2va_plan.OBSERVE_CONCURRENCY)

            async def call(messages: list[dict[str, JsonValue]], max_tokens: int) -> str:
                try:
                    completed = await self._complete(model, messages, budget, settings=(reasoning, max_tokens))
                except TruncatedRewriteError as cut:
                    spent.append(cut.usage)
                    raise
                spent.append(completed.usage)
                return completed.choices[0].message.content

            async def observe(url: str) -> str:
                async with gate:
                    return await call(
                        [
                            {"role": "system", "content": ref2va_plan.OBSERVE},
                            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
                        ],
                        ref2va_plan.OBSERVE_MAX_TOKENS,
                    )

            try:
                async with asyncio.timeout(budget):
                    async with asyncio.TaskGroup() as group:
                        tasks = [group.create_task(observe(url)) for url in urls]
                    observations = [task.result() for task in tasks]
                    content: list[dict[str, JsonValue]] = []
                    for number, (url, seen) in enumerate(zip(urls, observations), 1):
                        content += [
                            {"type": "text", "text": f"Picture {number}:"},
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": f"Observation of picture {number}:\n{seen}"},
                        ]
                    content.append({"type": "text", "text": f"USER REQUEST ({spec.duration} s):\n{spec.prompt}"})
                    raw = await call(
                        [
                            {
                                "role": "system",
                                "content": ref2va_plan.PLAN.format(duration=spec.duration, n=len(urls)),
                            },
                            {"role": "user", "content": content},
                        ],
                        ref2va_plan.PLAN_MAX_TOKENS,
                    )
                notes = ref2va_plan.plan_notes(ref2va_plan.parse_plan(raw), len(urls))
            finally:
                for part in spent:
                    usage = _sum_usage(usage, part)
                _log.info(
                    "H3 Ref2VA plan stage took %.1fs, cost %s", time.monotonic() - began, usage.cost
                )
            _remember(_PLAN_NOTES, memo_key, (notes, usage))
            return notes, usage
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  # the plan is optional: never fail the rewrite
            _log.info("H3 Ref2VA plan stage skipped: %s", type(exc).__name__)
            return None, usage

    async def rewrite(self, spec: ContextIRRequest) -> RewriteResult:
        if not self.api_key:
            raise RewriteError("H3 prompt rewrite is not configured", 503)
        from litellm.llms.causyn import h3_media

        ref2va = spec.mode == "ref2va"
        model = ref2va_model() if ref2va else base_model()  # a misconfigured model fails before any media is fetched
        prepared, raw = await h3_media.prepare_media_with_raw(self.client, spec)
        # One key per request (hashed off the event loop) serves both the facts memo and the first-answer cache.
        key = await asyncio.to_thread(media_key, spec, raw.videos, raw.audios)
        facts = await perceive_media(prepared, raw, key) if ref2va else None
        # The provider clock starts once fetch and perception are done; the attempt deadline still bounds it.
        started = time.monotonic()
        system = system_prompt() + (
            "\nFor the current Ref2VA request, the official six-section reference format replaces the application's three-field format. "
            "Use subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. "
            "When continuation is requested, start from the final visible state of the source clip and continue its motion. "
            "Do not restart a subject entrance, reset positions, or loop the source unless explicitly requested."
            + ref2va_addendum(
                any(isinstance(item, VideoItem) for item in spec.content),
                any(isinstance(item, AudioItem) for item in spec.content),
                any(isinstance(item, ImageItem) for item in spec.content),
            )
            if ref2va
            else ""
        )
        user = prepared.user_content(facts, model, send_video_enabled()) if ref2va else prepared.user_content()
        plan_usage = RewriteUsage()
        if ref2va and not any(isinstance(item, (VideoItem, AudioItem)) for item in spec.content):
            notes, plan_usage = await self.ref2va_plan_notes(prepared, key, started)
            if notes is not None:
                user.append({"type": "text", "text": "\n\n" + notes})
        messages: list[dict[str, JsonValue]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # A retry after a retryable failure of the repair call reuses the first answer instead of paying for it again.
        answer_key = key + model
        has_video = any(isinstance(item, VideoItem) for item in spec.content)
        cached_answer = _FIRST_ANSWERS.get(answer_key)
        truncated = False
        if cached_answer is not None:
            prompt, usage = cached_answer
        else:
            try:
                completed = await self._complete(model, messages, max(5.0, _remaining(started)), ref2va)
                prompt = completed.choices[0].message.content.strip()
                usage = completed.usage
            except TruncatedRewriteError as cut:
                prompt, usage, truncated = "", cut.usage, True
            usage = _sum_usage(usage, plan_usage)  # a cached first answer already carries the plan usage
        keep_first_answer = False
        soft: list[str] = []
        hard_failure: RewriteError | None = None
        if not truncated:
            try:
                validate_prompt(prompt, spec, facts, soft=soft if ref2va else None)
            except RewriteError as failure:
                hard_failure = failure
        # Deterministic fidelity findings (none with a reference video). The critic runs for t2va/i2va/fl2va/l2va only:
        # evaluation showed it does not improve any Ref2VA request.
        literal = [] if truncated or has_video else literal_violations(prompt, spec)
        if hard_failure is None and not truncated and not ref2va and critic_enabled():
            budget = _remaining(started)
            if budget >= MIN_REPAIR_BUDGET_S:
                defects, critic_usage = await self.fidelity_defects(prepared, prompt, budget)
                usage = _sum_usage(usage, critic_usage)
                soft.extend(f"Fidelity: {defect}" for defect in defects)
        soft.extend(literal)

        def result(text: str, used: RewriteUsage, leftover: list[str]) -> RewriteResult:
            if leftover:
                _log.info("H3 rewrite accepted with soft violations: %s", [v.split(": ", 1)[0][-60:] for v in leftover])
            return RewriteResult(
                prompt=text,
                usage=used,
                model=model,
                system_sha256=hashlib.sha256(system.encode()).hexdigest(),
                soft_violations=tuple(leftover),
            )

        try:
            if truncated:
                # The cut-off text is not echoed back (it may be huge or looping): ask again, concisely.
                remaining = _remaining(started)
                if remaining < MIN_REPAIR_BUDGET_S:
                    raise TruncatedRewriteError(
                        "H3 prompt rewrite provider returned an invalid response", 502, retryable=False
                    )
                messages.append(
                    {"role": "user", "content": TRUNCATION_REPAIR_MESSAGE if ref2va else TRUNCATION_REPAIR_MESSAGE_BASE}
                )
                again = await self._complete(model, messages, remaining, ref2va)  # a second truncation raises
                again_soft: list[str] = []
                again_prompt = again.choices[0].message.content.strip()
                validate_prompt(again_prompt, spec, facts, soft=again_soft)
                again_soft.extend([] if has_video else literal_violations(again_prompt, spec))
                return result(again_prompt, _sum_usage(usage, again.usage), again_soft)
            if hard_failure is None and not soft:
                return result(prompt, usage, [])
            remaining = _remaining(started)
            if remaining < MIN_REPAIR_BUDGET_S:
                if hard_failure is not None:
                    raise hard_failure
                return result(prompt, usage, soft)  # soft-only first answer is acceptable as is
            listed = (
                (*(hard_failure.violations or (str(hard_failure),)), *literal) if hard_failure is not None else tuple(soft)
            )
            _remember(_FIRST_ANSWERS, answer_key, (prompt, usage))
            messages += [
                {"role": "assistant", "content": prompt},
                {"role": "user", "content": _repair_message(listed, _provided_labels(spec), base=not ref2va)},
            ]
            try:
                repaired = await self._complete(model, messages, remaining, ref2va)
            except RewriteError as call_failure:
                if hard_failure is None:  # a usable first answer beats a failed improvement
                    return result(prompt, usage, soft)
                keep_first_answer = call_failure.retryable  # a retried attempt reuses the first answer
                raise
            repaired_prompt = repaired.choices[0].message.content.strip()
            total = _sum_usage(usage, repaired.usage)
            repaired_soft: list[str] = []
            try:
                validate_prompt(repaired_prompt, spec, facts, soft=repaired_soft)
                repaired_soft.extend([] if has_video else literal_violations(repaired_prompt, spec))
            except RewriteError as repair_failure:
                if hard_failure is None:  # the repair made it worse: keep the first answer
                    return result(prompt, total, soft)
                if repair_failure.new_checks_only and hard_failure.new_checks_only:
                    # Only the structural checks added later still fail and the first answer passed every older
                    # check: a usable first answer beats a permanent failure.
                    return result(prompt, total, [*hard_failure.violations, *literal])
                raise
            # Never worse: critic defects are not re-evaluated on the repaired answer, so compare deterministic findings only.
            if hard_failure is None and len(repaired_soft) > sum(not v.startswith("Fidelity: ") for v in soft):
                return result(prompt, total, soft)
            return result(repaired_prompt, total, repaired_soft)
        finally:
            if not keep_first_answer:
                _FIRST_ANSWERS.pop(answer_key, None)


async def _rewrite_single_shot(spec: ContextIRRequest) -> RewriteResult:
    async with httpx.AsyncClient(trust_env=False) as client:
        return await H3PromptRewriter(client, os.getenv("OPENROUTER_API_KEY", "")).rewrite(spec)


async def rewrite_prompt(spec: ContextIRRequest) -> RewriteResult:
    """纯图片 Ref2VA 仅在显式开启开关且服务已配置时走多步 Context IR，其余一律单次改写。

    开关 CAUSYN_REF2VA_CONTEXT_IR_SERVICE 默认关闭：服务优化完成前，生产的所有
    Ref2VA（含纯图片）与 t2va/fl2va 一样走单次改写，即使部署里仍配着服务地址。
    """
    from litellm.llms.causyn import context_ir_client as service

    base_url = service.service_base_url()
    if not service.service_enabled() or base_url is None or not service.should_use_service(spec):
        return await _rewrite_single_shot(spec)
    from litellm.llms.causyn.task_telemetry import current_task

    kwargs = {}
    deadline = ATTEMPT_DEADLINE.get()
    if deadline is not None:
        kwargs["budget_s"] = service.poll_budget(deadline - time.monotonic())
    async with httpx.AsyncClient(trust_env=False) as client:
        return await service.rewrite_via_service(
            spec,
            base_url=base_url,
            api_key=service.service_api_key(),
            idempotency_key=service.idempotency_key_for(spec, current_task()),
            http=client,
            **kwargs,
        )
