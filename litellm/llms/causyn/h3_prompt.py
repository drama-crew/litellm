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
from contextvars import ContextVar
from collections import OrderedDict
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from functools import lru_cache
from importlib.resources import files
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

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

MODEL = "qwen/qwen3.8-flash"
# Ref2VA rewrite model allow-list with each model's input modalities. Audio is sent only to `omni`.
REF2VA_MODEL_CAPS: dict[str, frozenset[str]] = {
    "qwen/qwen3.8-flash": frozenset({"text", "image", "video"}),
    "qwen/qwen3.8-omni-flash": frozenset({"text", "image", "audio", "video"}),
    "qwen/qwen3.8-max-0902": frozenset({"text", "image", "video"}),
}
# Per-model reasoning request and completion budget. max-0902 rejects `reasoning.enabled=false`.
REF2VA_MODEL_REASONING: dict[str, tuple[dict[str, JsonValue], int]] = {
    "qwen/qwen3.8-flash": ({"enabled": False}, 4096),
    "qwen/qwen3.8-omni-flash": ({"enabled": False}, 4096),
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
        return MODEL
    if chosen not in REF2VA_MODEL_CAPS:
        if chosen not in _warned_models:
            _warned_models.add(chosen)
            _log.warning("%s is not an allow-listed Ref2VA rewrite model: %r", REF2VA_MODEL_ENV, chosen[:80])
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
_FACTS_MEMO: OrderedDict[str, RefFacts] = OrderedDict()
_FIRST_ANSWERS: OrderedDict[str, str] = OrderedDict()


def _remember(store: OrderedDict, key: str, value: object) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > _MEMO_SIZE:
        store.popitem(last=False)


def media_key(spec: ContextIRRequest, videos: tuple[bytes, ...], audios: tuple[bytes, ...]) -> str:
    """Stable per-request key: the submitted media URLs plus the digest of every fetched video/audio byte string."""
    digest = hashlib.sha256(spec.model_dump_json().encode())
    for data in (*videos, b"|", *audios):
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


async def perceive_media(spec: ContextIRRequest, raw: object) -> RefFacts:
    """Perception for one request, memoised across durable-task retries (in-process LRU of 32).

    A retry still re-fetches the media (the bytes are not cached) but skips the up to 20 s of CPU work. Any failure
    degrades to missing facts, never to an error.
    """
    videos_raw: tuple[bytes, ...] = getattr(raw, "videos", ())
    audios_raw: tuple[bytes, ...] = getattr(raw, "audios", ())
    key = media_key(spec, videos_raw, audios_raw)
    cached = _FACTS_MEMO.get(key)
    if cached is not None:
        _FACTS_MEMO.move_to_end(key)
        return cached
    facts = await _perceive(videos_raw, audios_raw)
    _remember(_FACTS_MEMO, key, facts)
    return facts


async def _perceive(videos_raw: tuple[bytes, ...], audios_raw: tuple[bytes, ...]) -> RefFacts:

    def work() -> RefFacts:
        deadline = time.monotonic() + PERCEPTION_BUDGET_S
        budget = keyframe_budget(len(videos_raw))
        videos = tuple(analyze_video(data, max_keyframes=budget, deadline=deadline) for data in videos_raw)
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
        return RefFacts(
            videos=(None,) * len(videos_raw), audios=(None,) * len(audios_raw), audio_raw=audios_raw
        )


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
_V_WORDS = "The rewritten H3 prompt detailed_description must have 200 to 750 words"
_V_RETENTION = "The rewritten H3 prompt retention_analysis lines must start with a defined label"
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


def ref2va_violations(prompt: str, spec: ContextIRRequest, facts: RefFacts | None) -> list[str]:
    """Findings beyond the section layout; each starts with a fixed phrase, details follow a colon."""
    found: list[str] = []
    sections = {name: _section(prompt, REFERENCE_FIELDS, name) for name in REFERENCE_FIELDS}
    provided = frozenset(_provided_labels(spec))
    used = frozenset(f"<{kind} {number}>" for kind, number in _MEDIA_LABEL.findall(prompt))
    if used != provided:
        wrong = sorted(used ^ provided)
        found.append(f"{_V_LABELS}: {', '.join(wrong)}")
    undefined = sorted(label for label in provided if label not in sections["subject_definitions"])
    if undefined:
        found.append(f"{_V_DEFINE}: {', '.join(undefined)}")
    description = sections["detailed_description"]
    shots = tuple(int(match.group(1)) for match in re.finditer(r"\[Shot (\d+)\]", description))
    if not shots or shots[0] != 1 or tuple(dict.fromkeys(shots)) != tuple(range(1, max(shots) + 1)):
        found.append(f"{_V_SHOTS}: number [Shot N] sequentially from 1 in detailed_description")
    times = tuple(_seconds(*match.groups()) for match in _TIMESTAMP.finditer(description))
    if any(later <= earlier for earlier, later in zip(times, times[1:])) or any(t >= spec.duration for t in times):
        found.append(f"{_V_TIMES}: 'At MM:SS.mmm' must strictly increase and stay below {spec.duration}.000")
    words = len(description.split())
    if not DESCRIPTION_WORDS[0] <= words <= DESCRIPTION_WORDS[1]:
        found.append(f"{_V_WORDS}: it has {words}, aim for 350-500")
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
        found.append(f"{_V_RETENTION}: {bad[0]!r}")
    elif unmentioned:
        found.append(f"{_V_RETENTION}: add a retention line for {', '.join(unmentioned)}")
    found.extend(_cut_violations(spec, facts, sections, description))
    return found


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


def validate_prompt(prompt: str, spec: ContextIRRequest, facts: RefFacts | None = None) -> None:
    if not prompt or len(prompt) > 7000:
        raise RewriteError("The rewritten H3 prompt must contain 1 to 7000 characters")
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
    if spec.mode == "ref2va":
        found = ref2va_violations(prompt, spec, facts)
        if found:
            raise RewriteError(found[0].split(": ", 1)[0], violations=tuple(found))


def ref2va_addendum(has_video: bool, has_audio: bool) -> str:
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
    else:
        parts.append("`detailed_description` uses `[Shot N]` markers numbered from 1 and has 200–750 words.")
    parts.append(
        "Write retention lines in the form `<Label> (appears in [Shot ...]): marker - ...`. "
        "Every provided media label appears in `subject_definitions`, and no other media labels are used."
    )
    return "\n" + " ".join(parts)


def _repair_message(violations: tuple[str, ...], labels: tuple[str, ...]) -> str:
    listed = "\n".join(f"- {violation}" for violation in violations)
    return (
        "Your previous answer broke these output rules:\n"
        f"{listed}\n"
        f"The provided media labels are exactly: {', '.join(labels) or 'none'}.\n"
        "Write the complete prompt again in the same six-section format and fix every point."
    )


def _sum_usage(first: RewriteUsage, second: RewriteUsage) -> RewriteUsage:
    cost = None if first.cost is None and second.cost is None else (first.cost or 0.0) + (second.cost or 0.0)
    return RewriteUsage(
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        completion_tokens=first.completion_tokens + second.completion_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
        cost=cost,
    )


class H3PromptRewriter:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key

    async def _complete(self, model: str, messages: list[dict[str, JsonValue]], budget: float) -> _Completion:
        reasoning, max_tokens = REF2VA_MODEL_REASONING.get(model, REF2VA_MODEL_REASONING[MODEL])
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
            raise RewriteError("H3 prompt rewrite provider returned an incomplete response", retryable=True) from exc
        except ValidationError as exc:
            # A wrong/unlisted model id is a provider routing glitch (retry); a truncated or malformed answer
            # (e.g. finish_reason=length) is permanent, as before.
            routing = any(error["loc"][:1] == ("model",) for error in exc.errors())
            raise RewriteError("H3 prompt rewrite provider returned an invalid response", 502, retryable=routing) from exc

    async def rewrite(self, spec: ContextIRRequest) -> RewriteResult:
        if not self.api_key:
            raise RewriteError("H3 prompt rewrite is not configured", 503)
        from litellm.llms.causyn import h3_media

        started = time.monotonic()
        ref2va = spec.mode == "ref2va"
        model = ref2va_model() if ref2va else MODEL  # a misconfigured model fails before any media is fetched
        prepared, raw = await h3_media.prepare_media_with_raw(self.client, spec)
        facts = await perceive_media(prepared, raw) if ref2va else None
        system = system_prompt() + (
            "\nFor the current Ref2VA request, the official six-section reference format replaces the application's three-field format. "
            "Use subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. "
            "When continuation is requested, start from the final visible state of the source clip and continue its motion. "
            "Do not restart a subject entrance, reset positions, or loop the source unless explicitly requested."
            + ref2va_addendum(
                any(isinstance(item, VideoItem) for item in spec.content),
                any(isinstance(item, AudioItem) for item in spec.content),
            )
            if ref2va
            else ""
        )
        user = prepared.user_content(facts, model, send_video_enabled()) if ref2va else prepared.user_content()
        messages: list[dict[str, JsonValue]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # A retry after a retryable failure of the repair call reuses the first answer instead of paying for it again.
        answer_key = media_key(spec, raw.videos, raw.audios) + model if ref2va else ""
        cached_answer = _FIRST_ANSWERS.get(answer_key) if ref2va else None
        if cached_answer is not None:
            prompt, usage = cached_answer, RewriteUsage()
        else:
            completed = await self._complete(model, messages, ATTEMPT_BUDGET_S)
            prompt = completed.choices[0].message.content.strip()
            usage = completed.usage
        try:
            validate_prompt(prompt, spec, facts)
        except RewriteError as failure:
            remaining = ATTEMPT_BUDGET_S - (time.monotonic() - started)
            deadline = ATTEMPT_DEADLINE.get()
            if deadline is not None:
                remaining = min(remaining, deadline - time.monotonic())
            if not ref2va or remaining < MIN_REPAIR_BUDGET_S:
                raise
            _remember(_FIRST_ANSWERS, answer_key, prompt)
            messages += [
                {"role": "assistant", "content": prompt},
                {
                    "role": "user",
                    "content": _repair_message(failure.violations or (str(failure),), _provided_labels(spec)),
                },
            ]
            repaired = await self._complete(model, messages, remaining)  # a failure here keeps the cached first answer
            _FIRST_ANSWERS.pop(answer_key, None)
            prompt = repaired.choices[0].message.content.strip()
            usage = _sum_usage(usage, repaired.usage)
            validate_prompt(prompt, spec, facts)
        _FIRST_ANSWERS.pop(answer_key, None)
        return RewriteResult(
            prompt=prompt, usage=usage, model=model, system_sha256=hashlib.sha256(system.encode()).hexdigest()
        )


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
