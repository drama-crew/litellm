from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from typing import Annotated, Generic, Literal, TypeVar
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from litellm.litellm_core_utils.causyn_reference_limits import reference_count_limit_violation
from litellm.proxy.video_endpoints.minimax_h3_paths import LEGACY_PUBLIC_MODEL, PUBLIC_MODEL, UNAVAILABLE_PUBLIC_MODEL

ModelName = Literal["MiniMax-H3", "MiniMax-H3-Max", "causyn-1.1"]
ContentModel = TypeVar("ContentModel", bound=str)
Ratio = Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
Resolution = Literal["480P", "768P", "2K"]
MODEL_ROUTES: dict[str, str] = {
    "MiniMax-H3": "hailuo-h3",
    "MiniMax-H3-Max": "hailuo-h3-max",
    "causyn-1.1": "causyn-1.1",
}
TASK_PREFIX = "h3_task_"
TASK_TTL = 7 * 24 * 3600
PromptExpansionMode = Literal["disabled", "balanced", "quality"]
# Public duration window. The official API also lists 4 s for MiniMax-H3, which this deployment does not offer yet.
MIN_PUBLIC_DURATION = 5
MAX_PUBLIC_DURATION = 15
DURATION_MESSAGE = (
    f"duration must be an integer from {MIN_PUBLIC_DURATION} through {MAX_PUBLIC_DURATION} seconds "
    "(4 s is not offered yet)"
)


def canonical_public_model(value: object) -> str:
    """Official ``MiniMax-H3`` (canonical) or its legacy ``minimax-h3`` alias; anything else is a clear 400."""
    if value in (PUBLIC_MODEL, LEGACY_PUBLIC_MODEL):
        return PUBLIC_MODEL
    if value == UNAVAILABLE_PUBLIC_MODEL:
        raise ValueError(f"model {UNAVAILABLE_PUBLIC_MODEL} is not available on this service; use {PUBLIC_MODEL}")
    raise ValueError(f"model must be {PUBLIC_MODEL}")


def resolution_unavailable(resolution: str) -> str:
    return f"resolution {resolution} is not available on this service; only 768P is supported"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaURL(StrictModel):
    url: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
            return value
        if value.startswith("data:") and ";base64," in value:
            try:
                base64.b64decode(value.split(",", 1)[1], validate=True)
            except ValueError as exc:
                raise ValueError("invalid Base64 media") from exc
            return value
        if value.startswith("mm_file://"):
            raise ValueError(
                "mm_file:// references are not supported (no file-upload API here); "
                "provide a public HTTP(S) URL or a Base64 data URL"
            )
        raise ValueError("media must be a public HTTP(S) URL or Base64 data URL")


class TextItem(StrictModel):
    type: Literal["text"]
    text: str = Field(min_length=1, max_length=7000)


class ImageItem(StrictModel):
    type: Literal["image_url"]
    image_url: MediaURL
    role: Literal["first_frame", "last_frame", "reference_image"] = "first_frame"


class VideoItem(StrictModel):
    type: Literal["video_url"]
    video_url: MediaURL
    role: Literal["reference_video"] = "reference_video"


class AudioItem(StrictModel):
    type: Literal["audio_url"]
    audio_url: MediaURL
    role: Literal["reference_audio"] = "reference_audio"


ContentItem = Annotated[TextItem | ImageItem | VideoItem | AudioItem, Field(discriminator="type")]


def _reference_media(content: list[ContentItem]) -> dict[str, object]:
    # This is the wire shape the LibTV backend actually reads (typed lists plus
    # modeType); every route this facade drives resolves to LibTV, which has no
    # concept of a flat "references" list, so building anything else here means
    # LibTV silently drops the reference media and bills a plain text-to-video.
    grouped = {
        kind: values
        for kind, values in (
            ("reference_images", [item.image_url.url for item in content if isinstance(item, ImageItem)]),
            ("reference_videos", [item.video_url.url for item in content if isinstance(item, VideoItem)]),
            ("reference_audios", [item.audio_url.url for item in content if isinstance(item, AudioItem)]),
        )
        if values
    }
    return {**grouped, **({"parameters": {"modeType": "mixed2video"}} if len(content) > 1 else {})}


class H3Content(StrictModel, Generic[ContentModel]):
    model: ContentModel
    content: list[ContentItem] = Field(min_length=1, max_length=16)
    resolution: Resolution
    duration: int = Field(ge=4, le=15, strict=True)
    ratio: Ratio = "adaptive"
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_combination(self) -> Self:
        text = [item for item in self.content if isinstance(item, TextItem)]
        images = [item for item in self.content if isinstance(item, ImageItem)]
        videos = [item for item in self.content if isinstance(item, VideoItem)]
        audios = [item for item in self.content if isinstance(item, AudioItem)]
        if len(text) != 1 or not text[0].text.strip():
            raise ValueError("content must include exactly one non-empty text item")
        if self.model == "MiniMax-H3-Max":
            if self.duration < 5 or self.resolution == "2K":
                raise ValueError("MiniMax-H3-Max supports 5-15 seconds and 480P/768P")
        elif self.resolution == "480P":
            raise ValueError(resolution_unavailable("480P"))
        keyframes = [item for item in images if item.role != "reference_image"]
        references = [item for item in images if item.role == "reference_image"]
        if self.model == "causyn-1.1":
            if self.resolution != "768P":
                raise ValueError("causyn-1.1 supports 768P resolution")
            if videos or audios:
                raise ValueError("causyn-1.1 supports image references only")
            if references and self.ratio == "adaptive":
                raise ValueError("causyn-1.1 reference images require an explicit ratio")
        if keyframes and (references or videos or audios):
            raise ValueError("first/last frames and reference media cannot be mixed")
        if any(sum(item.role == role for item in keyframes) > 1 for role in ["first_frame", "last_frame"]):
            raise ValueError("at most one first_frame and one last_frame are allowed")
        if len(references) > 9 or len(videos) > 3 or len(audios) > 3:
            raise ValueError("reference count exceeds 9 images, 3 videos or 3 audio clips")
        if self.model == "MiniMax-H3-Max" and (references or videos or audios):
            raise ValueError("MiniMax-H3-Max does not support reference-to-video")
        if len(self.content) == 1 and self.ratio == "adaptive":
            raise ValueError("text-to-video requires an explicit, non-adaptive ratio")
        return self

    @property
    def keyframes(self) -> tuple[ImageItem, ...]:
        return tuple(item for item in self.content if isinstance(item, ImageItem) and item.role != "reference_image")

    @property
    def effective_ratio(self) -> Ratio:
        return "adaptive" if self.keyframes else self.ratio

    def internal_body(self) -> dict[str, object]:
        if self.model == "causyn-1.1":
            return {
                "model": "causyn-1.1",
                "prompt": next(item.text for item in self.content if isinstance(item, TextItem)),
                "seconds": str(self.duration),
                "resolution": "768p",
                "aspect_ratio": self.effective_ratio,
                "generate_audio": True,
                "references": [
                    {
                        "role": "reference" if item.role == "reference_image" else item.role,
                        "media_type": "image",
                        "url": item.image_url.url,
                    }
                    for item in self.content
                    if isinstance(item, ImageItem)
                ],
            }
        media: dict[str, object] = (
            {
                **{
                    "image" if item.role == "first_frame" else "last_image": item.image_url.url
                    for item in self.keyframes
                },
                "parameters": {"modeType": "frames2video"},
            }
            if self.keyframes
            else _reference_media(self.content)
        )
        return {
            "model": MODEL_ROUTES[self.model],
            "prompt": next(item.text for item in self.content if isinstance(item, TextItem)),
            "seconds": str(self.duration),
            "resolution": self.resolution,
            "aspect_ratio": self.effective_ratio,
            "generate_audio": True,
            **media,
        }


class MiniMaxH3Content(H3Content[ModelName]):
    pass


MAX_REFERENCE_ITEMS = 12


class ExtraOptions(StrictModel):
    """Official ``extra`` object: ``prompt_expansion_mode`` is its only key."""

    prompt_expansion_mode: PromptExpansionMode | None = None

    @field_validator("prompt_expansion_mode", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("prompt_expansion_mode must be one of disabled, balanced, quality")
        return value


class MiniMaxH3Create(H3Content[Literal["MiniMax-H3"]]):
    duration: int = Field(strict=True)
    extra: ExtraOptions | None = None

    @field_validator("model", mode="before")
    @classmethod
    def accept_official_and_legacy_name(cls, value: object) -> str:
        return canonical_public_model(value)

    @field_validator("duration")
    @classmethod
    def supported_duration(cls, value: int) -> int:
        if not MIN_PUBLIC_DURATION <= value <= MAX_PUBLIC_DURATION:
            raise ValueError(DURATION_MESSAGE)
        return value

    @property
    def prompt_expansion_mode(self) -> PromptExpansionMode | None:
        return self.extra.prompt_expansion_mode if self.extra else None

    @model_validator(mode="after")
    def validate_hyperflow(self) -> Self:
        if self.resolution != "768P":
            raise ValueError(resolution_unavailable(self.resolution))
        videos = [item for item in self.content if isinstance(item, VideoItem)]
        audios = [item for item in self.content if isinstance(item, AudioItem)]
        images = [item for item in self.content if isinstance(item, ImageItem) and item.role == "reference_image"]
        if videos or audios:
            if violation := reference_count_limit_violation(len(images), len(videos), len(audios)):
                raise ValueError(violation)
            if self.ratio == "adaptive":
                raise ValueError("reference media requires an explicit ratio")
        if self.ratio == "adaptive" and images:
            raise ValueError("reference images require an explicit ratio")
        return self

    def internal_body(self) -> dict[str, object]:
        references: list[dict[str, str]] = []
        for item in self.content:
            if isinstance(item, ImageItem):
                role = "reference" if item.role == "reference_image" else item.role
                references.append({"role": role, "media_type": "image", "url": item.image_url.url})
            elif isinstance(item, VideoItem):
                references.append({"role": "reference", "media_type": "video", "url": item.video_url.url})
            elif isinstance(item, AudioItem):
                references.append({"role": "reference", "media_type": "audio", "url": item.audio_url.url})
        return {
            "model": "causyn-1.1",
            "prompt": next(item.text for item in self.content if isinstance(item, TextItem)),
            "seconds": str(self.duration),
            "resolution": "768p",
            "aspect_ratio": self.effective_ratio,
            "generate_audio": True,
            "references": references,
        }


class MiniMaxH3DirectCreate(MiniMaxH3Create):
    """Direct facade: same limits as the IR prefix; its prompt is passed unchanged instead of being rewritten."""


class MiniMaxTask(StrictModel):
    v: Literal[1] = 1
    native_id: str = Field(min_length=1)
    model: ModelName
    created_at: int
    duration: int
    resolution: Resolution
    ratio: Ratio
    image_count: int
    owner: str = Field(min_length=1)


def task_key() -> bytes:
    secret = os.getenv("LITELLM_VIDEO_ID_SECRET") or os.getenv("LITELLM_SALT_KEY") or os.getenv("LITELLM_MASTER_KEY")
    if not secret:
        raise RuntimeError("Video task signing is not configured")
    return hashlib.sha256(b"litellm-minimax-h3-task-v1\0" + secret.encode()).digest()


def encode_task(task: MiniMaxTask) -> str:
    nonce = os.urandom(12)
    payload = task.model_dump_json().encode()
    cipher = AESGCM(task_key()).encrypt(nonce, payload, TASK_PREFIX.encode())
    return TASK_PREFIX + base64.urlsafe_b64encode(nonce + cipher).rstrip(b"=").decode("ascii")


def decode_task(value: str) -> MiniMaxTask:
    try:
        if not value.startswith(TASK_PREFIX) or len(value) > 16384:
            raise ValueError("invalid task ID")
        encoded = value[len(TASK_PREFIX) :]
        blob = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        payload = AESGCM(task_key()).decrypt(blob[:12], blob[12:], TASK_PREFIX.encode())
        task = MiniMaxTask.model_validate(json.loads(payload))
        if not 0 <= time.time() - task.created_at <= TASK_TTL:
            raise ValueError("task is outside the 7-day query window")
        return task
    except Exception as exc:
        raise ValueError("task not found or outside the 7-day query window") from exc
