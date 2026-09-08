from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from typing import Annotated, Literal
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

ModelName = Literal["MiniMax-H3", "MiniMax-H3-Max"]
Ratio = Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
Resolution = Literal["480P", "768P", "2K"]
MODEL_ROUTES: dict[str, str] = {"MiniMax-H3": "hailuo-h3", "MiniMax-H3-Max": "hailuo-h3-max"}
TASK_PREFIX = "h3_task_"
TASK_TTL = 7 * 24 * 3600


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


class MiniMaxH3Create(StrictModel):
    model: ModelName
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
            raise ValueError("MiniMax-H3 supports 768P/2K")
        keyframes = [item for item in images if item.role != "reference_image"]
        references = [item for item in images if item.role == "reference_image"]
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
        media: dict[str, object] = (
            {
                **{
                    "image" if item.role == "first_frame" else "last_image": item.image_url.url
                    for item in self.keyframes
                },
                "parameters": {"modeType": "frames2video"},
            }
            if self.keyframes
            else {
                **{
                    kind: values
                    for kind, values in (
                        (
                            "reference_images",
                            [item.image_url.url for item in self.content if isinstance(item, ImageItem)],
                        ),
                        (
                            "reference_videos",
                            [item.video_url.url for item in self.content if isinstance(item, VideoItem)],
                        ),
                        (
                            "reference_audios",
                            [item.audio_url.url for item in self.content if isinstance(item, AudioItem)],
                        ),
                    )
                    if values
                },
                **({"parameters": {"modeType": "mixed2video"}} if len(self.content) > 1 else {}),
            }
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
