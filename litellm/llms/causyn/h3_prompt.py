from __future__ import annotations

import asyncio
import hashlib
import os
import re
from functools import lru_cache
from importlib.resources import files
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self

from litellm.proxy.video_endpoints.minimax_h3_models import (
    AudioItem,
    ContentItem,
    ImageItem,
    MiniMaxH3Create,
    Ratio,
    TextItem,
    VideoItem,
)

MODEL = "qwen/qwen3.8-flash"
PUBLIC_MODEL = "causyn-h3-context-ir"
AUTH_MODEL = "causyn-1.1"
PRICE_CREDITS = 4.0
BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REFERENCE_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


class RewriteError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class ContextIRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Literal["MiniMax-H3", "causyn-1.1"]
    content: tuple[ContentItem, ...] = Field(min_length=1, max_length=16)
    duration: int = Field(ge=4, le=15, strict=True)
    ratio: Ratio = "adaptive"
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        MiniMaxH3Create(
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

    def require_supported(self) -> None:
        if any(isinstance(item, AudioItem) for item in self.content):
            raise RewriteError("Reference audio is not supported by this Context IR service", 422)

    def user_content(self) -> list[dict[str, JsonValue]]:
        task = {"t2va": "t2av", "i2va": "i2av", "l2va": "l2av", "fl2va": "fl2av", "ref2va": "Ref2VA"}[self.mode]
        return [
            *[
                part
                for index, item in enumerate(self.ordered_media, 1)
                for part in self._media_parts(
                    item, sum(type(previous) is type(item) for previous in self.ordered_media[:index])
                )
            ],
            {
                "type": "text",
                "text": f"task: {task}\nresolution: {self.effective_ratio}\nduration: {self.duration}s\noriginal_prompt: {self.prompt}",
            },
        ]

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
        raise RewriteError("Reference audio is not supported", 422)


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


class _Completion(BaseModel):
    model: Literal["qwen/qwen3.8-flash"]
    choices: tuple[_Choice, ...] = Field(min_length=1, max_length=1)
    usage: RewriteUsage


@lru_cache(maxsize=1)
def system_prompt() -> str:
    return files("litellm.llms.causyn").joinpath("prompts/h3-system.txt").read_text(encoding="utf-8")


def validate_prompt(prompt: str, spec: ContextIRRequest) -> None:
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


class H3PromptRewriter:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key

    async def rewrite(self, spec: ContextIRRequest) -> RewriteResult:
        spec.require_supported()
        if not self.api_key:
            raise RewriteError("H3 prompt rewrite is not configured", 503)
        from litellm.llms.causyn.h3_media import prepare_media

        prepared = await prepare_media(self.client, spec)
        system = system_prompt() + (
            "\nFor the current Ref2VA request, the official six-section reference format replaces the application's three-field format. "
            "Use subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. "
            "When continuation is requested, start from the final visible state of the source clip and continue its motion. "
            "Do not restart a subject entrance, reset positions, or loop the source unless explicitly requested."
            if spec.mode == "ref2va"
            else ""
        )
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prepared.user_content()}],
            "max_tokens": 4096,
            "stream": False,
            "temperature": 0,
            "reasoning": {"enabled": False},
            "provider": {"allow_fallbacks": False, "require_parameters": True},
        }
        async with asyncio.timeout(90):
            response = await self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=85,
                follow_redirects=False,
            )
        if response.status_code != 200:
            raise RewriteError("H3 prompt rewrite provider unavailable", 429 if response.status_code == 429 else 502)
        completed = _Completion.model_validate(response.json())
        prompt = completed.choices[0].message.content.strip()
        validate_prompt(prompt, spec)
        return RewriteResult(
            prompt=prompt, usage=completed.usage, system_sha256=hashlib.sha256(system.encode()).hexdigest()
        )


async def rewrite_prompt(spec: ContextIRRequest) -> RewriteResult:
    async with httpx.AsyncClient(trust_env=False) as client:
        return await H3PromptRewriter(client, os.getenv("OPENROUTER_API_KEY", "")).rewrite(spec)
