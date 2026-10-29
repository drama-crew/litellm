from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal, assert_never

from pydantic import BaseModel, StrictFloat, ValidationError

PRIORITY_ENV = "CAUSYN_H3_READY_PRIORITY_ENABLED"
MAX_PRIORITY = 100_000
_STREAMS = frozenset(("worker:tasks:video_generate", "worker:tasks:video_generate_ref2va"))


class _Reference(BaseModel):
    media_type: Literal["image", "video", "audio"]


class _Features(BaseModel):
    duration_seconds: StrictFloat
    references: tuple[_Reference, ...] = ()


@dataclass(frozen=True, slots=True)
class PriorityRoute:
    stream: str
    priority: int = 0
    catalog_key: str | None = None


def _reference_weight(reference: _Reference) -> int:
    match reference.media_type:
        case "image":
            return 7
        case "video":
            return 35
        case "audio":
            return 20
    assert_never(reference.media_type)


def estimated_priority(request: object) -> int:
    try:
        features = _Features.model_validate(request)
    except ValidationError:
        return 0
    duration = features.duration_seconds
    if not math.isfinite(duration) or not 0 < duration <= 15:
        return 0
    requested_frames = round(duration * 24)
    aligned_frames = requested_frames + (5 - requested_frames) % 17
    frames = aligned_frames if aligned_frames <= 360 else aligned_frames - 17
    reference_weight = sum(_reference_weight(reference) for reference in features.references)
    estimate = (80 + reference_weight) * (max(5, frames) / 124) ** 1.35
    return min(MAX_PRIORITY, max(1, math.ceil(estimate * 10)))


def priority_route(
    stream: str,
    *,
    model: str,
    request: object,
    pool: str | None = None,
) -> PriorityRoute:
    if (
        os.getenv(PRIORITY_ENV, "false").lower() not in ("true", "1")
        or model != "causyn-1.1"
        or pool is not None
        or stream not in _STREAMS
    ):
        return PriorityRoute(stream=stream)
    priority = estimated_priority(request)
    if priority == 0:
        return PriorityRoute(stream=stream)
    return PriorityRoute(
        stream=f"{stream}:cost:{priority}",
        priority=priority,
        catalog_key="worker:priority-streams:" + stream.removeprefix("worker:tasks:"),
    )
