from __future__ import annotations

import os
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from litellm.llms.causyn.context_ir_store import PREFIX, BillingIdentity, ContextIRTask, RedisPort
from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError
from litellm.llms.libtv.transfer import get_transfer_redis, status_key
from litellm.llms.libtv.video_generate import (
    VideoGenerateError,
    VideoGenerateSettings,
    enqueue_video_generate,  # pyright: ignore[reportUnknownVariableType]  # shared engine has untyped Redis ports
)
from litellm.proxy.video_endpoints.minimax_h3_models import ImageItem, MediaURL, Ratio, TextItem


class VideoReference(BaseModel):
    role: Literal["first_frame", "last_frame"]
    url: str


class VideoPromptInput(BaseModel):
    prompt: str
    duration_seconds: int
    ratio: Ratio
    references: tuple[VideoReference, ...] = ()

    def context_ir(self) -> ContextIRRequest:
        return ContextIRRequest(
            model="causyn-1.1",
            duration=self.duration_seconds,
            ratio=self.ratio,
            content=(
                TextItem(type="text", text=self.prompt),
                *(
                    ImageItem(type="image_url", image_url=MediaURL(url=ref.url), role=ref.role)
                    for ref in self.references
                ),
            ),
        )


class VideoSubmission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    model: Literal["causyn-1.1"]
    deadline_ts: float
    request: dict[str, JsonValue]
    task_metadata: dict[str, JsonValue]


def rewritten_video_payload(task: ContextIRTask) -> VideoSubmission:
    if task.video_payload is None or task.result is None:
        raise RewriteError("Video rewrite result is unavailable", 503)
    payload = VideoSubmission.model_validate(task.video_payload)
    return payload.model_copy(
        update={
            "request": {**payload.request, "prompt": task.result.prompt},
            "task_metadata": {
                **payload.task_metadata,
                "context_ir_task_id": task.id,
                "prompt_rewrite_model": task.result.model,
                "prompt_rewrite_system_sha256": task.result.system_sha256,
            },
        }
    )


async def deliver_video_prompt(task: ContextIRTask) -> None:
    if task.video_payload is None:
        return
    payload = rewritten_video_payload(task)
    redis: RedisPort | None = get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL"))
    if redis is None:
        raise RuntimeError("Video task persistence is not configured")
    if await redis.get(status_key(payload.task_id)) is not None:
        return
    if time.time() >= payload.deadline_ts:
        raise RewriteError("Video submission expired before GPU admission", 503)
    try:
        await enqueue_video_generate(
            payload.model_dump(mode="json"),
            redis_factory=lambda: redis,
            settings=VideoGenerateSettings.from_environment(),
        )
    except VideoGenerateError as exc:
        if exc.code in {"invalid_params", "invalid_url"}:
            raise RewriteError("Invalid video generation input", 400) from None
        raise


async def submit_video_prompt(payload: VideoSubmission, billing: BillingIdentity) -> None:
    from litellm.llms.causyn.context_ir import get_context_ir_service

    service = get_context_ir_service()
    await service.create(
        VideoPromptInput.model_validate(payload.request).context_ir(),
        owner=f"video:{payload.task_id}",
        billing=billing,
        task_id=PREFIX + payload.task_id,
        listed=False,
        video_payload=payload.model_dump(mode="json"),
    )
