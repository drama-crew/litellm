from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
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
    role: Literal["first_frame", "last_frame", "reference"]
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
                    ImageItem(
                        type="image_url",
                        image_url=MediaURL(url=ref.url),
                        role="reference_image" if ref.role == "reference" else ref.role,
                    )
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


async def deliver_video_prompt(
    task: ContextIRTask,
    *,
    redis_factory: Callable[[], RedisPort | None] | None = None,
    enqueue: Callable[..., Awaitable[str]] | None = None,
) -> None:
    """Hand a finished rewrite to the render queue.

    ``redis_factory`` and ``enqueue`` exist so tests can drive this exact path
    instead of substituting their own error: the classification below is the
    behaviour worth protecting, and patching module globals leaks across tests
    under randomised ordering. Both default to the production collaborators.
    """
    if task.video_payload is None:
        return
    payload = rewritten_video_payload(task)
    resolve = redis_factory or (lambda: get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL")))
    submit = enqueue or enqueue_video_generate
    redis: RedisPort | None = resolve()
    if redis is None:
        raise RuntimeError("Video task persistence is not configured")
    if await redis.get(status_key(payload.task_id)) is not None:
        return
    if time.time() >= payload.deadline_ts:
        raise RewriteError("Video submission expired before GPU admission", 503)
    try:
        await submit(
            payload.model_dump(mode="json"),
            redis_factory=lambda: redis,
            settings=VideoGenerateSettings.from_environment(),
        )
    except VideoGenerateError as exc:
        if exc.code in {"invalid_params", "invalid_url"}:
            raise RewriteError("Invalid video generation input", 400) from None
        if exc.code in {"no_capacity_available", "no_worker_available"}:
            # 渲染侧饱和是背压，不是故障。不分类的话它会逃到 process() 的兜底
            # 处理：每 2 秒重试一次、每次打一条完整栈、无退避无计数，一直到视频
            # deadline——把"GPU 正忙"这个正常状态当成了未预期错误。
            #
            # 429 让 RewriteError 自带 retryable，并复用改写阶段那套指数退避
            # （封顶 30s）。改写成果必须保住：它已经花掉了真实的模型调用，所以
            # 这里重新排期而不是失败。上界由视频自己的 deadline 负责。
            raise RewriteError("Render queue is saturated", 429, retryable=True) from None
        raise


async def submit_video_prompt(payload: VideoSubmission, billing: BillingIdentity) -> None:
    from litellm.llms.causyn.context_ir import get_context_ir_service

    service = get_context_ir_service()
    await service.create(
        VideoPromptInput.model_validate(payload.request).context_ir(),
        owner=f"video:{payload.task_id}",
        billing=billing,
        price=0.0,
        task_id=PREFIX + payload.task_id,
        listed=False,
        video_payload=payload.model_dump(mode="json"),
    )
