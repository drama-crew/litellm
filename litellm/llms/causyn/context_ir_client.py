from __future__ import annotations

import os
import re
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError
from litellm.proxy.video_endpoints.minimax_h3_models import AudioItem, ImageItem, VideoItem


class ContextIRImage(BaseModel):
    model_config = ConfigDict(frozen=True)
    slot: int = Field(ge=1)
    url: str


class ContextIRSubmission(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str
    images: tuple[ContextIRImage, ...]
    duration_s: int
    ratio: str
    idempotency_key: str


class ContextIRTaskView(BaseModel):
    """服务响应的边界校验。

    这是跨服务的信任边界，形状漂移应当在这里变成一条明确的错误，而不是让一个缺了
    caption 的"成功"往下游走。
    """

    model_config = ConfigDict(frozen=True)
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    completed_steps: tuple[str, ...] = ()
    caption: str | None = None
    error: str | None = None
    failed_step: str | None = None
    # 服务侧并行改动会给失败任务带上这两个字段；旧服务没有，缺省即 None。
    retryable: bool | None = None
    error_kind: str | None = None
    upstream_status: int | None = None


TASKS_PATH = "/v1/context-ir/tasks"
SUBMIT_TIMEOUT_S = 30.0
POLL_TIMEOUT_S = 30.0
_BASE_URL_ENV = "DRAMA_CONTEXT_IR_BASE_URL"
_API_KEY_ENV = "DRAMA_CONTEXT_IR_API_KEY"
_ENABLE_ENV = "CAUSYN_REF2VA_CONTEXT_IR_SERVICE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


_LEGACY_TRANSIENT = re.compile(r"HTTP (?:429|408|5\d\d)\b")
MAX_RESUBMISSIONS = 8


def is_transient_failure(task: ContextIRTaskView) -> bool:
    """Whether a failed service task died of an upstream hiccup (worth a fresh task).

    Explicit signals win; only when the service says nothing do we fall back to
    sniffing the legacy error text. Unknown stays terminal.
    """
    if task.retryable is not None:
        return task.retryable
    if task.error_kind is not None:
        return task.error_kind == "upstream_transient"
    return bool(task.error and _LEGACY_TRANSIENT.search(task.error))


def _failure(task: ContextIRTaskView, *, retryable: bool) -> RewriteError:
    from litellm.llms.causyn.h3_prompt import redact_provider_detail

    step = task.failed_step or "unknown"
    return RewriteError(
        f"Context IR service failed at step {step}",
        502,
        retryable=retryable,
        upstream_status=task.upstream_status,
        detail=redact_provider_detail(task.error),
    )


def service_enabled() -> bool:
    return (os.getenv(_ENABLE_ENV) or "").strip().lower() in _TRUTHY


def service_base_url() -> str | None:
    raw = (os.getenv(_BASE_URL_ENV) or "").strip().rstrip("/")
    return raw or None


def service_api_key() -> str | None:
    return os.getenv(_API_KEY_ENV) or None


def should_use_service(spec: ContextIRRequest) -> bool:
    return spec.mode == "ref2va" and all(isinstance(item, ImageItem) for item in spec.ordered_media)


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _media_url(item: ImageItem | VideoItem | AudioItem) -> str:
    if isinstance(item, ImageItem):
        return item.image_url.url
    if isinstance(item, VideoItem):
        return item.video_url.url
    return item.audio_url.url


def _payload(spec: ContextIRRequest, idempotency_key: str) -> ContextIRSubmission:
    return ContextIRSubmission(
        prompt=spec.prompt,
        images=tuple(
            ContextIRImage(slot=index, url=_media_url(item)) for index, item in enumerate(spec.ordered_media, 1)
        ),
        duration_s=spec.duration,
        ratio=str(spec.ratio),
        idempotency_key=idempotency_key,
    )


def _parse(response: httpx.Response) -> ContextIRTaskView:
    """服务响应的边界校验：形状漂移变成一条明确的错误，而不是让缺 caption 的
    "成功"往下游走。"""
    try:
        return ContextIRTaskView.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        raise RewriteError("Context IR service returned an unrecognised task shape", 502, retryable=True) from exc


async def submit(
    spec: ContextIRRequest,
    *,
    base_url: str,
    api_key: str | None,
    idempotency_key: str,
    http: httpx.AsyncClient,
) -> ContextIRTaskView:
    try:
        response = await http.post(
            f"{base_url}{TASKS_PATH}",
            json=_payload(spec, idempotency_key).model_dump(mode="json"),
            headers=_headers(api_key),
            timeout=SUBMIT_TIMEOUT_S,
        )
    except (httpx.TransportError, TimeoutError) as exc:
        raise RewriteError("Context IR service is unreachable", 503, retryable=True) from exc
    if response.status_code != 200:
        raise RewriteError(
            "Context IR service rejected the request",
            response.status_code,
            retryable=response.status_code in _RETRYABLE_STATUS,
            retry_after=response.headers.get("retry-after"),
        )
    return _parse(response)


async def poll(
    task_id: str,
    *,
    base_url: str,
    api_key: str | None,
    http: httpx.AsyncClient,
) -> ContextIRTaskView:
    try:
        response = await http.get(
            f"{base_url}{TASKS_PATH}/{task_id}",
            headers=_headers(api_key),
            timeout=POLL_TIMEOUT_S,
        )
    except (httpx.TransportError, TimeoutError) as exc:
        raise RewriteError("Context IR service is unreachable", 503, retryable=True) from exc
    if response.status_code != 200:
        raise RewriteError(
            "Context IR service status lookup failed",
            response.status_code,
            retryable=response.status_code in _RETRYABLE_STATUS,
        )
    return _parse(response)


SERVICE_MODEL = "h3-context-ir-online"
DEFAULT_POLL_INTERVAL_S = 3.0
# 调用点给单次 rewrite 的上限是 155s；留出提交与收尾的余量。
DEFAULT_BUDGET_S = 120.0
# 服务自己把整个任务限定在 840s 内（内部会重试上游 429）。单次调用只轮询 DEFAULT_BUDGET_S
# （必须小于任务租约 LEASE），其余靠 poll=True 的 1s 重入接力；总等待由
# context_ir.REWRITE_RETRY_WINDOW_S 兜底，它必须 >= SERVICE_TASK_BOUND_S + 余量。
SERVICE_TASK_BOUND_S = 840.0
# 单次尝试超时里要留给提交（SUBMIT_TIMEOUT_S）和收尾的余量。
ATTEMPT_MARGIN_S = 35.0


def poll_budget(remaining_attempt_s: float) -> float:
    """Poll budget that fits inside what is left of the attempt timeout."""
    return max(1.0, min(DEFAULT_BUDGET_S, remaining_attempt_s - ATTEMPT_MARGIN_S))


async def rewrite_via_service(
    spec: ContextIRRequest,
    *,
    base_url: str,
    api_key: str | None,
    idempotency_key: str,
    http: httpx.AsyncClient,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    budget_s: float = DEFAULT_BUDGET_S,
):
    """把服务的 submit/poll 适配成 ContextIRService 要的单次 rewrite。

    九张参考图的改写可能跑 3-5 分钟，远超调用点的 155s 单次上限，所以这里不等到底：
    在预算内轮询，没完成就抛 retryable，让既有的重试机制充当轮询循环。幂等键保证
    重入时命中同一个任务——否则每次重试都会把整条链（81 次量级的模型调用）重跑一遍。
    """
    import asyncio
    import hashlib
    import time

    from litellm.llms.causyn.h3_prompt import RewriteResult, RewriteUsage

    # 服务判定失败但原因是上游瞬时错误时，用带后缀的新幂等键开一个新任务。
    # 已经失败的旧任务在重入时会被幂等命中并立刻返回 failed，沿链走到第一个没死的任务即可；
    # 退避与总预算由外层（deadline 约束的重试）负责，这里只在"刚提交就已失败"时才前进。
    attempt = 0
    while True:
        key = idempotency_key if attempt == 0 else f"{idempotency_key}:retry-{attempt}"
        submitted = await submit(spec, base_url=base_url, api_key=api_key, idempotency_key=key, http=http)
        if submitted.status == "failed" and is_transient_failure(submitted):
            attempt += 1
            if attempt > MAX_RESUBMISSIONS:
                raise _failure(submitted, retryable=False)
            continue
        break
    task_id = submitted.id
    deadline = time.monotonic() + budget_s
    task = submitted
    while True:
        status = task.status
        if status == "succeeded":
            caption = (task.caption or "").strip()
            if not caption:
                raise RewriteError("Context IR service reported success without a caption", 502)
            return RewriteResult(
                prompt=caption,
                usage=RewriteUsage(),
                model=SERVICE_MODEL,
                system_sha256=hashlib.sha256(SERVICE_MODEL.encode()).hexdigest(),
            )
        if status == "failed":
            # 永久失败是终态；瞬时失败抛 retryable，外层退避后重入并走到下一个幂等键。
            raise _failure(task, retryable=is_transient_failure(task) and attempt < MAX_RESUBMISSIONS)
        if time.monotonic() >= deadline:
            raise RewriteError(f"Context IR task {task_id} is still running", 503, retryable=True, poll=True)
        if poll_interval_s:
            await asyncio.sleep(poll_interval_s)
        task = await poll(task_id, base_url=base_url, api_key=api_key, http=http)


def idempotency_key_for(spec: ContextIRRequest, task_id: str = "") -> str:
    """从请求内容派生幂等键。

    rewrite_prompt 拿不到任务 id，但同一个视频任务的 spec 是恒定的，所以内容哈希
    足以把重试锚定到同一个改写任务上——这正是避免整条链重跑的关键。
    """
    import hashlib

    material = "\x1f".join(
        [
            spec.prompt,
            str(spec.duration),
            str(spec.ratio),
            *[_media_url(item) for item in spec.ordered_media],
            *([task_id] if task_id else []),
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()
