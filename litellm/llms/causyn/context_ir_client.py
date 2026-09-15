from __future__ import annotations

import os
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError


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


TASKS_PATH = "/v1/context-ir/tasks"
SUBMIT_TIMEOUT_S = 30.0
POLL_TIMEOUT_S = 30.0
_BASE_URL_ENV = "DRAMA_CONTEXT_IR_BASE_URL"
_API_KEY_ENV = "DRAMA_CONTEXT_IR_API_KEY"
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def service_base_url() -> str | None:
    raw = (os.getenv(_BASE_URL_ENV) or "").strip().rstrip("/")
    return raw or None


def service_api_key() -> str | None:
    return os.getenv(_API_KEY_ENV) or None


def should_use_service(spec: ContextIRRequest) -> bool:
    return spec.mode == "ref2va"


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _payload(spec: ContextIRRequest, idempotency_key: str) -> ContextIRSubmission:
    return ContextIRSubmission(
        prompt=spec.prompt,
        images=tuple(
            ContextIRImage(slot=index, url=str(item.image_url.url)) for index, item in enumerate(spec.ordered_media, 1)
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

    submitted = await submit(
        spec,
        base_url=base_url,
        api_key=api_key,
        idempotency_key=idempotency_key,
        http=http,
    )
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
            # 服务已判定失败是终态；继续重试只会把同一个失败重放三遍。
            step = task.failed_step or "unknown"
            raise RewriteError(f"Context IR service failed at step {step}", 502, retryable=False)
        if time.monotonic() >= deadline:
            raise RewriteError(f"Context IR task {task_id} is still running", 503, retryable=True)
        if poll_interval_s:
            await asyncio.sleep(poll_interval_s)
        task = await poll(task_id, base_url=base_url, api_key=api_key, http=http)


def idempotency_key_for(spec: ContextIRRequest) -> str:
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
            *[str(item.image_url.url) for item in spec.ordered_media],
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()
