from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache

from pydantic import JsonValue

from litellm.llms.causyn.context_ir_budget import settle_reservation
from litellm.llms.causyn.context_ir_callback import notify_callback
from litellm.llms.causyn.context_ir_store import (
    PENDING,
    READY,
    BillingIdentity,
    ContextIRStore,
    ContextIRTask,
    Notification,
    RedisPort,
    new_task_id,
)
from litellm.llms.causyn.h3_prompt import (
    ATTEMPT_DEADLINE,
    AUTH_MODEL,
    PRICE_CREDITS,
    PUBLIC_MODEL,
    ContextIRRequest,
    RewriteError,
    RewriteResult,
    causyn_reference_limit_violation,
    rewrite_prompt,
)
from litellm.llms.causyn.video_prompt import deliver_video_prompt
from litellm.llms.libtv.billing_outbox import CausynBillingEvent, enqueue_causyn_billing
from litellm.llms.libtv.transfer import get_transfer_redis

from .task_telemetry import interval, stage, task_scope

logger = logging.getLogger(__name__)
Rewrite = Callable[[ContextIRRequest], Awaitable[RewriteResult]]
Settle = Callable[[ContextIRTask], Awaitable[None]]
Notify = Callable[[str, dict[str, JsonValue]], Awaitable[bool]]


# 改写重试预算。causyn-1.1 的改写钉死在单一上游 provider（无法回退），限流是常态，
# 所以预算按任务自己的 deadline 而不是固定次数；MiniMax-H3 共用同一代码路径，保持原来
# 的 3 次 / 30s 封顶退避。
LEGACY_MAX_ATTEMPTS = 3
REWRITE_RETRY_WINDOW_S = float(os.getenv("CAUSYN_REWRITE_RETRY_WINDOW_SECONDS", "900"))
# 一次改写尝试最长 155s，加上租约收尾余量；任务在此之后仍没有结果才算过期。
ATTEMPT_TIMEOUT_S = 155.0
EXPIRY_MARGIN_S = 85.0
LEGACY_EXPIRY_S = 600.0
REWRITE_BACKOFF_CAP_S = 60.0
RENDER_RESERVE_S = 1800.0
DEFAULT_TASK_DEADLINE_S = 3600.0


@dataclass(frozen=True)
class RewritePolicy:
    max_attempts: int | None
    window_s: float
    backoff_cap_s: float
    render_reserve_s: float
    expiry_s: float

    def anchor(self, task: ContextIRTask, now: float) -> float:
        """Where the retry window starts: queue entry for the legacy budget, the first attempt otherwise."""
        if self.max_attempts is not None:
            return float(task.created_at)
        return task.rewrite_started_at if task.rewrite_started_at is not None else now

    def next_delay(self, task: ContextIRTask, exc: RewriteError, *, now: float | None = None) -> float | None:
        """Seconds to wait before the next rewrite attempt, or None to give up."""
        if not exc.retryable:
            return None
        if self.max_attempts is not None and task.attempts >= self.max_attempts:
            return None
        now = time.time() if now is None else now
        legacy_cap = self.max_attempts is not None
        delay = exc.retry_delay(
            task.attempts,
            cap=self.backoff_cap_s,
            poll_fast=not legacy_cap,
            jitter=1.0 if legacy_cap else max(1.0, 0.5 * min(self.backoff_cap_s, 2.0**task.attempts)),
        )
        if legacy_cap:
            return min(delay, max(0.0, self.anchor(task, now) + self.expiry_s - now))
        if now + delay - self.anchor(task, now) > self.window_s:
            return None
        if task_deadline(task) - (now + delay) < self.render_reserve_s:
            return None
        return delay


def task_deadline(task: ContextIRTask) -> float:
    raw = (task.video_payload or {}).get("deadline_ts")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    try:
        seconds = float(os.getenv("CAUSYN_H3_DEADLINE_SECONDS", DEFAULT_TASK_DEADLINE_S))
    except ValueError:
        seconds = DEFAULT_TASK_DEADLINE_S
    return task.created_at + seconds


def rewrite_policy(task: ContextIRTask) -> RewritePolicy:
    if task.request.model == AUTH_MODEL:
        return RewritePolicy(
            None,
            REWRITE_RETRY_WINDOW_S,
            REWRITE_BACKOFF_CAP_S,
            RENDER_RESERVE_S,
            REWRITE_RETRY_WINDOW_S + ATTEMPT_TIMEOUT_S + EXPIRY_MARGIN_S,
        )
    return RewritePolicy(LEGACY_MAX_ATTEMPTS, LEGACY_EXPIRY_S, 30.0, 0.0, LEGACY_EXPIRY_S)


async def settle_task(task: ContextIRTask) -> None:
    from litellm.llms.causyn.h3_prompt import AUTH_MODEL

    actual = actual_cost(task)
    if task.reservation is not None:
        await settle_reservation(task.id, task.reservation, actual)
    phase = financial_event(task)
    binding = phase.binding if phase else None
    if actual > 0 or phase is not None:
        await enqueue_causyn_billing(
            get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL")),
            CausynBillingEvent(
                provider_task_id=task.id,
                response_cost=actual,
                model=PUBLIC_MODEL,
                task_type="h3_context_ir",
                api_key=task.billing.api_key,
                team_id=task.billing.team_id,
                user_id=task.billing.user_id,
                organization_id=task.billing.organization_id,
                deployment_id=AUTH_MODEL,
                reservation_id=str(task.reservation["reservation_id"])
                if task.reservation and task.reservation.get("reservation_id")
                else None,
                moderation_intent_id=binding.intent_id if binding else None,
                metering_event_json=phase.model_dump_json() if phase else None,
                billing_facts_json=financial_facts(task).model_dump_json(),
                occurred_at=financial_facts(task).ended_at.isoformat(),
            ),
        )


def financial_facts(task: ContextIRTask):
    from datetime import datetime, timezone
    from decimal import Decimal

    from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts

    if task.financial_facts_json is not None:
        return BillingFacts.model_validate_json(task.financial_facts_json)
    usage = task.result.usage if task.result else None
    return BillingFacts(
        started_at=datetime.fromtimestamp(task.created_at, timezone.utc),
        ended_at=datetime.fromtimestamp(task.updated_at, timezone.utc),
        route="h3_context_ir",
        pricing=(("output_cost_per_task", Decimal(str(task.price))),),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
    )


def financial_event(task: ContextIRTask):
    from decimal import Decimal

    from litellm.llms.causyn.h3_prompt import AUTH_MODEL
    from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent, request_id

    if task.financial_event_json is not None:
        return PhaseEvent.model_validate_json(task.financial_event_json)
    actual = actual_cost(task)
    binding = BillingBinding.model_validate_json(task.metering_binding_json) if task.metering_binding_json else None
    phase = (
        PhaseEvent(
            binding=binding,
            request_id=request_id(binding, "completion"),
            phase="completion",
            provider="causyn",
            deployment_id=AUTH_MODEL,
            native_id=task.id,
            provider_task_id=task.id,
            amount=Decimal(str(actual)),
            finalized=True,
            facts=financial_facts(task),
        )
        if binding
        else None
    )
    return phase


def actual_cost(task: ContextIRTask) -> float:
    if task.financial_actual is not None:
        return task.financial_actual
    return task.price if task.result is not None and task.status not in {"failed", "cancelled"} else 0.0


def freeze_financial(task: ContextIRTask) -> ContextIRTask:
    from decimal import Decimal

    from litellm.proxy.video_endpoints.moderation_metering_projection import actual_debit

    if (
        task.financial_actual is not None
        or task.financial_facts_json is not None
        or task.financial_event_json is not None
    ):
        return task
    raw = Decimal(str(actual_cost(task)))
    frozen = task.model_copy(
        update={
            "financial_actual": float(actual_debit(raw)),
            "financial_facts_json": financial_facts(task).with_actual(raw).model_dump_json(),
        }
    )
    phase = financial_event(frozen)
    return frozen.model_copy(update={"financial_event_json": phase.model_dump_json() if phase else None})


class ContextIRService:
    def __init__(
        self,
        store: ContextIRStore,
        rewrite: Rewrite = rewrite_prompt,
        settle: Settle = settle_task,
        notify: Notify = notify_callback,
        deliver: Settle = deliver_video_prompt,
    ) -> None:
        self.store = store
        self.rewrite = rewrite
        self.settle = settle
        self.notify = notify
        self.deliver = deliver
        self.worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self.run(), name="causyn-h3-context-ir")

    async def stop(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None

    async def run(self) -> None:
        async with asyncio.TaskGroup() as group:
            for queue in (PENDING, READY):
                for _ in range(4):
                    group.create_task(self.run_lane(queue))

    async def run_lane(self, queue: str) -> None:
        while True:
            try:
                for task_id in await self.store.due(queue, limit=1):
                    await self.process(task_id, defer_completion=queue == PENDING)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Context IR queue processing failed")
            await asyncio.sleep(0.5)

    async def create(
        self,
        spec: ContextIRRequest,
        *,
        owner: str,
        billing: BillingIdentity,
        price: float = PRICE_CREDITS,
        task_id: str | None = None,
        listed: bool = True,
        reservation: dict[str, JsonValue] | None = None,
        video_payload: dict[str, JsonValue] | None = None,
    ) -> ContextIRTask:
        if spec.model == AUTH_MODEL:
            if spec.duration < 5:
                raise RewriteError("duration must be from 5 through 15 seconds for causyn-1.1", 400)
            violation = causyn_reference_limit_violation(spec)
            if violation is not None:
                raise RewriteError(violation, 400)
        now = int(time.time())
        from litellm.proxy.video_endpoints.moderation_metering_runtime import CONTEXT

        scope = CONTEXT.get()
        task = ContextIRTask(
            id=task_id or new_task_id(owner),
            owner=owner,
            request=spec,
            created_at=now,
            trace_created_ns=time.time_ns(),
            updated_at=now,
            billing=billing,
            price=price,
            listed=listed,
            reservation=reservation,
            metering_binding_json=scope.binding.model_dump_json()
            if scope is not None and video_payload is None
            else None,
            video_payload=video_payload,
        )
        return await self.store.create(self.with_notification(task))

    async def process(self, task_id: str, *, defer_completion: bool = False) -> None:
        with task_scope(task_id), stage("causyn.ir.attempt"):
            token = uuid.uuid4().hex
            if not await self.store.claim(task_id, token):
                return
            try:
                async with asyncio.timeout(210):
                    await self.process_claimed(task_id, token, defer_completion=defer_completion)
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.wait_for(self.store.retry(task_id, token, 0), timeout=5))
                raise
            except Exception:
                logger.exception("Context IR persistence or settlement failed for %s", task_id)
                await asyncio.wait_for(self.store.retry(task_id, token, 2), timeout=5)

    async def process_claimed(self, task_id: str, token: str, *, defer_completion: bool = False) -> None:
        original = await self.store.get(task_id)
        if original is None:
            return
        original = await self.flush_notifications(original, token)
        if original.status in {"succeeded", "failed", "cancelled"}:
            if not original.settled:
                await self.settle(original)
            await self.finish(original.model_copy(update={"settled": True}), token)
            return
        if original.result is None and self.expired(original):
            await self.fail(original, token, "Context IR task expired")
            return
        running = original.model_copy(update={"status": "running", "updated_at": int(time.time())})
        await self.store.save(self.with_notification(running) if original.status != "running" else running, token)
        refreshed = await self.store.get(task_id)
        if refreshed is None:
            return
        running = await self.flush_notifications(refreshed, token)
        if running.result is not None:
            await self.complete(running, token)
            return
        await self.rewrite_and_complete(running, token, defer_completion=defer_completion)

    @staticmethod
    def expired(task: ContextIRTask, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        policy = rewrite_policy(task)
        if policy.max_attempts is not None:
            return now - task.created_at > policy.expiry_s
        # Queue wait does not eat the rewrite window, but a render must still fit.
        if task_deadline(task) - now < policy.render_reserve_s:
            return True
        return task.rewrite_started_at is not None and now - task.rewrite_started_at > policy.expiry_s

    async def rewrite_and_complete(self, running: ContextIRTask, token: str, *, defer_completion: bool = False) -> None:
        task_id = running.id
        policy = rewrite_policy(running)
        if policy.max_attempts is not None and running.attempts >= policy.max_attempts:
            await self.fail(running, token, "H3 prompt rewrite retry limit exceeded")
            return
        now = time.time()
        first_attempt = running.rewrite_started_at is None and running.attempts == 0
        if first_attempt:
            interval("causyn.ir.queue", running.trace_created_ns or running.created_at * 1_000_000_000, time.time_ns())
            running = running.model_copy(update={"rewrite_started_at": now if policy.max_attempts is None else None})
        attempted = running.model_copy(update={"attempts": running.attempts + 1})
        await self.store.save(attempted, token)
        attempt_timeout = max(0.0, min(ATTEMPT_TIMEOUT_S, policy.anchor(attempted, now) + policy.expiry_s - now))
        deadline_token = ATTEMPT_DEADLINE.set(time.monotonic() + attempt_timeout)
        try:
            try:
                async with asyncio.timeout(attempt_timeout):
                    with stage("causyn.prompt.rewrite", attempt=attempted.attempts):
                        result = await self.rewrite(attempted.request)
            except TimeoutError as exc:
                if policy.max_attempts is not None:
                    raise
                # 尝试超时不是终态：服务侧任务多半还在跑，走 poll 路径让窗口决定何时放弃。
                raise RewriteError("H3 prompt rewrite attempt timed out", 503, retryable=True, poll=True) from exc
        except RewriteError as exc:
            delay = policy.next_delay(attempted, exc)
            if exc.retryable and delay is not None:
                logger.warning(
                    "Context IR rewrite transient failure for %s attempt=%s upstream_status=%s retry_in=%.1fs",
                    task_id,
                    attempted.attempts,
                    exc.upstream_status,
                    delay,
                )
                if exc.poll and policy.max_attempts is None:
                    # 仍在运行的轮询重入不是失败：不涨退避指数，也不发 queued 通知。
                    await self.store.save(attempted.model_copy(update={"attempts": running.attempts}), token)
                else:
                    await self.store.save(
                        self.with_notification(attempted.model_copy(update={"status": "queued"})), token
                    )
                await self.store.retry(task_id, token, delay)
                return
            logger.warning(
                "Context IR rewrite failed for %s attempt=%s upstream_status=%s detail=%s",
                task_id,
                attempted.attempts,
                exc.upstream_status,
                exc.detail,
            )
            await self.fail(attempted, token, str(exc), detail=exc.describe_upstream())
            return
        except Exception:
            logger.exception("Context IR rewrite failed for %s", task_id)
            await self.fail(attempted, token, "H3 prompt rewrite failed")
            return
        finally:
            ATTEMPT_DEADLINE.reset(deadline_token)
        completed = attempted.model_copy(
            update={"result": result, "updated_at": int(time.time()), "rewrite_completed_ns": time.time_ns()}
        )
        await self.store.save(completed, token)
        if defer_completion:
            await self.store.retry(task_id, token, 0)
            return
        await self.complete(completed, token)

    async def complete(self, task: ContextIRTask, token: str) -> None:
        try:
            with stage("causyn.gpu.admission"):
                await self.deliver(task)
        except RewriteError as exc:
            if exc.retryable:
                # 渲染侧饱和：改写成果已经花掉真实模型调用，不能因为下游忙就作废。
                # 用投递自己的计数器退避——改写的 attempts 在这一阶段不再增长，
                # 共用它会让退避永远停在同一档、只剩抖动。上界交给视频自己的
                # deadline：deliver_video_prompt 过期时抛的是不可重试的 503。
                retried = task.model_copy(update={"deliver_attempts": task.deliver_attempts + 1})
                await self.store.save(retried, token)
                await self.store.retry(task.id, token, exc.retry_delay(retried.deliver_attempts))
                return
            await self.fail(task, token, str(exc))
            return
        if task.video_payload is not None and task.rewrite_completed_ns is not None:
            interval("causyn.gpu.admission_wait", task.rewrite_completed_ns, time.time_ns())
        task = freeze_financial(task)
        await self.store.save(task, token)
        await self.settle(task)
        await self.finish(
            self.with_notification(
                task.model_copy(update={"status": "succeeded", "settled": True, "updated_at": int(time.time())})
            ),
            token,
        )

    async def fail(self, task: ContextIRTask, token: str, message: str, *, detail: str | None = None) -> None:
        failed = self.with_notification(
            task.model_copy(
                update={"status": "failed", "error": message, "error_detail": detail, "updated_at": int(time.time())}
            )
        )
        failed = freeze_financial(failed)
        await self.store.save(failed, token)
        await self.settle(failed)
        await self.finish(failed.model_copy(update={"settled": True}), token)
        interval(
            "causyn.task.processing",
            task.trace_created_ns or task.created_at * 1_000_000_000,
            time.time_ns(),
            root=True,
            outcome="failed",
        )

    async def wait(self, task_id: str) -> RewriteResult:
        async with asyncio.timeout(300):
            while True:
                task = await self.store.get(task_id)
                if task is None:
                    raise RewriteError("Context IR task not found", 404)
                if task.status == "succeeded" and task.result is not None:
                    return task.result
                if task.status in {"failed", "cancelled"}:
                    raise RewriteError(task.error or "Context IR task cancelled")
                await asyncio.sleep(0.1)

    async def cancel_or_delete(self, task_id: str, owner: str) -> str:
        task = await self.store.get(task_id)
        if task is None or task.owner != owner or not task.listed:
            raise RewriteError("Task not found", 404)
        token = uuid.uuid4().hex
        if not await self.store.claim(task_id, token):
            raise RewriteError("A running task cannot be cancelled", 400)
        current = await self.store.get(task_id)
        if current is None:
            raise RewriteError("Task not found", 404)
        if current.status in {"succeeded", "failed"} and current.settled:
            await self.store.delete(current, token)
            return "deleted"
        if current.status != "queued":
            await self.store.retry(task_id, token, 0)
            raise RewriteError("A running task cannot be cancelled", 400)
        cancelled = self.with_notification(
            current.model_copy(update={"status": "cancelled", "updated_at": int(time.time())})
        )
        cancelled = freeze_financial(cancelled)
        await self.store.save(cancelled, token)
        await self.settle(cancelled)
        await self.finish(cancelled.model_copy(update={"settled": True}), token)
        return "cancelled"

    @staticmethod
    def with_notification(task: ContextIRTask) -> ContextIRTask:
        if task.request.callback_url is None:
            return task
        return task.model_copy(
            update={"notifications": (*task.notifications, Notification(body={"task": task.public()}))}
        )

    async def flush_notifications(self, task: ContextIRTask, token: str) -> ContextIRTask:
        if not task.notifications or task.request.callback_url is None:
            return task
        event = task.notifications[0]
        delivered = await self.notify(task.request.callback_url, event.body)
        remaining = (
            task.notifications[1:]
            if delivered or event.attempts >= 4
            else (event.model_copy(update={"attempts": event.attempts + 1}), *task.notifications[1:])
        )
        updated = task.model_copy(update={"notifications": remaining})
        await self.store.save(updated, token)
        return await self.flush_notifications(updated, token) if delivered else updated

    async def finish(self, task: ContextIRTask, token: str) -> None:
        await self.store.save(task, token)
        updated = await self.flush_notifications(task, token)
        await self.store.save(updated, token, done=not updated.notifications)
        if updated.notifications:
            await self.store.retry(task.id, token, 30)


@lru_cache(maxsize=8)
def _service_for_loop(_loop: asyncio.AbstractEventLoop) -> ContextIRService:
    redis: RedisPort | None = get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL"))
    if redis is None:
        raise RewriteError("Context IR persistence is not configured", 503)
    return ContextIRService(ContextIRStore(redis))


def get_context_ir_service() -> ContextIRService:
    service = _service_for_loop(asyncio.get_running_loop())
    service.start()
    return service


async def stop_context_ir_service() -> None:
    if _service_for_loop.cache_info().currsize:
        await _service_for_loop(asyncio.get_running_loop()).stop()
        _service_for_loop.cache_clear()
