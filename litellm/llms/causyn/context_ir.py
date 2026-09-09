from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache

from pydantic import JsonValue

from litellm.llms.causyn.context_ir_budget import settle_reservation
from litellm.llms.causyn.context_ir_callback import notify_callback
from litellm.llms.causyn.context_ir_store import (
    BillingIdentity,
    ContextIRStore,
    ContextIRTask,
    Notification,
    RedisPort,
    new_task_id,
)
from litellm.llms.causyn.h3_prompt import (
    PRICE_CREDITS,
    PUBLIC_MODEL,
    ContextIRRequest,
    RewriteError,
    RewriteResult,
    rewrite_prompt,
)
from litellm.llms.causyn.video_prompt import deliver_video_prompt
from litellm.llms.libtv.billing_outbox import CausynBillingEvent, enqueue_causyn_billing
from litellm.llms.libtv.transfer import get_transfer_redis

logger = logging.getLogger(__name__)
Rewrite = Callable[[ContextIRRequest], Awaitable[RewriteResult]]
Settle = Callable[[ContextIRTask], Awaitable[None]]
Notify = Callable[[str, dict[str, JsonValue]], Awaitable[bool]]


async def settle_task(task: ContextIRTask) -> None:
    if task.result is not None and task.status not in {"failed", "cancelled"} and task.price > 0:
        await enqueue_causyn_billing(
            get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL")),
            CausynBillingEvent(
                provider_task_id=task.id,
                response_cost=task.price,
                model=PUBLIC_MODEL,
                task_type="h3_context_ir",
                api_key=task.billing.api_key,
                team_id=task.billing.team_id,
                user_id=task.billing.user_id,
                organization_id=task.billing.organization_id,
            ),
        )
    if task.reservation is not None:
        await settle_reservation(
            task.id,
            task.reservation,
            task.price if task.result is not None and task.status not in {"failed", "cancelled"} else 0.0,
        )


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
        while True:
            try:
                await asyncio.gather(*(self.process(task_id) for task_id in await self.store.due()))
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
        spec.require_supported()
        now = int(time.time())
        task = ContextIRTask(
            id=task_id or new_task_id(owner),
            owner=owner,
            request=spec,
            created_at=now,
            updated_at=now,
            billing=billing,
            price=price,
            listed=listed,
            reservation=reservation,
            video_payload=video_payload,
        )
        return await self.store.create(self.with_notification(task))

    async def process(self, task_id: str) -> None:
        token = uuid.uuid4().hex
        if not await self.store.claim(task_id, token):
            return
        try:
            async with asyncio.timeout(210):
                await self.process_claimed(task_id, token)
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.wait_for(self.store.retry(task_id, token, 0), timeout=5))
            raise
        except Exception:
            logger.exception("Context IR persistence or settlement failed for %s", task_id)
            await asyncio.wait_for(self.store.retry(task_id, token, 2), timeout=5)

    async def process_claimed(self, task_id: str, token: str) -> None:
        original = await self.store.get(task_id)
        if original is None:
            return
        original = await self.flush_notifications(original, token)
        if original.status in {"succeeded", "failed", "cancelled"}:
            if not original.settled:
                await self.settle(original)
            await self.finish(original.model_copy(update={"settled": True}), token)
            return
        if original.result is None and time.time() - original.created_at > 600:
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
        await self.rewrite_and_complete(running, token)

    async def rewrite_and_complete(self, running: ContextIRTask, token: str) -> None:
        task_id = running.id
        attempted = running.model_copy(update={"attempts": running.attempts + 1})
        await self.store.save(attempted, token)
        try:
            result = await self.rewrite(attempted.request)
        except RewriteError as exc:
            if exc.status_code == 429 and attempted.attempts < 3:
                await self.store.save(self.with_notification(attempted.model_copy(update={"status": "queued"})), token)
                await self.store.retry(task_id, token, (0.0, 2.0, 4.0)[attempted.attempts])
                return
            await self.fail(attempted, token, str(exc))
            return
        except Exception:
            logger.exception("Context IR rewrite failed for %s", task_id)
            await self.fail(attempted, token, "H3 prompt rewrite failed")
            return
        completed = attempted.model_copy(update={"result": result, "updated_at": int(time.time())})
        await self.store.save(completed, token)
        await self.complete(completed, token)

    async def complete(self, task: ContextIRTask, token: str) -> None:
        try:
            await self.deliver(task)
        except RewriteError as exc:
            await self.fail(task, token, str(exc))
            return
        await self.settle(task)
        await self.finish(
            self.with_notification(
                task.model_copy(update={"status": "succeeded", "settled": True, "updated_at": int(time.time())})
            ),
            token,
        )

    async def fail(self, task: ContextIRTask, token: str, message: str) -> None:
        failed = self.with_notification(
            task.model_copy(update={"status": "failed", "error": message, "updated_at": int(time.time())})
        )
        await self.store.save(failed, token)
        await self.settle(failed)
        await self.finish(failed.model_copy(update={"settled": True}), token)

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
