import asyncio
import json
import threading
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Literal

from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class _MaxParallelRequestsWaiter:
    __slots__ = ("future", "loop", "state")

    def __init__(self, loop: asyncio.AbstractEventLoop, future: asyncio.Future[None]):
        self.loop = loop
        self.future = future
        self.state: Literal["queued", "scheduled", "notified", "granted", "cancelled"] = "queued"


@dataclass(frozen=True, slots=True)
class MaxParallelRequestsConfig:
    capacity: int | None
    version: int


class MaxParallelRequestsLimiter:
    __slots__ = ("_active", "_capacity", "_config_version", "_lock", "_waiters")

    def __init__(self, capacity: int, config_version: int = 0):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._active = 0
        self._capacity: int | None = capacity
        self._config_version = config_version
        self._lock = threading.Lock()
        self._waiters: deque[_MaxParallelRequestsWaiter] = deque()

    @property
    def capacity(self) -> int | None:
        with self._lock:
            return self._capacity

    @property
    def _value(self) -> int | None:
        with self._lock:
            if self._capacity is None:
                return None
            return max(self._capacity - self._active, 0)

    async def acquire(self) -> bool:
        loop = asyncio.get_running_loop()
        waiter = _MaxParallelRequestsWaiter(loop=loop, future=loop.create_future())
        with self._lock:
            self._remove_closed_waiters_locked()
            if self._has_capacity_locked() and not self._waiters:
                self._active += 1
                waiter.state = "granted"
                return True
            self._waiters.append(waiter)
            scheduled = self._schedule_waiters_locked()
        self._notify(scheduled)

        while True:
            try:
                await waiter.future
            except BaseException:
                self._cancel_waiter(waiter)
                raise
            with self._lock:
                if self._has_capacity_locked():
                    waiter.state = "granted"
                    self._waiters.remove(waiter)
                    self._active += 1
                    scheduled = self._schedule_waiters_locked()
                    claimed = True
                else:
                    waiter.future = loop.create_future()
                    waiter.state = "queued"
                    scheduled = ()
                    claimed = False
            self._notify(scheduled)
            if claimed:
                return True

    def release(self) -> None:
        with self._lock:
            if self._active < 1:
                raise RuntimeError("cannot release an unacquired limiter")
            self._active -= 1
            scheduled = self._schedule_waiters_locked()
        self._notify(scheduled)

    def update_capacity(self, capacity: int | None, config_version: int) -> None:
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be at least 1")
        with self._lock:
            if config_version < self._config_version:
                return
            self._capacity = capacity
            self._config_version = config_version
            scheduled = self._schedule_waiters_locked()
        self._notify(scheduled)

    def locked(self) -> bool:
        with self._lock:
            return self._capacity is not None and self._active >= self._capacity

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    def _has_capacity_locked(self) -> bool:
        return self._capacity is None or self._active < self._capacity

    def _remove_closed_waiters_locked(self) -> None:
        closed = tuple(waiter for waiter in self._waiters if waiter.loop.is_closed())
        for waiter in closed:
            waiter.state = "cancelled"
        if closed:
            self._waiters = deque(waiter for waiter in self._waiters if waiter.state != "cancelled")

    def _cancel_waiter(self, waiter: _MaxParallelRequestsWaiter) -> None:
        with self._lock:
            if waiter.state == "cancelled":
                return
            if waiter.state == "granted":
                waiter.state = "cancelled"
                self._active -= 1
            else:
                waiter.state = "cancelled"
                self._waiters.remove(waiter)
            scheduled = self._schedule_waiters_locked()
        self._notify(scheduled)

    def _schedule_waiters_locked(self) -> tuple[_MaxParallelRequestsWaiter, ...]:
        self._remove_closed_waiters_locked()
        if not self._has_capacity_locked():
            return ()
        scheduled = tuple(waiter for waiter in self._waiters if waiter.state == "queued")
        for waiter in scheduled:
            waiter.state = "scheduled"
        return scheduled

    def _notify(self, waiters: tuple[_MaxParallelRequestsWaiter, ...]) -> None:
        pending = waiters
        while pending:
            waiter = pending[0]
            pending = pending[1:]
            try:
                waiter.loop.call_soon_threadsafe(self._wake_waiter, waiter)
            except RuntimeError:
                pending = (*pending, *self._cancel_scheduled_waiter(waiter))

    def _cancel_scheduled_waiter(
        self,
        waiter: _MaxParallelRequestsWaiter,
    ) -> tuple[_MaxParallelRequestsWaiter, ...]:
        with self._lock:
            if waiter.state != "scheduled":
                return ()
            waiter.state = "cancelled"
            self._waiters.remove(waiter)
            return self._schedule_waiters_locked()

    def _wake_waiter(self, waiter: _MaxParallelRequestsWaiter) -> None:
        with self._lock:
            if waiter.state != "scheduled":
                return
            if waiter.future.done():
                waiter.state = "cancelled"
                self._waiters.remove(waiter)
                scheduled = self._schedule_waiters_locked()
            else:
                waiter.state = "notified"
                waiter.future.set_result(None)
                return
        self._notify(scheduled)


MaxParallelRequestsClient = asyncio.Semaphore | MaxParallelRequestsLimiter


class AsyncSemaphoreLease:
    __slots__ = ("_held", "_semaphore")

    def __init__(self, semaphore: MaxParallelRequestsClient):
        self._semaphore = semaphore
        self._held = False

    async def acquire(self) -> None:
        if self._held:
            return
        await self._semaphore.acquire()
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        self._semaphore.release()


active_max_parallel_request_lease: ContextVar[AsyncSemaphoreLease | None] = ContextVar(
    "active_max_parallel_request_lease", default=None
)


async def release_max_parallel_request_lease_during(awaitable: Awaitable[None]) -> None:
    lease = active_max_parallel_request_lease.get()
    if lease is None:
        await awaitable
        return
    lease.release()
    await awaitable
    await lease.acquire()


class InitalizeCachedClient:
    @staticmethod
    def get_max_parallel_requests_cache_key(model_id: str, operation: str | None = None) -> str:
        encoded_identity = json.dumps((model_id, operation), ensure_ascii=False, separators=(",", ":"))
        return f"max_parallel_requests_client:{encoded_identity}"

    @staticmethod
    def set_max_parallel_requests_client(
        litellm_router_instance: LitellmRouter,
        model: dict,
        operation: str | None = None,
    ) -> MaxParallelRequestsLimiter | None:
        litellm_params = model.get("litellm_params", {})
        model_id = model["model_info"]["id"]
        cache_key = InitalizeCachedClient.get_max_parallel_requests_cache_key(
            model_id=model_id,
            operation=operation,
        )
        calculated_max_parallel_requests = calculate_max_parallel_requests(
            rpm=litellm_params.get("rpm"),
            max_parallel_requests=litellm_params.get("max_parallel_requests"),
            tpm=litellm_params.get("tpm"),
            default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
        )
        with litellm_router_instance._max_parallel_request_semaphores_lock:
            config = litellm_router_instance._max_parallel_request_configurations.get(model_id)
            if config is None:
                config = MaxParallelRequestsConfig(
                    capacity=calculated_max_parallel_requests,
                    version=0,
                )
                litellm_router_instance._max_parallel_request_configurations[model_id] = config
            if config.capacity is None:
                return None
            existing = litellm_router_instance._max_parallel_request_semaphores.get(cache_key)
            if existing is None:
                existing = MaxParallelRequestsLimiter(
                    capacity=config.capacity,
                    config_version=config.version,
                )
                litellm_router_instance._max_parallel_request_semaphores[cache_key] = existing
        existing.update_capacity(
            capacity=config.capacity,
            config_version=config.version,
        )
        return existing

    @staticmethod
    def update_max_parallel_requests_clients(
        litellm_router_instance: LitellmRouter,
        model: dict,
    ) -> None:
        litellm_params = model.get("litellm_params", {})
        model_id = model["model_info"]["id"]
        calculated_max_parallel_requests = calculate_max_parallel_requests(
            rpm=litellm_params.get("rpm"),
            max_parallel_requests=litellm_params.get("max_parallel_requests"),
            tpm=litellm_params.get("tpm"),
            default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
        )
        prefix = "max_parallel_requests_client:"
        with litellm_router_instance._max_parallel_request_semaphores_lock:
            current = litellm_router_instance._max_parallel_request_configurations.get(model_id)
            config = MaxParallelRequestsConfig(
                capacity=calculated_max_parallel_requests or None,
                version=0 if current is None else current.version + 1,
            )
            litellm_router_instance._max_parallel_request_configurations[model_id] = config
            limiters = tuple(
                limiter
                for key, limiter in litellm_router_instance._max_parallel_request_semaphores.items()
                if json.loads(key.removeprefix(prefix))[0] == model_id
            )
        for limiter in limiters:
            limiter.update_capacity(
                capacity=config.capacity,
                config_version=config.version,
            )
