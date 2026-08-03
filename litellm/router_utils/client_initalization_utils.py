import asyncio
import json
import threading
from collections import deque
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Awaitable, Iterator, Literal

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
        self.state: Literal["queued", "granted", "cancelled"] = "queued"


class MaxParallelRequestsLimiter:
    __slots__ = ("_active", "_capacity", "_lock", "_waiters")

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._active = 0
        self._capacity = capacity
        self._lock = threading.Lock()
        self._waiters: deque[_MaxParallelRequestsWaiter] = deque()

    @property
    def capacity(self) -> int:
        with self._lock:
            return self._capacity

    @property
    def _value(self) -> int:
        with self._lock:
            return max(self._capacity - self._active, 0)

    async def acquire(self) -> bool:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        waiter = _MaxParallelRequestsWaiter(loop=loop, future=future)
        with self._lock:
            if self._active < self._capacity and not self._waiters:
                self._active += 1
                waiter.state = "granted"
                return True
            self._waiters.append(waiter)
        try:
            await future
            return True
        except BaseException:
            self._cancel_waiter(waiter)
            raise

    def release(self) -> None:
        with self._lock:
            if self._active < 1:
                raise RuntimeError("cannot release an unacquired limiter")
            self._active -= 1
            granted = self._grant_waiters_locked()
        self._notify(granted)

    def update_capacity(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        with self._lock:
            self._capacity = capacity
            granted = self._grant_waiters_locked()
        self._notify(granted)

    def locked(self) -> bool:
        with self._lock:
            return self._active >= self._capacity

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    def _cancel_waiter(self, waiter: _MaxParallelRequestsWaiter) -> None:
        with self._lock:
            if waiter.state == "cancelled":
                return
            if waiter.state == "queued":
                waiter.state = "cancelled"
                return
            waiter.state = "cancelled"
            self._active -= 1
            granted = self._grant_waiters_locked()
        self._notify(granted)

    def _grant_waiters_locked(self) -> tuple[_MaxParallelRequestsWaiter, ...]:
        return tuple(self._iter_granted_waiters_locked())

    def _iter_granted_waiters_locked(self) -> Iterator[_MaxParallelRequestsWaiter]:
        while self._active < self._capacity and self._waiters:
            waiter = self._waiters.popleft()
            if waiter.state != "queued":
                continue
            waiter.state = "granted"
            self._active += 1
            yield waiter

    @staticmethod
    def _notify(waiters: tuple[_MaxParallelRequestsWaiter, ...]) -> None:
        for waiter in waiters:
            waiter.loop.call_soon_threadsafe(MaxParallelRequestsLimiter._resolve_waiter, waiter)

    @staticmethod
    def _resolve_waiter(waiter: _MaxParallelRequestsWaiter) -> None:
        if not waiter.future.done():
            waiter.future.set_result(None)


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
        if not calculated_max_parallel_requests:
            return None
        with litellm_router_instance._max_parallel_request_semaphores_lock:
            existing = litellm_router_instance._max_parallel_request_semaphores.get(cache_key)
            if existing is not None:
                return existing
            limiter = MaxParallelRequestsLimiter(calculated_max_parallel_requests)
            litellm_router_instance._max_parallel_request_semaphores[cache_key] = limiter
            return limiter

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
        if not calculated_max_parallel_requests:
            return
        prefix = "max_parallel_requests_client:"
        with litellm_router_instance._max_parallel_request_semaphores_lock:
            limiters = tuple(
                limiter
                for key, limiter in litellm_router_instance._max_parallel_request_semaphores.items()
                if json.loads(key.removeprefix(prefix))[0] == model_id
            )
            for limiter in limiters:
                limiter.update_capacity(calculated_max_parallel_requests)
