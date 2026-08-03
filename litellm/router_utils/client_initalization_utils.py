import asyncio
import json
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Awaitable

from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class AsyncSemaphoreLease:
    __slots__ = ("_held", "_semaphore")

    def __init__(self, semaphore: asyncio.Semaphore):
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
    ) -> asyncio.Semaphore | None:
        litellm_params = model.get("litellm_params", {})
        model_id = model["model_info"]["id"]
        cache_key = InitalizeCachedClient.get_max_parallel_requests_cache_key(
            model_id=model_id,
            operation=operation,
        )
        existing = litellm_router_instance._max_parallel_request_semaphores.get(cache_key)
        if existing is not None:
            return existing
        calculated_max_parallel_requests = calculate_max_parallel_requests(
            rpm=litellm_params.get("rpm"),
            max_parallel_requests=litellm_params.get("max_parallel_requests"),
            tpm=litellm_params.get("tpm"),
            default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
        )
        if not calculated_max_parallel_requests:
            return None
        semaphore = asyncio.Semaphore(calculated_max_parallel_requests)
        litellm_router_instance._max_parallel_request_semaphores[cache_key] = semaphore
        return semaphore
