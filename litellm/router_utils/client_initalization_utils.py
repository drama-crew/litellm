import asyncio
from typing import TYPE_CHECKING, Any

from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class InitalizeCachedClient:
    @staticmethod
    def get_max_parallel_requests_cache_key(model_id: str, operation: str | None = None) -> str:
        operation_segment = f"_{operation}" if operation else ""
        return f"{model_id}{operation_segment}_max_parallel_requests_client"

    @staticmethod
    def set_max_parallel_requests_client(
        litellm_router_instance: LitellmRouter,
        model: dict,
        operation: str | None = None,
    ):
        litellm_params = model.get("litellm_params", {})
        model_id = model["model_info"]["id"]
        rpm = litellm_params.get("rpm", None)
        tpm = litellm_params.get("tpm", None)
        max_parallel_requests = litellm_params.get("max_parallel_requests", None)
        calculated_max_parallel_requests = calculate_max_parallel_requests(
            rpm=rpm,
            max_parallel_requests=max_parallel_requests,
            tpm=tpm,
            default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
        )
        if calculated_max_parallel_requests:
            semaphore = asyncio.Semaphore(calculated_max_parallel_requests)
            cache_key = InitalizeCachedClient.get_max_parallel_requests_cache_key(
                model_id=model_id,
                operation=operation,
            )
            litellm_router_instance.cache.set_cache(
                key=cache_key,
                value=semaphore,
                local_only=True,
            )
