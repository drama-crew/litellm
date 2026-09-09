"""Best-effort history capture at the authenticated public video boundary."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Request
from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import openapi_logs as logs
from litellm.types.videos.main import VideoObject

RAW_METHOD = TypeAdapter[Callable[..., Awaitable[object]]](Callable[..., Awaitable[object]])
INTEGER = TypeAdapter(int)


@dataclass(frozen=True)
class RawDatabase:
    read: Callable[..., Awaitable[object]]
    write: Callable[..., Awaitable[object]]

    async def query_raw(self, query: str, *args: object) -> object:
        return await self.read(query, *args)

    async def execute_raw(self, query: str, *args: object) -> int:
        return INTEGER.validate_python(await self.write(query, *args))


def raw_method(client: object, name: str) -> Callable[..., Awaitable[object]]:
    return RAW_METHOD.validate_python(getattr(client, name))


def database() -> logs.Database | None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    # PrismaWrapper delegates generated methods dynamically. Validate callables
    # at this boundary instead of pretending the wrapper declares those methods.
    return RawDatabase(
        raw_method(prisma_client.db, "query_raw"),
        raw_method(prisma_client.db, "execute_raw"),
    )


async def persist(operation: Awaitable[None]) -> None:
    try:
        await asyncio.wait_for(operation, timeout=2)
    except Exception as exc:  # noqa: BLE001  # history must not turn accepted generations into retryable failures
        verbose_proxy_logger.warning("Open API history write failed: %s", type(exc).__name__)


async def start(request: Request, auth: UserAPIKeyAuth, body: object) -> str | None:
    log_id = str(uuid.uuid4())
    request.scope["openapi_log_id"] = log_id

    async def operation() -> None:
        db = database()
        if db is None:
            return
        spec: object = request.scope.get("minimax_h3_spec")
        data = logs.JSON_OBJECT.validate_python(body)
        payload = spec.model_dump(mode="json", exclude_none=True) if isinstance(spec, BaseModel) else data
        await logs.create_log(
            db,
            log_id=log_id,
            owner=logs.key_owner(auth.api_key or auth.token or ""),
            user_id=auth.user_id or "__video_monitor__",
            endpoint="minimax_h3" if spec is not None else "videos",
            model=str(data.get("model") or ""),
            payload=payload if auth.user_id else {},
            started_at=datetime.now(timezone.utc),
        )

    await persist(operation())
    return log_id


def video_payload(video: VideoObject) -> dict[str, JsonValue]:
    url: object = video._hidden_params.get("url")  # pyright: ignore[reportPrivateUsage]  # VideoObject exposes provider download URLs only here
    return logs.JSON_OBJECT.validate_python(
        {
            **video.model_dump(mode="json", exclude_none=True),
            **({"url": url} if isinstance(url, str) and url.startswith("https://") else {}),
        }
    )


async def submitted(log_id: str | None, video: object) -> None:
    if log_id is None or not isinstance(video, VideoObject):
        return

    async def operation() -> None:
        db = database()
        if db is not None:
            await logs.submitted(db, log_id, video_payload(video))

    await persist(operation())


async def failed(log_id: str | None, exc: Exception) -> None:
    if log_id is None:
        return

    async def operation() -> None:
        db = database()
        if db is not None:
            await logs.failed(db, log_id, logs.safe_error(f"{type(exc).__name__}: {exc}"))

    await persist(operation())


async def observe(auth: UserAPIKeyAuth, video: object) -> None:
    if not isinstance(video, VideoObject):
        return

    async def operation() -> None:
        db = database()
        if db is not None:
            await logs.observe(
                db,
                owner=logs.key_owner(auth.api_key or auth.token or ""),
                task_id=video.id,
                response=video_payload(video),
            )

    await persist(operation())


async def public_id(request: Request, task_id: str) -> None:
    log_id: object = request.scope.get("openapi_log_id")
    if not isinstance(log_id, str):
        return

    async def operation() -> None:
        db = database()
        if db is not None:
            await logs.public_id(db, log_id, task_id)

    await persist(operation())
