from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Protocol
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.video_endpoints.openapi_log_capture import database
from litellm.proxy.video_endpoints.openapi_logs import Database, JSON_OBJECT, normalized_status, safe_error
from litellm.types.videos.utils import decode_video_id_with_provider

if TYPE_CHECKING:
    from litellm import Router
    from litellm.llms.causyn.handler import CausynVideoHandler

router = APIRouter()


class MonitorRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    task_id: str | None = None
    public_task_id: str | None = None
    model: str
    status: str
    started_at: datetime
    observed_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    trace_id: str | None = None
    provider_task_id: str | None = None
    observation_error: str | None = None


ROWS = TypeAdapter(list[MonitorRow])


class MonitorPage(BaseModel):
    items: list[MonitorRow]
    next_after: str


class ProviderObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    error: JsonValue = None
    provider_task_id: str


class Progress(BaseModel):
    status: int
    failed_reason: JsonValue = None


class ProgressClient(Protocol):
    async def _apost(self, path: str, body: dict[str, JsonValue], step: str) -> dict[str, JsonValue]: ...


class Credentials(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    model: str = ""


COLUMNS = """id,task_id,public_task_id,model,
    CASE WHEN result->'error'->>'code'='cancelled' THEN 'cancelled' ELSE status END AS status,
    started_at,observed_at,finished_at,
    coalesce(error,result->'error'->>'message',result->>'error') AS error,result->>'trace_id' AS trace_id"""


def require_admin(auth: UserAPIKeyAuth) -> None:
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(403, "Proxy admin authentication required")


def with_provider_id(row: MonitorRow) -> MonitorRow:
    return row.model_copy(
        update={
            "provider_task_id": decode_video_id_with_provider(row.task_id).get("video_id") if row.task_id else None,
        }
    )


async def list_pending(db: Database, since: datetime, after: str) -> MonitorPage:
    if since.tzinfo is None or since > datetime.now(timezone.utc):
        raise HTTPException(422, "Invalid coverage start")
    rows = ROWS.validate_python(
        await db.query_raw(
            f"""SELECT {COLUMNS} FROM "LiteLLM_OpenApiLog"
        WHERE endpoint IN ('videos','minimax_h3') AND historical=false AND id > $2
        AND (status IN ('queued','running','unknown') OR
             (status IN ('failed','timeout','error') AND observed_at >= $1::timestamptz))
        ORDER BY id LIMIT 50""",
            since,
            after,
        )
    )
    return MonitorPage(items=[with_provider_id(r) for r in rows], next_after=rows[-1].id if len(rows) == 50 else "")


async def read_libtv(client: ProgressClient, task_id: str) -> ProviderObservation:
    from litellm.llms.libtv.client import parse_progress

    raw = await client._apost("/api/task/generation/progress", {"taskIds": [task_id]}, "generation/progress")
    progress = Progress.model_validate(parse_progress(raw, "video", task_id))
    return ProviderObservation(
        status={1: "running", 2: "succeeded", 3: "failed"}.get(progress.status, "running"),
        error=progress.failed_reason,
        provider_task_id=task_id,
    )


async def read_causyn(handler: CausynVideoHandler, task_id: str, provider_id: str) -> ProviderObservation:
    from litellm.llms.causyn.topaz import TopazState
    from litellm.llms.libtv.video_generate import fetch_video_generate_task_metadata

    body = await handler._status(task_id)
    if body.error and body.error.code == "cancelled":
        return ProviderObservation(status="cancelled", provider_task_id=provider_id)
    if body.status != "succeeded":
        return ProviderObservation(
            status="running" if body.status == "claimed" else body.status or "unknown",
            error=JSON_OBJECT.validate_python(body.error.model_dump(exclude_none=True)) if body.error else None,
            provider_task_id=provider_id,
        )
    metadata = await fetch_video_generate_task_metadata(provider_id, redis=handler._redis_factory())
    topaz = await handler._redis_factory().get("causyn:topaz:" + provider_id)
    if topaz:
        saved = TopazState.from_json(topaz)
        if saved.phase in ("failed", "indeterminate"):
            return ProviderObservation(
                status="failed",
                error={"code": "upscale_failed", "message": saved.failure_message},
                provider_task_id=provider_id,
            )
        return ProviderObservation(
            status="succeeded" if saved.phase == "completed" else "running", provider_task_id=provider_id
        )
    return ProviderObservation(
        status="running" if metadata and metadata.get("requested_resolution") == "2k" else "succeeded",
        provider_task_id=provider_id,
    )


async def passive_status(task_id: str, model_router: Router | None) -> ProviderObservation:
    decoded = decode_video_id_with_provider(task_id)
    provider = decoded.get("custom_llm_provider")
    provider_id = decoded.get("video_id")
    if not provider or not provider_id:
        raise ValueError("unsupported_video_identity")
    if provider == "causyn":
        from litellm.llms.causyn.handler import CausynVideoHandler

        return await read_causyn(CausynVideoHandler(), task_id, provider_id)
    model_id = model_router.resolve_video_model_id_alias(provider, decoded.get("model_id")) if model_router else None
    deployment = model_router.get_deployment(model_id=model_id) if model_id and model_router else None
    if deployment is None:
        raise ValueError("video_deployment_unavailable")
    params = JSON_OBJECT.validate_python(deployment.litellm_params.model_dump(exclude_none=True))
    credentials = Credentials.model_validate(params)
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    http = AsyncHTTPHandler(timeout=10)
    try:
        if provider == "libtv":
            from litellm.llms.libtv.handler import LibTVLLM

            client = LibTVLLM()._make_client(credentials.api_key, params, async_client=http)
            return await read_libtv(client, provider_id)
        if provider == "wavespeed":
            from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig

            config = WaveSpeedVideoConfig()
            headers = config.validate_environment({}, credentials.model, api_key=credentials.api_key)
            base = (credentials.api_base or config.default_api_base).rstrip("/")
            response = await http.get(
                base + config.poll_route.format(task_id=quote(provider_id, safe="")), headers=headers
            )
            response.raise_for_status()
            payload = response.json()
            return ProviderObservation(
                status=normalized_status(config._status(config._raw_status(payload))),
                error=config._error(payload),
                provider_task_id=provider_id,
            )
        if provider == "xiaoyunque":
            from litellm.llms.xiaoyunque.handler import XiaoyunqueLLM, decode_composite_task_id

            handler = XiaoyunqueLLM()
            xq = handler._make_client(credentials.api_key, params, async_client=http)
            thread_id, run_id = decode_composite_task_id(provider_id)
            raw = await xq.aquery_result(thread_id, run_id)
            video = handler._video_status(task_id, raw)
            return ProviderObservation(
                status=normalized_status(video.status), error=video.error, provider_task_id=provider_id
            )
        raise ValueError("passive_video_provider_unsupported")
    finally:
        await http.close()


Reader = Callable[[str, "Router | None"], Awaitable[ProviderObservation]]


async def probe(db: Database, log_id: str, model_router: Router | None, reader: Reader = passive_status) -> MonitorRow:
    rows = ROWS.validate_python(
        await db.query_raw(
            f"""SELECT {COLUMNS} FROM "LiteLLM_OpenApiLog"
        WHERE id=$1 AND endpoint IN ('videos','minimax_h3') AND historical=false""",
            log_id,
        )
    )
    if not rows:
        raise HTTPException(404, "Video request not found")
    row = with_provider_id(rows[0])
    if row.status not in ("queued", "running", "unknown"):
        return row
    if not row.task_id:
        return (
            row.model_copy(update={"observation_error": "submission_result_missing"})
            if (datetime.now(timezone.utc) - row.started_at).total_seconds() >= 600
            else row
        )
    try:
        result = await asyncio.wait_for(reader(row.task_id, model_router), timeout=12)
        now = datetime.now(timezone.utc)
        message = result.error if isinstance(result.error, str) else json.dumps(result.error, ensure_ascii=False)
        return row.model_copy(
            update={
                "status": result.status,
                "provider_task_id": result.provider_task_id,
                "error": safe_error(message) if result.error else None,
                "observed_at": now,
                "finished_at": now
                if result.status in ("failed", "succeeded", "cancelled", "timeout")
                else row.finished_at,
            }
        )
    except Exception as exc:
        return row.model_copy(
            update={"observation_error": safe_error(str(exc)) if isinstance(exc, ValueError) else type(exc).__name__}
        )


@router.get("/internal/video-monitor", response_model=MonitorPage, include_in_schema=False)
async def monitor_list(
    since: datetime,
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    db: Annotated[Database | None, Depends(database)],
    after: Annotated[str, Query(max_length=200)] = "",
) -> MonitorPage:
    require_admin(auth)
    if db is None:
        raise HTTPException(503, "History database unavailable")
    return await list_pending(db, since, after)


class ProbeRequest(BaseModel):
    log_id: str = Field(min_length=1, max_length=200)


@router.post("/internal/video-monitor/probe", response_model=MonitorRow, include_in_schema=False)
async def monitor_probe(
    body: ProbeRequest,
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    db: Annotated[Database | None, Depends(database)],
) -> MonitorRow:
    require_admin(auth)
    if db is None:
        raise HTTPException(503, "History database unavailable")
    from litellm.proxy.proxy_server import llm_router

    return await probe(db, body.log_id, llm_router)
