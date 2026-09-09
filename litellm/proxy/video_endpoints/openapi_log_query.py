"""Indexed, bounded list query. Payload columns are fetched only after pagination."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.video_endpoints.openapi_log_capture import database
from litellm.proxy.video_endpoints.openapi_logs import Database, Status


class LogFilters(BaseModel):
    page: int = Field(default=1, ge=1, le=10000)
    page_size: int = Field(default=20, ge=1, le=50)
    endpoint: Literal["videos", "minimax_h3"] | None = None
    status: Status | None = None
    model: str | None = Field(default=None, max_length=200)
    task_id: str | None = Field(default=None, max_length=16384)
    start: datetime | None = None
    end: datetime | None = None


class LogRow(BaseModel):
    id: str
    task_id: str | None
    endpoint: str
    model: str
    status: Status
    started_at: datetime
    observed_at: datetime
    finished_at: datetime | None
    elapsed_estimated: bool
    historical: bool
    input: JsonValue = None
    result: JsonValue = None
    error: str | None = None
    cost_usd: float | None = None


class LogPage(BaseModel):
    items: list[LogRow]
    total: int
    page: int
    page_size: int


def query_conditions(user_id: str, filters: LogFilters, now: datetime) -> tuple[str, tuple[object, ...]]:
    end = filters.end or now
    start = filters.start or end - timedelta(days=30)
    if start.tzinfo is None or end.tzinfo is None or start >= end or end - start > timedelta(days=90):
        raise HTTPException(422, "Use a timezone-aware time range of at most 90 days")
    optional = tuple(
        (column, value)
        for column, value in (
            ("endpoint", filters.endpoint),
            ("status", filters.status),
            ("model", filters.model),
            ("public_task_id", filters.task_id),
        )
        if value is not None
    )
    clauses = tuple(f'AND "{column}" = ${i + 4}' for i, (column, _) in enumerate(optional))
    return "user_id=$1 AND started_at >= $2::timestamptz AND started_at < $3::timestamptz " + " ".join(clauses), (
        user_id,
        start,
        end,
        *(value for _, value in optional),
    )


async def list_logs(db: Database, user_id: str, filters: LogFilters, now: datetime) -> LogPage:
    where, args = query_conditions(user_id, filters, now)
    # MATERIALIZED prevents wide JSON payload reads before LIMIT. A single MVCC
    # snapshot keeps count and page consistent while other requests are arriving.
    sql = f"""WITH selected AS MATERIALIZED (
        SELECT id FROM "LiteLLM_OpenApiLog" WHERE {where}
        ORDER BY started_at DESC, id DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}
    ), total AS (SELECT count(*) AS n FROM "LiteLLM_OpenApiLog" WHERE {where})
    SELECT (SELECT n FROM total) AS total, coalesce(jsonb_agg(to_jsonb(r) ORDER BY r.started_at DESC,r.id DESC)
        FILTER (WHERE r.id IS NOT NULL),'[]'::jsonb) AS items
    FROM total LEFT JOIN LATERAL (
        SELECT l.id, coalesce(l.public_task_id,l.task_id) AS task_id, l.endpoint,l.model,l.status,
        l.started_at,l.observed_at,l.finished_at,l.elapsed_estimated,l.historical,l.input,l.result,l.error,
        coalesce((SELECT spend FROM "LiteLLM_SpendLogs" b WHERE b.request_id=l.billing_request_id AND b.api_key=l.owner),
        (SELECT sum(s.amount) FROM "LiteLLM_OpenApiLogSpend" s WHERE s.owner=l.owner AND s.task_id=l.task_id)) AS cost_usd
        FROM selected JOIN "LiteLLM_OpenApiLog" l ON l.id=selected.id
    ) r ON true"""
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
        await db.query_raw(sql, *args, filters.page_size, (filters.page - 1) * filters.page_size)
    )
    return LogPage(
        items=TypeAdapter(list[LogRow]).validate_python(rows[0]["items"]),
        total=TypeAdapter(int).validate_python(rows[0]["total"]),
        page=filters.page,
        page_size=filters.page_size,
    )


router = APIRouter()


@router.get("/internal/openapi-logs", response_model=LogPage, include_in_schema=False)
async def history(
    user_id: Annotated[str, Query(min_length=1, max_length=200)],
    filters: Annotated[LogFilters, Depends()],
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    db: Annotated[Database | None, Depends(database)],
) -> LogPage:
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(403, "Proxy admin authentication required")
    if db is None:
        raise HTTPException(503, "History database unavailable")
    return await list_logs(db, user_id, filters, datetime.now(timezone.utc))
