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
    endpoint: Literal["videos", "minimax_h3", "minimax_h3_test"] | None = None
    status: Status | None = None
    model: str | None = Field(default=None, max_length=200)
    task_id: str | None = Field(default=None, max_length=16384)
    # Platform API-key id: matches the originating virtual key's metadata.openapi_key_id,
    # or (legacy keys) its key_alias "openapi-<key_id>".
    key_id: str | None = Field(default=None, min_length=1, max_length=200)
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
    key_id: str | None = None


class LogPage(BaseModel):
    items: list[LogRow]
    total: int
    page: int
    page_size: int


# Virtual keys that belong to a platform API key. Revoked keys move to the deleted table (indexed on token),
# so their history stays reachable. Log rows reference the key by owner = hashed token.
# Rotated keys are aliased "openapi-<id>-r<8 hex>": compared with a fixed-length prefix (no LIKE), so wildcard
# characters in the bound value are inert.
ALIAS_MATCH_SQL = (
    "(key_alias = 'openapi-' || {p}::text OR ("
    "left(key_alias, char_length({p}::text) + 10) = 'openapi-' || {p}::text || '-r' "
    "AND char_length(key_alias) = char_length({p}::text) + 18 "
    "AND substr(key_alias, char_length({p}::text) + 11) ~ '^[0-9a-f]{{8}}$'))"
)
KEY_OWNERS_SQL = (
    'SELECT token FROM "LiteLLM_VerificationToken" '
    "WHERE metadata->>'openapi_key_id' = {p}::text OR " + ALIAS_MATCH_SQL + " "
    'UNION ALL SELECT token FROM "LiteLLM_DeletedVerificationToken" '
    "WHERE metadata->>'openapi_key_id' = {p}::text OR " + ALIAS_MATCH_SQL
)


def query_conditions(user_id: str, filters: LogFilters, now: datetime) -> tuple[str, tuple[object, ...]]:
    end = filters.end or now
    start = filters.start or end - timedelta(days=30)
    if start.tzinfo is None or end.tzinfo is None or start >= end or end - start > timedelta(days=90):
        raise HTTPException(422, "Use a timezone-aware time range of at most 90 days")
    # key_id is always the first optional parameter ($4) so its SQL can reference it twice.
    key_clauses = (f'AND owner IN ({KEY_OWNERS_SQL.format(p="$4")})',) if filters.key_id is not None else ()
    key_args = (filters.key_id,) if filters.key_id is not None else ()
    base = 3 + len(key_args)
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
    clauses = (*key_clauses, *(f'AND "{column}" = ${i + 1 + base}' for i, (column, _) in enumerate(optional)))
    return "user_id=$1 AND started_at >= $2::timestamptz AND started_at < $3::timestamptz " + " ".join(clauses), (
        user_id,
        start,
        end,
        *key_args,
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
        (SELECT coalesce(nullif(k.metadata->>'openapi_key_id', ''),
            nullif(CASE WHEN k.key_alias LIKE 'openapi-%' THEN regexp_replace(substr(k.key_alias, 9), '-r[0-9a-f]{{8}}$', '') END, ''))
         FROM (SELECT 0 AS src, metadata, key_alias FROM "LiteLLM_VerificationToken" WHERE token=l.owner
               UNION ALL SELECT 1 AS src, metadata, key_alias FROM "LiteLLM_DeletedVerificationToken" WHERE token=l.owner
               ORDER BY src LIMIT 1) k) AS key_id,
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
