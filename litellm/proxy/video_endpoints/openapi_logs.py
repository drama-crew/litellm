"""Task-level, tenant-scoped history. Never replay requests or bill from this table."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.types.videos.utils import decode_video_id_with_provider

Status = Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown"]
TERMINAL = frozenset(("succeeded", "failed", "cancelled"))
SECRET = re.compile(r"authorization|cookie|api.?key|token|secret|password|credential", re.I)
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
STATUS_NAMES: dict[str, Status] = {
    "queued": "queued",
    "pending": "queued",
    "in_progress": "running",
    "running": "running",
    "completed": "succeeded",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


class Database(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...
    async def execute_raw(self, query: str, *args: object) -> int: ...


class VideoSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = ""
    status: str = "unknown"
    completed_at: int | None = None
    error: JsonValue = None


def key_owner(identity: str) -> str:
    return identity if re.fullmatch(r"[0-9a-f]{64}", identity) else hashlib.sha256(identity.encode()).hexdigest()


def safe_json(value: JsonValue, depth: int = 0) -> JsonValue:
    """Bound log size; exclude secrets, inline bytes and arbitrary headers."""
    if depth > 12:
        return "[nested content omitted]"
    if isinstance(value, dict):
        return {
            k: safe_error(json.dumps(v, ensure_ascii=False)) if k.lower() == "error" else safe_json(v, depth + 1)
            for k, v in tuple(value.items())[:100]
            if not SECRET.search(k) and k.lower() not in ("headers", "metadata")
        }
    if isinstance(value, list):
        return [safe_json(v, depth + 1) for v in value[:40]]
    if isinstance(value, str):
        if value.startswith("data:"):
            return "[inline media omitted]"
        return value[:16000] + (" [truncated]" if len(value) > 16000 else "")
    return value


def safe_error(message: str) -> str:
    bearer = re.sub(r"(?i)(bearer\s+|sk-)[A-Za-z0-9_.-]+", "[redacted]", message[:4000])
    return re.sub(
        r"(?i)([\"']?(?:api.?key|authorization|token|secret|password)[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,}]+)",
        r"\1[redacted]",
        bearer,
    )[:2000]


def encode_payload(value: JsonValue) -> str:
    encoded = json.dumps(safe_json(value), ensure_ascii=False)
    return encoded if len(encoded.encode()) <= 65536 else json.dumps({"_truncated": True, "summary": encoded[:12000]})


def normalized_status(value: str) -> Status:
    return STATUS_NAMES.get(value, "unknown")


async def create_log(
    db: Database,
    *,
    log_id: str,
    owner: str,
    user_id: str,
    endpoint: str,
    model: str,
    payload: JsonValue,
    started_at: datetime,
) -> None:
    await db.execute_raw(
        """INSERT INTO "LiteLLM_OpenApiLog"
        (id, owner, user_id, endpoint, model, status, started_at, observed_at, input, historical, elapsed_estimated)
        VALUES ($1,$2,$3,$4,$5,'queued',$6::timestamptz,$6::timestamptz,$7::jsonb,false,false) ON CONFLICT (id) DO NOTHING""",
        log_id,
        owner,
        user_id,
        endpoint,
        model,
        started_at,
        encode_payload(payload),
    )


async def submitted(db: Database, log_id: str, response: dict[str, JsonValue]) -> None:
    snap = VideoSnapshot.model_validate(response)
    decoded = decode_video_id_with_provider(snap.id)
    billing_id = (
        "causyn:" + str(decoded.get("video_id") or snap.id) if decoded.get("custom_llm_provider") == "causyn" else None
    )
    await db.execute_raw(
        """UPDATE "LiteLLM_OpenApiLog" SET task_id=$2, public_task_id=$2, billing_request_id=$6,
        status=$3, result=$4::jsonb, observed_at=now(), finished_at=CASE WHEN $7 THEN
        coalesce(to_timestamp($5::double precision),now()) ELSE NULL END,
        elapsed_estimated=($7 AND $5::bigint IS NULL), error=$8 WHERE id=$1""",
        log_id,
        snap.id or None,
        normalized_status(snap.status),
        encode_payload(response),
        snap.completed_at,
        billing_id,
        normalized_status(snap.status) in TERMINAL,
        safe_error(encode_payload(snap.error)) if snap.error is not None else None,
    )


async def observe(db: Database, *, task_id: str, owner: str, response: dict[str, JsonValue]) -> None:
    snap = VideoSnapshot.model_validate(response)
    state = normalized_status(snap.status)
    # No write for identical queued/running polls. A terminal observation is immutable.
    await db.execute_raw(
        """UPDATE "LiteLLM_OpenApiLog" SET status=$3, result=$4::jsonb,
        error=$5, observed_at=now(), finished_at=CASE WHEN $6 THEN coalesce(to_timestamp($7::double precision),now())
        ELSE NULL END, elapsed_estimated=($6 AND $7::bigint IS NULL)
        WHERE owner=$1 AND task_id=$2 AND status NOT IN ('succeeded','failed','cancelled') AND status IS DISTINCT FROM $3""",
        owner,
        task_id,
        state,
        encode_payload(response),
        safe_error(encode_payload(snap.error)) if snap.error is not None else None,
        state in TERMINAL,
        snap.completed_at,
    )


async def failed(db: Database, log_id: str, message: str) -> None:
    await db.execute_raw(
        """UPDATE "LiteLLM_OpenApiLog" SET status='failed', error=$2,
        finished_at=now(), observed_at=now() WHERE id=$1""",
        log_id,
        message[:2000],
    )


async def public_id(db: Database, log_id: str, task_id: str) -> None:
    await db.execute_raw('UPDATE "LiteLLM_OpenApiLog" SET public_task_id=$2 WHERE id=$1', log_id, task_id)


async def record_spend(db: Database, payload: dict[str, JsonValue]) -> None:
    raw = payload.get("metadata")
    metadata = JSON_OBJECT.validate_json(raw) if isinstance(raw, str) else JSON_OBJECT.validate_python(raw or {})
    task = metadata.get("open_api_task_id")
    amount = payload.get("spend")
    if not isinstance(task, str) or not isinstance(amount, (float, int)) or amount <= 0:
        return
    await db.execute_raw(
        """INSERT INTO "LiteLLM_OpenApiLogSpend" (id, owner, task_id, amount)
        VALUES ($1,$2,$3,$4) ON CONFLICT (id) DO NOTHING""",
        payload.get("request_id"),
        payload.get("api_key"),
        task,
        amount,
    )
