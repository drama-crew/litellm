"""Idempotent import of historical video summaries; no provider/API calls.

Run after the new Prisma schema is deployed. Old spend rows retained neither
request bodies nor task status: preserve that absence rather than fabricate it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from prisma import Prisma
from pydantic import BaseModel, TypeAdapter

from litellm.proxy.video_endpoints.openapi_logs import safe_error
from litellm.types.videos.utils import decode_video_id_with_provider


class LegacyRow(BaseModel):
    request_id: str
    api_key: str
    user: str
    model: str
    status: str | None
    startTime: datetime
    endTime: datetime
    spend: float
    error: str | None


async def main() -> None:
    db = Prisma()
    await db.connect()
    try:
        # Newest 90 days are bounded; keyset batches avoid growing OFFSET scans.
        # Stop at the first live capture so failed submissions (no task ID)
        # cannot be imported again under a different legacy ID.
        cursor = ""
        count = 0
        while True:
            rows = TypeAdapter(list[LegacyRow]).validate_python(
                await db.query_raw(
                    """SELECT request_id, api_key, "user", model,
                status, "startTime", "endTime", spend, metadata->'error_information'->>'error_message' AS error
                FROM "LiteLLM_SpendLogs" WHERE call_type='avideo_generation'
                AND coalesce("user",'')<>'' AND "startTime">now()-interval '90 days'
                AND "startTime" < coalesce((SELECT min(started_at) FROM "LiteLLM_OpenApiLog" WHERE NOT historical),now())
                AND request_id>$1
                ORDER BY request_id LIMIT 250""",
                    cursor,
                )
            )
            if not rows:
                break
            for row in rows:
                decoded = decode_video_id_with_provider(row.request_id)
                billing_id = (
                    "causyn:" + str(decoded.get("video_id") or row.request_id)
                    if decoded.get("custom_llm_provider") == "causyn"
                    else None
                )
                failed = row.status == "failure"
                await db.execute_raw(
                    """INSERT INTO "LiteLLM_OpenApiLog"
                    (id,owner,user_id,endpoint,model,task_id,public_task_id,billing_request_id,status,started_at,
                    observed_at,finished_at,elapsed_estimated,historical,error)
                    SELECT $1,$2,$3,'unknown',$4,$5,$5,$6,$7,$8::timestamptz,$9::timestamptz,$10::timestamptz,true,true,$11
                    WHERE NOT EXISTS (SELECT 1 FROM "LiteLLM_OpenApiLog" WHERE owner=$2 AND task_id=$5)
                    ON CONFLICT (id) DO NOTHING""",
                    "legacy:" + row.request_id,
                    row.api_key,
                    row.user,
                    row.model,
                    None if failed else row.request_id,
                    billing_id,
                    "failed" if failed else "unknown",
                    row.startTime,
                    row.endTime,
                    row.endTime if failed else None,
                    safe_error(row.error) if row.error else None,
                )
                if row.spend > 0:
                    await db.execute_raw(
                        """INSERT INTO "LiteLLM_OpenApiLogSpend" (id,owner,task_id,amount)
                        VALUES ($1,$2,$1,$3) ON CONFLICT (id) DO NOTHING""",
                        row.request_id,
                        row.api_key,
                        row.spend,
                    )
            cursor = rows[-1].request_id
            count += len(rows)
            logging.warning("Processed %s historical summaries", count)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
