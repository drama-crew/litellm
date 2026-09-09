from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.video_endpoints import monitor

NOW = datetime.now(timezone.utc)


def record(**updates):
    return {
        "id": "log-1",
        "task_id": "video-task",
        "model": "seedance-2.5",
        "status": "running",
        "started_at": NOW,
        "observed_at": NOW,
        **updates,
    }


def test_admin_boundary():
    with pytest.raises(HTTPException) as denied:
        monitor.require_admin(UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER))
    assert denied.value.status_code == 403
    monitor.require_admin(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN))


@pytest.mark.asyncio
async def test_keyset_page_returns_last_id_without_skipping_rows():
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[record(id=f"{i:03}") for i in range(50)]))
    result = await monitor.list_pending(db, NOW, "000")
    assert result.next_after == "049"
    assert len(result.items) == 50
    query, since, after = db.query_raw.call_args.args
    assert "id > $2" in query and "ORDER BY id LIMIT 50" in query
    assert (since, after) == (NOW, "000")


@pytest.mark.asyncio
async def test_confirmed_failure_and_query_failure_are_distinct():
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[record()]))
    reader = AsyncMock(
        return_value=monitor.ProviderObservation(
            status="failed", error={"message": "upstream rejected"}, provider_task_id="up-1"
        )
    )
    failed = await monitor.probe(db, "log-1", None, reader)
    assert failed.status == "failed" and failed.finished_at
    assert failed.provider_task_id == "up-1" and failed.observation_error is None
    reader.side_effect = TimeoutError("secret in driver message")
    unavailable = await monitor.probe(db, "log-1", None, reader)
    assert unavailable.status == "running"
    assert unavailable.observation_error == "TimeoutError"
    assert "secret" not in unavailable.model_dump_json()


@pytest.mark.asyncio
async def test_terminal_submission_error_does_not_poll():
    db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[record(status="failed", task_id=None, error="invalid ratio")])
    )
    reader = AsyncMock()
    row = await monitor.probe(db, "log-1", None, reader)
    assert row.error == "invalid ratio"
    reader.assert_not_called()


@pytest.mark.asyncio
async def test_libtv_passive_probe_only_reads_progress():
    raw = AsyncMock(return_value={"data": {"progresses": [{"taskId": "provider-123", "status": 2}]}})
    result = await monitor.read_libtv(SimpleNamespace(_apost=raw), "provider-123")
    assert result.status == "succeeded"
    raw.assert_awaited_once_with("/api/task/generation/progress", {"taskIds": ["provider-123"]}, "generation/progress")


@pytest.mark.asyncio
async def test_causyn_cancel_and_pending_upscale_are_passive():
    import json
    from litellm.llms.causyn.handler import _StatusEnvelope

    async def get(key):
        return json.dumps({"requested_resolution": "2k"}) if key.startswith("worker:task:metadata:") else None

    redis = SimpleNamespace(get=get)
    success = SimpleNamespace(
        _status=AsyncMock(return_value=_StatusEnvelope(status="succeeded")), _redis_factory=lambda: redis
    )
    assert (await monitor.read_causyn(success, "opaque", "abc")).status == "running"
    cancelled = SimpleNamespace(
        _status=AsyncMock(return_value=_StatusEnvelope(status="failed", error={"code": "cancelled"})),
        _redis_factory=lambda: redis,
    )
    assert (await monitor.read_causyn(cancelled, "opaque", "abc")).status == "cancelled"
