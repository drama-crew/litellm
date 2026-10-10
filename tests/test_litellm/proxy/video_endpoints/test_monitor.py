from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.video_endpoints import monitor

NOW = datetime.now(timezone.utc)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPEN_VIDEO_MONITOR_TEST_DSN"), reason="isolated PostgreSQL required")
async def test_attribution_migration_and_submission_failure_roundtrip():
    import asyncpg
    from litellm.proxy.video_endpoints.openapi_logs import RequestAttribution, create_log, failed

    conn = await asyncpg.connect(os.environ["OPEN_VIDEO_MONITOR_TEST_DSN"])
    schema = "video_origin_" + uuid.uuid4().hex
    await conn.execute(f'CREATE SCHEMA "{schema}"')

    class DB:
        async def execute_raw(self, sql, *args):
            await conn.execute(sql, *args)
            return 1

        async def query_raw(self, sql, *args):
            return [dict(row) for row in await conn.fetch(sql, *args)]

    try:
        await conn.execute(f'SET search_path TO "{schema}"')
        migrations = Path(__file__).resolve().parents[4] / "litellm-proxy-extras/litellm_proxy_extras/migrations"
        await conn.execute((migrations / "20260909094200_openapi_task_history/migration.sql").read_text())
        upgrade = (migrations / "20260930050000_video_request_attribution/migration.sql").read_text()
        await conn.execute(upgrade)
        await conn.execute(upgrade)
        await create_log(
            DB(),
            log_id="attempt-1",
            owner="key",
            user_id="owner",
            endpoint="videos",
            model="seedance-2.5",
            payload={},
            started_at=NOW,
            attribution=RequestAttribution(
                call_source="studio", project_id="project-1", generation_id="generation-1", artifact_id="shot-1"
            ),
        )
        await failed(DB(), "attempt-1", "upstream rejected before creating task")
        row = (await monitor.list_pending(DB(), NOW, "")).items[0]
        assert row.status == "failed" and row.task_id is None
        assert row.call_source == "studio" and row.generation_id == "generation-1"
        assert row.project_id == "project-1" and row.artifact_id == "shot-1" and row.user_id == "owner"
    finally:
        await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_submit_failure_preserves_attribution_without_provider_task_id(monkeypatch):
    from starlette.requests import Request
    from litellm.proxy.video_endpoints import openapi_log_capture as capture

    db = SimpleNamespace(execute_raw=AsyncMock(), query_raw=AsyncMock())
    monkeypatch.setattr(capture, "database", lambda: db)
    context = {
        "call_source": "studio",
        "project_id": "project-1",
        "generation_id": "generation-1",
        "artifact_id": "shot-1",
    }
    request = Request({"type": "http", "headers": [(b"x-litellm-spend-logs-metadata", json.dumps(context).encode())]})
    auth = UserAPIKeyAuth(api_key="test-key", user_id="owner-1", metadata={"project_id": "project-1"})
    log_id = await capture.start(request, auth, {"model": "seedance-2.5", "prompt": "汉" * 100000})
    await capture.failed(log_id, RuntimeError("provider refused submission"))
    args = db.execute_raw.call_args_list[0].args
    assert args[7:11] == ("studio", "project-1", "generation-1", "shot-1")
    assert db.execute_raw.call_args_list[1].args[1] == log_id
    db.query_raw.return_value = [record(id=log_id, task_id=None, status="failed", user_id="owner-1", **context)]
    row = (await monitor.list_pending(db, NOW, "")).items[0]
    assert row.generation_id == "generation-1" and row.project_id == "project-1"
    assert row.call_source == "studio" and row.task_id is None


@pytest.mark.parametrize(
    "header,project",
    [
        ('{"call_source":"studio","project_id":"other","generation_id":"g","artifact_id":"a"}', "project-1"),
        ('{"call_source":"studio","project_id":"project-1","generation_id":"g","artifact_id":"a"}', None),
        ("invalid json", "project-1"),
        ("x" * 9000, "project-1"),
    ],
)
def test_unverified_attribution_cannot_classify_external_request_as_studio(header, project):
    from starlette.requests import Request
    from litellm.proxy.video_endpoints.openapi_log_capture import request_attribution

    request = Request({"type": "http", "headers": [(b"x-litellm-spend-logs-metadata", header.encode())]})
    result = request_attribution(request, UserAPIKeyAuth(user_id="owner-1", metadata={"project_id": project}))
    assert result.call_source == "open_api" and result.generation_id is None


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


@pytest.mark.asyncio
async def test_key_without_user_still_has_minimal_monitor_record(monkeypatch):
    from starlette.requests import Request
    from litellm.proxy.video_endpoints import openapi_log_capture as capture

    db = SimpleNamespace(execute_raw=AsyncMock(), query_raw=AsyncMock())
    monkeypatch.setattr(capture, "database", lambda: db)
    request = Request({"type": "http", "method": "POST", "path": "/videos", "headers": []})
    log_id = await capture.start(
        request, UserAPIKeyAuth(api_key="test-monitor-key"), {"model": "causyn-1.1", "prompt": "private user input"}
    )
    assert log_id == request.scope["openapi_log_id"]
    args = db.execute_raw.call_args.args
    assert "__video_monitor__" in args
    assert args[-1] == "{}"
    assert "private user input" not in str(args)


@pytest.mark.asyncio
async def test_test_pool_task_is_logged_under_its_own_endpoint_and_flagged_for_the_monitor(monkeypatch):
    from starlette.requests import Request
    from litellm.proxy.video_endpoints import openapi_log_capture as capture

    db = SimpleNamespace(execute_raw=AsyncMock(), query_raw=AsyncMock())
    monkeypatch.setattr(capture, "database", lambda: db)
    prod = Request({"type": "http", "method": "POST", "path": "/videos", "headers": []})
    test = Request({"type": "http", "method": "POST", "path": "/videos", "headers": [], "causyn_test_pool": True})
    auth = UserAPIKeyAuth(api_key="test-key", user_id="owner-1")
    await capture.start(prod, auth, {"model": "causyn-1.1"})
    await capture.start(test, auth, {"model": "causyn-1.1"})
    prod_args, test_args = (call.args for call in db.execute_raw.call_args_list)
    assert prod_args[4] == "videos" and test_args[4] == "minimax_h3_test"  # still logged, distinguishable
    db.query_raw.return_value = [record(id="a", endpoint="minimax_h3_test", worker_pool="test", status="failed")]
    page = await monitor.list_pending(db, NOW, "")
    assert page.items[0].worker_pool == "test"
    assert "minimax_h3_test" in db.query_raw.call_args.args[0]
    db.query_raw.return_value = [record(id="b", status="failed")]
    assert (await monitor.list_pending(db, NOW, "")).items[0].worker_pool is None
    assert "WHEN endpoint='minimax_h3_test' THEN 'test'" in monitor.COLUMNS
