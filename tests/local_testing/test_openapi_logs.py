from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from litellm.proxy.video_endpoints.openapi_logs import encode_payload, key_owner, observe, record_spend
from litellm.proxy.video_endpoints.openapi_log_query import LogFilters, query_conditions, list_logs


class FakeDB:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    async def execute_raw(self, query, *args):
        self.calls.append((query, args))
        return 1

    async def query_raw(self, query, *args):
        self.calls.append((query, args))
        return self.result


def test_redacts_secrets_and_inline_media_and_bounds_payload():
    encoded = encode_payload(
        {
            "api_key": "private",
            "headers": {"Authorization": "private"},
            "prompt": "hello",
            "content": ["data:image/png;base64,bytes"],
            "nested": {"password": "private"},
        }
    )
    assert "private" not in encoded and "bytes" not in encoded and "hello" in encoded
    assert len(encode_payload({"prompt": "汉" * 100000}).encode()) <= 65536


def test_key_identity_accepts_already_hashed_keys():
    assert key_owner(key_owner("test-secret")) == key_owner("test-secret")


def test_filters_are_parameterized_and_scoped():
    now = datetime.now(timezone.utc)
    sql, args = query_conditions("owner", LogFilters(model="x' OR 1=1", status="failed"), now)
    assert "x' OR" not in sql and "user_id=$1" in sql
    assert args[0] == "owner" and args[-1] == "x' OR 1=1"
    with pytest.raises(HTTPException):
        query_conditions("owner", LogFilters(start=now - timedelta(days=100)), now)


@pytest.mark.asyncio
async def test_poll_updates_are_tenant_scoped_and_do_not_rewrite_terminal_or_unchanged():
    db = FakeDB()
    await observe(
        db,
        task_id="task",
        owner="key-owner",
        response={"id": "task", "status": "completed", "completed_at": 1700000000},
    )
    query, args = db.calls[0]
    assert "owner=$1 AND task_id=$2" in query and "status IS DISTINCT FROM $3" in query
    assert "status NOT IN ('succeeded','failed','cancelled')" in query
    assert args[:3] == ("key-owner", "task", "succeeded") and args[-2:] == (True, 1700000000)


@pytest.mark.asyncio
async def test_zero_cost_poll_is_not_written_and_billing_receipts_are_idempotent():
    db = FakeDB()
    await record_spend(db, {"spend": 0, "metadata": {"open_api_task_id": "task"}})
    assert not db.calls
    await record_spend(
        db, {"spend": 0.2, "request_id": "bill", "api_key": "owner", "metadata": {"open_api_task_id": "task"}}
    )
    assert "ON CONFLICT (id) DO NOTHING" in db.calls[0][0]
    assert db.calls[0][1] == ("bill", "owner", "task", 0.2)


@pytest.mark.asyncio
async def test_pagination_precedes_payload_read_and_returns_count_on_empty_page():
    db = FakeDB([{"total": 23, "items": []}])
    page = await list_logs(db, "owner", LogFilters(page=3, page_size=20), datetime.now(timezone.utc))
    assert page.total == 23 and not page.items
    query, args = db.calls[0]
    assert "selected AS MATERIALIZED" in query and "SELECT id FROM" in query
    assert args[-2:] == (20, 40)


def test_http_query_filters_and_admin_boundary():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.video_endpoints.openapi_log_capture import database
    from litellm.proxy.video_endpoints.openapi_log_query import router

    app = FastAPI()
    app.include_router(router)
    db = FakeDB([{"items": [], "total": 0}])
    app.dependency_overrides[database] = lambda: db
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    client = TestClient(app)
    result = client.get("/internal/openapi-logs?user_id=alice&status=failed&page=2")
    assert result.status_code == 200, result.text
    assert db.calls[-1][1][0] == "alice" and "failed" in db.calls[-1][1]
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    assert client.get("/internal/openapi-logs?user_id=alice").status_code == 403


def test_error_redaction_preserves_actionable_reason():
    from litellm.proxy.video_endpoints.openapi_logs import safe_error

    message = safe_error("HTTP 403: {'api_key': 'private-value', 'Authorization': 'Bearer private-bearer'}")
    assert "HTTP 403" in message and "private-value" not in message and "private-bearer" not in message


@pytest.mark.asyncio
async def test_history_failure_is_not_an_api_failure():
    from litellm.proxy.video_endpoints.openapi_log_capture import persist

    async def broken_storage():
        raise RuntimeError("database offline")

    await persist(broken_storage())


def test_causyn_poll_timestamp_is_not_an_authoritative_finish_time():
    from litellm.proxy.video_endpoints.openapi_logs import VideoSnapshot, completion_time
    from litellm.types.videos.utils import encode_video_id_with_provider

    video = VideoSnapshot(id=encode_video_id_with_provider("native-task", "causyn"), completed_at=1700000000)
    assert completion_time(video) is None
    assert completion_time(VideoSnapshot(id="other-task", completed_at=1700000000)) == 1700000000


def test_minimax_boundary_records_original_input_and_exact_public_task_id(monkeypatch):
    import json
    from unittest.mock import AsyncMock
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3
    from litellm.proxy.video_endpoints import openapi_log_capture
    from litellm.types.videos.main import VideoObject

    db = FakeDB()
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "synthetic-history-test-secret")
    monkeypatch.setattr(openapi_log_capture, "database", lambda: db)
    completed = VideoObject(id="video-native", object="video", status="completed")
    completed._hidden_params = {"url": "https://media.example/result.mp4"}
    processor = AsyncMock(
        side_effect=[
            VideoObject(id="video-native", object="video", status="queued"),
            completed,
        ]
    )
    monkeypatch.setattr(h3.endpoints.ProxyBaseLLMRequestProcessing, "base_process_llm_request", processor)
    app = FastAPI()
    app.dependency_overrides[h3.user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-owner", user_id="owner-user")
    app.include_router(h3.router)
    client = TestClient(app)
    result = client.post(
        "/v2/video_generation",
        json={
            "model": "MiniMax-H3",
            "content": [{"type": "text", "text": "original prompt"}],
            "resolution": "768P",
            "duration": 8,
            "ratio": "16:9",
        },
    )
    assert result.status_code == 200, result.text
    public_id = result.json()["task_id"]
    assert db.calls[0][1][2:4] == ("owner-user", "minimax_h3")
    assert json.loads(db.calls[0][1][-1])["model"] == "MiniMax-H3"
    assert db.calls[1][1][1] == "video-native"
    assert db.calls[2][1][1] == public_id
    polled = client.get("/v2/query/video_generation/" + public_id)
    assert polled.status_code == 200, polled.text
    assert db.calls[-1][1][:3] == (key_owner("sk-owner"), "video-native", "succeeded")
