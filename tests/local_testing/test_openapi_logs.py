from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from litellm.proxy.video_endpoints.openapi_log_query import LogFilters, list_logs, query_conditions
from litellm.proxy.video_endpoints.openapi_logs import encode_payload, key_owner, observe, record_spend


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

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
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


# --- per-key filter (developer platform): key_id param + key_id on every row ---


def _http_client(db, role=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.video_endpoints.openapi_log_capture import database
    from litellm.proxy.video_endpoints.openapi_log_query import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[database] = lambda: db
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role or LitellmUserRoles.PROXY_ADMIN)
    return TestClient(app)


def test_internal_logs_filters_by_key_id_metadata():
    now = datetime.now(timezone.utc)
    sql, args = query_conditions("alice", LogFilters(key_id="key-1"), now)
    # Matches the originating virtual key's metadata (live or already-revoked keys), bound as a parameter.
    assert "metadata->>'openapi_key_id' = $4" in sql
    assert '"LiteLLM_VerificationToken"' in sql and '"LiteLLM_DeletedVerificationToken"' in sql
    assert args == ("alice", args[1], args[2], "key-1") and "key-1" not in sql
    # The key filter resolves to the log's owner column (hashed token), which is how a log row is tied to a key.
    assert "owner IN (" in sql


def test_internal_logs_filters_by_key_alias_fallback():
    now = datetime.now(timezone.utc)
    sql, args = query_conditions("alice", LogFilters(key_id="key-1", status="failed"), now)
    # Keys created before the metadata existed are only identifiable by the "openapi-<key_id>" alias convention.
    assert "key_alias = 'openapi-' || $4::text" in sql
    assert " OR " in sql.split("owner IN (")[1]
    # key_id is positioned right after the time window, followed by later optional filters; no inlined values.
    assert args[3] == "key-1" and args[-1] == "failed"
    assert "$5" in sql and "'openapi-key-1'" not in sql


@pytest.mark.asyncio
async def test_rows_include_key_id():
    row = {
        "id": "r1",
        "task_id": "t",
        "endpoint": "minimax_h3",
        "model": "causyn-1.1",
        "status": "succeeded",
        "started_at": "2026-10-01T00:00:00+00:00",
        "observed_at": "2026-10-01T00:00:01+00:00",
        "finished_at": None,
        "elapsed_estimated": False,
        "historical": False,
        "key_id": "key-1",
    }
    legacy = {**row, "id": "r2", "key_id": None}
    db = FakeDB([{"total": 2, "items": [row, legacy]}])
    page = await list_logs(db, "alice", LogFilters(), datetime.now(timezone.utc))
    assert [r.key_id for r in page.items] == ["key-1", None]
    # A single round trip: key_id is derived inside the paginated lateral join, not looked up per row.
    assert len(db.calls) == 1
    query = db.calls[0][0]
    assert "AS key_id" in query and "metadata->>'openapi_key_id'" in query
    assert "substr(" in query and "'openapi-%'" in query  # alias fallback when metadata is absent
    assert "token=l.owner" in query


@pytest.mark.asyncio
async def test_user_scope_still_enforced():
    db = FakeDB([{"total": 0, "items": []}])
    await list_logs(db, "alice", LogFilters(key_id="someone-elses-key"), datetime.now(timezone.utc))
    query, args = db.calls[0]
    # user_id is always the first, ANDed condition in both the page query and the count, key_id only narrows it.
    assert query.count("user_id=$1 AND started_at") == 2
    assert args[0] == "alice"
    client = _http_client(FakeDB([{"items": [], "total": 0}]))
    assert client.get("/internal/openapi-logs?key_id=k").status_code == 422  # user_id still required
    from litellm.proxy._types import LitellmUserRoles

    denied = _http_client(FakeDB([{"items": [], "total": 0}]), LitellmUserRoles.INTERNAL_USER)
    assert denied.get("/internal/openapi-logs?user_id=alice&key_id=k").status_code == 403


@pytest.mark.asyncio
async def test_key_filter_pagination_and_total_use_same_filter():
    db = FakeDB([{"total": 7, "items": []}])
    page = await list_logs(db, "alice", LogFilters(key_id="key-1", page=2, page_size=5), datetime.now(timezone.utc))
    query, args = db.calls[0]
    assert page.total == 7 and page.page == 2
    assert args[3] == "key-1" and args[-2:] == (5, 5)
    # Page selection and total count share the identical filtered WHERE clause.
    assert query.count("openapi_key_id") >= 4  # 2 branches x (selected + total) + row key_id derivation
    assert query.count("owner IN (") == 2
    assert f"LIMIT ${len(args) - 1} OFFSET ${len(args)}" in query


def test_http_key_id_query_param_reaches_database_and_is_optional():
    db = FakeDB([{"items": [], "total": 0}])
    client = _http_client(db)
    assert client.get("/internal/openapi-logs?user_id=alice&key_id=abc").status_code == 200
    assert "abc" in db.calls[-1][1] and "owner IN (" in db.calls[-1][0]
    db2 = FakeDB([{"items": [], "total": 0}])
    assert _http_client(db2).get("/internal/openapi-logs?user_id=alice").status_code == 200
    assert "owner IN (" not in db2.calls[-1][0]
    assert _http_client(db2).get("/internal/openapi-logs?user_id=alice&key_id=").status_code == 422


@pytest.mark.asyncio
async def test_row_key_id_lookup_prefers_live_token_deterministically():
    db = FakeDB([{"total": 0, "items": []}])
    await list_logs(db, "alice", LogFilters(), datetime.now(timezone.utc))
    query = db.calls[0][0]
    lookup = query[query.index("AS key_id") - 700 : query.index("AS key_id")]
    live = lookup.index('"LiteLLM_VerificationToken" WHERE token=l.owner')
    deleted = lookup.index('"LiteLLM_DeletedVerificationToken" WHERE token=l.owner')
    # Live key is source 0, revoked key source 1, and the UNION is explicitly ordered by source before LIMIT 1.
    assert "0 AS src" in lookup and "1 AS src" in lookup and live < deleted
    assert lookup.index("ORDER BY src") < lookup.index("LIMIT 1")


@pytest.mark.asyncio
async def test_row_key_id_ignores_empty_metadata_and_bare_alias_prefix():
    db = FakeDB([{"total": 0, "items": []}])
    await list_logs(db, "alice", LogFilters(), datetime.now(timezone.utc))
    query = db.calls[0][0]
    # Empty metadata falls back to the alias; an alias of exactly "openapi-" yields NULL, never ''.
    assert "nullif(k.metadata->>'openapi_key_id', '')" in query
    assert "nullif(CASE WHEN k.key_alias LIKE 'openapi-%' THEN substr(k.key_alias, 9) END, '')" in query
