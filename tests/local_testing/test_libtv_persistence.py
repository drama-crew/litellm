import hashlib
from typing import Any, List, Optional, Tuple

import httpx
import pytest

from litellm.llms.libtv import persistence
from litellm.llms.libtv.persistence import (
    LibTVPersistence,
    account_key,
    get_persistence,
    normalize_source_key,
    url_alive,
)


class FakeDb:
    def __init__(
        self,
        query_rows: Optional[List[dict]] = None,
        raise_on: Optional[str] = None,
    ):
        self.query_rows = query_rows if query_rows is not None else []
        self.raise_on = raise_on
        self.query_calls: List[Tuple[str, Tuple[Any, ...]]] = []
        self.execute_calls: List[Tuple[str, Tuple[Any, ...]]] = []

    async def query_raw(self, sql: str, *params: Any) -> List[dict]:
        self.query_calls.append((sql, params))
        if self.raise_on == "query_raw":
            raise RuntimeError("db unreachable")
        return self.query_rows

    async def execute_raw(self, sql: str, *params: Any) -> int:
        self.execute_calls.append((sql, params))
        if self.raise_on == "execute_raw":
            raise RuntimeError("db unreachable")
        return 1


@pytest.fixture(autouse=True)
def reset_tables_ready():
    persistence._tables_ready = False
    yield
    persistence._tables_ready = False


def test_normalize_source_key_url_strips_scheme_query_fragment():
    key = normalize_source_key("url", "https://cdn.example.com/path/to/file.png?sig=abc#frag", None)
    assert key == "cdn.example.com/path/to/file.png"


def test_normalize_source_key_bytes_is_sha1():
    data = b"hello world"
    key = normalize_source_key("bytes", "", data)
    assert key == f"sha1:{hashlib.sha1(data).hexdigest()}"


def test_normalize_source_key_garbage_returns_none():
    assert normalize_source_key("url", "", None) is None
    assert normalize_source_key("url", None, None) is None
    assert normalize_source_key("bytes", "", None) is None
    assert normalize_source_key("unknown", "https://x.com/y", None) is None


def test_account_key_stable_and_16_chars():
    k1 = account_key("token-abc")
    k2 = account_key("token-abc")
    k3 = account_key("token-xyz")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 16
    assert k1 == hashlib.sha1(b"token-abc").hexdigest()[:16]


@pytest.mark.asyncio
async def test_cached_upload_hit_returns_url_and_touches_last_used_at():
    db = FakeDb(query_rows=[{"cdn_url": "https://cdn.example.com/f.png"}])
    p = LibTVPersistence(db)

    result = await p.cached_upload("acct1", "cdn.example.com/f.png")

    assert result == "https://cdn.example.com/f.png"
    assert len(db.query_calls) == 1
    select_sql, select_params = db.query_calls[0]
    assert "SELECT" in select_sql
    assert select_params == ("acct1", "cdn.example.com/f.png")
    assert len(db.execute_calls) >= 1
    update_sql, update_params = db.execute_calls[-1]
    assert "UPDATE" in update_sql
    assert "last_used_at" in update_sql
    assert update_params == ("acct1", "cdn.example.com/f.png")


@pytest.mark.asyncio
async def test_cached_upload_miss_returns_none():
    db = FakeDb(query_rows=[])
    p = LibTVPersistence(db)

    result = await p.cached_upload("acct1", "cdn.example.com/missing.png")

    assert result is None


@pytest.mark.asyncio
async def test_cached_upload_touch_failure_does_not_fail_read():
    class TouchFailDb(FakeDb):
        async def execute_raw(self, sql: str, *params: Any) -> int:
            if "UPDATE" in sql:
                raise RuntimeError("touch failed")
            return await super().execute_raw(sql, *params)

    db = TouchFailDb(query_rows=[{"cdn_url": "https://cdn.example.com/f.png"}])
    p = LibTVPersistence(db)

    result = await p.cached_upload("acct1", "cdn.example.com/f.png")

    assert result == "https://cdn.example.com/f.png"


@pytest.mark.asyncio
async def test_store_upload_upserts():
    db = FakeDb()
    p = LibTVPersistence(db)

    await p.store_upload("acct1", "cdn.example.com/f.png", "https://cdn.example.com/f.png", 12345)

    upsert_calls = [c for c in db.execute_calls if "INSERT" in c[0]]
    assert len(upsert_calls) == 1
    sql, params = upsert_calls[0]
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql
    assert params == ("acct1", "cdn.example.com/f.png", "https://cdn.example.com/f.png", 12345)


@pytest.mark.asyncio
async def test_delete_upload_deletes():
    db = FakeDb()
    p = LibTVPersistence(db)

    await p.delete_upload("acct1", "cdn.example.com/f.png")

    delete_calls = [c for c in db.execute_calls if "DELETE" in c[0]]
    assert len(delete_calls) == 1
    sql, params = delete_calls[0]
    assert params == ("acct1", "cdn.example.com/f.png")


@pytest.mark.asyncio
async def test_cached_upload_query_raw_raises_returns_none():
    db = FakeDb(raise_on="query_raw")
    p = LibTVPersistence(db)

    result = await p.cached_upload("acct1", "cdn.example.com/f.png")

    assert result is None


@pytest.mark.asyncio
async def test_store_upload_execute_raw_raises_is_noop():
    db = FakeDb(raise_on="execute_raw")
    p = LibTVPersistence(db)

    result = await p.store_upload("acct1", "cdn.example.com/f.png", "https://cdn.example.com/f.png", 1)

    assert result is None


@pytest.mark.asyncio
async def test_delete_upload_execute_raw_raises_is_noop():
    db = FakeDb(raise_on="execute_raw")
    p = LibTVPersistence(db)

    result = await p.delete_upload("acct1", "cdn.example.com/f.png")

    assert result is None


@pytest.mark.asyncio
async def test_cached_upload_miss_does_not_touch_last_used_at():
    db = FakeDb(query_rows=[])
    p = LibTVPersistence(db)

    result = await p.cached_upload("acct1", "cdn.example.com/missing.png")

    assert result is None
    assert not any("UPDATE" in sql for sql, _ in db.execute_calls)


@pytest.mark.asyncio
async def test_ensure_tables_retries_after_failed_create():
    db = FakeDb(raise_on="execute_raw")
    p = LibTVPersistence(db)

    await p.ensure_tables()
    await p.ensure_tables()

    assert len(db.execute_calls) == 2
    assert persistence._tables_ready is False


@pytest.mark.asyncio
async def test_cached_project_hit_returns_project_uuid_and_team_id():
    db = FakeDb(query_rows=[{"project_uuid": "proj-1", "team_id": "7"}])
    p = LibTVPersistence(db)

    result = await p.cached_project("acct1", "2026-07-15")

    assert result == {"project_uuid": "proj-1", "team_id": "7"}
    assert len(db.query_calls) == 1
    select_sql, select_params = db.query_calls[0]
    assert "SELECT" in select_sql
    assert "LiteLLM_LibTVProjects" in select_sql
    assert select_params == ("acct1", "2026-07-15")


@pytest.mark.asyncio
async def test_cached_project_hit_with_null_team_id():
    db = FakeDb(query_rows=[{"project_uuid": "proj-1", "team_id": None}])
    p = LibTVPersistence(db)

    result = await p.cached_project("acct1", "2026-07-15")

    assert result == {"project_uuid": "proj-1", "team_id": None}


@pytest.mark.asyncio
async def test_cached_project_miss_returns_none():
    db = FakeDb(query_rows=[])
    p = LibTVPersistence(db)

    result = await p.cached_project("acct1", "2026-07-15")

    assert result is None


@pytest.mark.asyncio
async def test_cached_project_query_raw_raises_returns_none():
    db = FakeDb(raise_on="query_raw")
    p = LibTVPersistence(db)

    result = await p.cached_project("acct1", "2026-07-15")

    assert result is None


@pytest.mark.asyncio
async def test_store_project_inserts_with_do_nothing():
    db = FakeDb()
    p = LibTVPersistence(db)

    await p.store_project("acct1", "2026-07-15", "proj-1", "7")

    insert_calls = [c for c in db.execute_calls if "INSERT" in c[0]]
    assert len(insert_calls) == 1
    sql, params = insert_calls[0]
    assert "LiteLLM_LibTVProjects" in sql
    assert "ON CONFLICT" in sql
    assert "DO NOTHING" in sql
    assert params == ("acct1", "2026-07-15", "proj-1", "7")


@pytest.mark.asyncio
async def test_store_project_accepts_null_team_id():
    db = FakeDb()
    p = LibTVPersistence(db)

    await p.store_project("acct1", "2026-07-15", "proj-1", None)

    insert_calls = [c for c in db.execute_calls if "INSERT" in c[0]]
    assert insert_calls[0][1] == ("acct1", "2026-07-15", "proj-1", None)


@pytest.mark.asyncio
async def test_store_project_execute_raw_raises_is_noop():
    db = FakeDb(raise_on="execute_raw")
    p = LibTVPersistence(db)

    result = await p.store_project("acct1", "2026-07-15", "proj-1", "7")

    assert result is None


@pytest.mark.asyncio
async def test_invalidate_project_deletes():
    db = FakeDb()
    p = LibTVPersistence(db)

    await p.invalidate_project("acct1", "2026-07-15")

    delete_calls = [c for c in db.execute_calls if "DELETE" in c[0]]
    assert len(delete_calls) == 1
    sql, params = delete_calls[0]
    assert "LiteLLM_LibTVProjects" in sql
    assert params == ("acct1", "2026-07-15")


@pytest.mark.asyncio
async def test_invalidate_project_execute_raw_raises_is_noop():
    db = FakeDb(raise_on="execute_raw")
    p = LibTVPersistence(db)

    result = await p.invalidate_project("acct1", "2026-07-15")

    assert result is None


@pytest.mark.asyncio
async def test_ensure_tables_creates_both_tables_once():
    db = FakeDb()
    p = LibTVPersistence(db)

    await p.cached_upload("acct1", "cdn.example.com/f.png")
    calls_after_first = len(db.execute_calls)
    await p.cached_upload("acct1", "cdn.example.com/g.png")

    create_calls = [c for c in db.execute_calls if "CREATE TABLE" in c[0]]
    assert len(create_calls) == 2
    assert any("LiteLLM_LibTVUploadCache" in c[0] for c in create_calls)
    assert any("LiteLLM_LibTVProjects" in c[0] for c in create_calls)
    assert calls_after_first == 2
    assert len(db.execute_calls) == 2


def test_get_persistence_returns_none_when_prisma_client_none(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None, raising=False)

    assert get_persistence() is None


def test_get_persistence_returns_none_when_db_none(monkeypatch):
    from litellm.proxy import proxy_server

    class FakePrismaClient:
        db = None

    monkeypatch.setattr(proxy_server, "prisma_client", FakePrismaClient(), raising=False)

    assert get_persistence() is None


def test_get_persistence_returns_instance_when_db_present(monkeypatch):
    from litellm.proxy import proxy_server

    class FakePrismaClient:
        db = FakeDb()

    monkeypatch.setattr(proxy_server, "prisma_client", FakePrismaClient(), raising=False)

    result = get_persistence()

    assert isinstance(result, LibTVPersistence)


@pytest.mark.asyncio
async def test_url_alive_200_is_true(monkeypatch):
    async def fake_get(self, url, headers=None, follow_redirects=None, timeout=None):
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await url_alive("https://cdn.example.com/f.png") is True


@pytest.mark.asyncio
async def test_url_alive_206_is_true(monkeypatch):
    async def fake_get(self, url, headers=None, follow_redirects=None, timeout=None):
        return httpx.Response(206, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await url_alive("https://cdn.example.com/f.png") is True


@pytest.mark.asyncio
async def test_url_alive_404_is_false(monkeypatch):
    async def fake_get(self, url, headers=None, follow_redirects=None, timeout=None):
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await url_alive("https://cdn.example.com/f.png") is False


@pytest.mark.asyncio
async def test_url_alive_403_is_false(monkeypatch):
    async def fake_get(self, url, headers=None, follow_redirects=None, timeout=None):
        return httpx.Response(403, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await url_alive("https://cdn.example.com/f.png") is False


@pytest.mark.asyncio
async def test_url_alive_500_is_true(monkeypatch):
    async def fake_get(self, url, headers=None, follow_redirects=None, timeout=None):
        return httpx.Response(500, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await url_alive("https://cdn.example.com/f.png") is True


@pytest.mark.asyncio
async def test_url_alive_exception_is_true(monkeypatch):
    async def fake_get(self, url, headers=None, follow_redirects=None, timeout=None):
        raise httpx.ConnectTimeout("boom")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await url_alive("https://cdn.example.com/f.png") is True


@pytest.mark.asyncio
async def test_url_alive_uses_ranged_get_with_redirects(monkeypatch):
    captured = {}

    async def fake_get(self, url, headers=None, follow_redirects=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["follow_redirects"] = follow_redirects
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    await url_alive("https://cdn.example.com/f.png")

    assert captured["url"] == "https://cdn.example.com/f.png"
    assert captured["headers"] == {"Range": "bytes=0-0"}
    assert captured["follow_redirects"] is True
