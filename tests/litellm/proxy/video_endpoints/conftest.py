from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.libtv_video_endpoints import endpoints as endpoints_module


class FakeRedis:
    """Minimal in-memory stand-in for the subset of redis.asyncio.Redis used
    by litellm.llms.libtv.video_generate.

    Every call is recorded to `.calls` as (op_name, key, kwargs) so tests can
    assert write ordering and exact kwargs (e.g. that SET happens before
    XADD and carries nx=True + ex=<ttl>).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.set_nx_result: bool = True
        self.alive: list[str] = ["vw-default"]
        self.capacity: dict[str, int] = {"vw-default": 1}
        self.status: dict[str, str] = {}
        self.result: dict[str, str] = {}
        # F2 regression fixtures: let a test force XADD to raise (simulating
        # a network blip strictly after the SET NX status write already
        # landed) and separately force the best-effort rollback DELETE to
        # raise too, so we can assert the *original* enqueue error survives
        # a failing rollback instead of being masked by it.
        self.xadd_exception: Optional[Exception] = None
        self.delete_exception: Optional[Exception] = None
        self.deleted_keys: list[str] = []

    @staticmethod
    def _task_id_from_key(key: str) -> str:
        return key.rsplit(":", 1)[-1]

    async def set(
        self, key: str, value: Any, *, ex: Optional[int] = None, nx: Optional[bool] = None
    ) -> bool:
        self.calls.append(("set", key, {"value": value, "ex": ex, "nx": nx}))
        if nx:
            return self.set_nx_result
        return True

    async def xadd(self, stream: str, fields: dict) -> str:
        self.calls.append(("xadd", stream, dict(fields)))
        if self.xadd_exception is not None:
            raise self.xadd_exception
        return "0-1"

    async def delete(self, key: str) -> int:
        self.calls.append(("delete", key, {}))
        if self.delete_exception is not None:
            raise self.delete_exception
        self.deleted_keys.append(key)
        return 1

    async def zrangebyscore(self, key: str, min_score: Any, max_score: Any) -> list:
        self.calls.append(("zrangebyscore", key, {"min": min_score, "max": max_score}))
        return list(self.alive)

    async def hgetall(self, key: str) -> dict:
        self.calls.append(("hgetall", key, {}))
        return dict(self.capacity)

    async def get(self, key: str) -> Optional[str]:
        self.calls.append(("get", key, {}))
        return self.status.get(self._task_id_from_key(key))

    async def lindex(self, key: str, index: int) -> Optional[str]:
        self.calls.append(("lindex", key, {"index": index}))
        return self.result.get(self._task_id_from_key(key))


@pytest.fixture(autouse=True)
def _video_generate_env(monkeypatch):
    """Default fail-open (i.e. non-empty, valid) allowlist env vars so tests
    that don't care about allowlist configuration still exercise the happy
    path. VALID's reference/staging_upload URLs use these exact hostnames.
    """
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_SOURCE_HOSTS", "source.example")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_TARGET_HOSTS", "target.example")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_ALLOW_HTTP", "false")


@pytest.fixture
def fake_redis(monkeypatch) -> FakeRedis:
    """A fresh FakeRedis, wired in by monkeypatching the endpoint module's
    `_get_redis_client` factory. `_get_redis_client()` is called at
    request-handling time (not at app-construction time), so it's enough
    for this monkeypatch to be applied before the test body issues its
    first request -- pytest guarantees that for any fixture listed as a
    test parameter, regardless of whether `client_admin`/`client_user`
    declare an explicit dependency on this fixture.
    """
    redis = FakeRedis()
    monkeypatch.setattr(endpoints_module, "_get_redis_client", lambda: redis)
    return redis


def _build_app(user: UserAPIKeyAuth) -> FastAPI:
    app = FastAPI()
    app.include_router(endpoints_module.router)
    app.dependency_overrides[user_api_key_auth] = lambda: user
    return app


@pytest.fixture
async def client_admin():
    app = _build_app(UserAPIKeyAuth(user_role="proxy_admin", user_id="admin"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def client_user():
    app = _build_app(UserAPIKeyAuth(user_id="user-1"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
