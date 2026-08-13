from __future__ import annotations

import orjson
import pytest
from starlette.requests import Request

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.image_endpoints import endpoints


def _request(body: dict) -> Request:
    raw = orjson.dumps(body)

    async def receive():
        return {"type": "http.request", "body": raw}

    return Request({"type": "http", "method": "POST", "path": "/v1/libtv/validated-media-transfer", "headers": []}, receive=receive)


def _raw_request(raw: bytes) -> Request:
    async def receive():
        return {"type": "http.request", "body": raw}

    return Request({"type": "http", "method": "POST", "path": "/v1/libtv/validated-media-transfer", "headers": []}, receive=receive)


@pytest.mark.asyncio
async def test_validated_transfer_endpoint_requires_proxy_admin():
    response = await endpoints.libtv_validated_media_transfer(
        _request({}), UserAPIKeyAuth(team_id="team-1", user_id="user-1")
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_validated_transfer_endpoint_maps_validation_failure_to_422(monkeypatch):
    class Router:
        async def execute(self, request):
            from litellm.llms.libtv.validated_transfer import ValidatedTransferError
            raise ValidatedTransferError("bad image")

    monkeypatch.setattr(endpoints, "get_validated_transfer_router", lambda: Router())
    response = await endpoints.libtv_validated_media_transfer(
        _request({"type": "validated_media_transfer", "mode": "probe", "source": {"url": "https://source.example/image.png"}, "hard_cap": 10}),
        UserAPIKeyAuth(user_role="proxy_admin", user_id="admin"),
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_validated_transfer_readiness_requires_proxy_admin(monkeypatch):
    class Router:
        executor = type("Executor", (), {"ready": lambda self: True})()

    monkeypatch.setattr(endpoints, "get_validated_transfer_router", lambda: Router())
    response = await endpoints.libtv_validated_media_transfer_readiness(UserAPIKeyAuth(user_role="proxy_admin", user_id="admin"))
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_validated_transfer_endpoint_keeps_route_metadata_on_success(monkeypatch):
    class Router:
        async def execute(self, request):
            return {"validation_version": "image-v1", "bytes": 1, "mime": "image/png", "width": 1, "height": 1, "sha256": "a" * 64, "route": "local_no_worker"}

    monkeypatch.setattr(endpoints, "get_validated_transfer_router", lambda: Router())
    response = await endpoints.libtv_validated_media_transfer(
        _request({"type": "validated_media_transfer", "mode": "probe", "source": {"url": "https://source.example/image.png"}, "hard_cap": 10}),
        UserAPIKeyAuth(user_role="proxy_admin", user_id="admin"),
    )
    assert response.status_code == 200
    assert orjson.loads(response.body)["route"] == "local_no_worker"


@pytest.mark.asyncio
async def test_validated_transfer_endpoint_maps_transient_failure_to_503(monkeypatch):
    class Router:
        async def execute(self, request):
            from litellm.llms.libtv.validated_transfer import ValidatedTransferError
            raise ValidatedTransferError("capacity", validation=False)

    monkeypatch.setattr(endpoints, "get_validated_transfer_router", lambda: Router())
    response = await endpoints.libtv_validated_media_transfer(
        _request({"type": "validated_media_transfer", "mode": "probe", "source": {"url": "https://source.example/image.png"}, "hard_cap": 10}),
        UserAPIKeyAuth(user_role="proxy_admin", user_id="admin"),
    )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_validated_transfer_endpoint_rejects_invalid_json_for_proxy_admin():
    response = await endpoints.libtv_validated_media_transfer(
        _raw_request(b"{"), UserAPIKeyAuth(user_role="proxy_admin", user_id="admin")
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_validated_transfer_readiness_reports_unconfigured_as_503(monkeypatch):
    def unavailable():
        from litellm.llms.libtv.validated_transfer import ValidatedTransferError
        raise ValidatedTransferError("missing config", validation=False)

    monkeypatch.setattr(endpoints, "get_validated_transfer_router", unavailable)
    response = await endpoints.libtv_validated_media_transfer_readiness(
        UserAPIKeyAuth(user_role="proxy_admin", user_id="admin")
    )
    assert response.status_code == 503
