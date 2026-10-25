"""Public ``/v1/causyn_files`` routes: thin, authenticated proxies over the platform's /internal/moderation/files."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.video_endpoints import causyn_files_endpoints
from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3
from litellm.proxy.video_endpoints.minimax_h3_paths import DIRECT_PREFIX, IR_PREFIX, TEST_PREFIXES

PREFIXES = (IR_PREFIX, DIRECT_PREFIX)
HEADERS = {"Authorization": "Bearer allowed"}
FILE_ID = "cf_" + "a1b2c3d4e5f6g7h8i9j0k1l2m3"[:26]
VIEW = {
    "file_id": FILE_ID,
    "kind": "video",
    "filename": "clip.mp4",
    "bytes": 1234,
    "content_type": "video/mp4",
    "status": "ready",
    "created_at": 1_700_000_000,
    "expires_at": 1_700_600_000,
}
UPLOAD = {
    "file_id": FILE_ID,
    "upload": {"url": "https://bucket.oss-cn-wulanchabu.aliyuncs.com", "fields": {"key": "k", "policy": "p"}},
    "expires_at": 1_700_000_300,
    "max_bytes": 52428800,
}
OK = {"status_code": 0, "status_msg": "success"}


class Platform:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.reply: tuple[int, dict] | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/internal/moderation")
        self.calls.append((path, json.loads(request.content)))
        if self.reply is not None:
            return httpx.Response(self.reply[0], json=self.reply[1])
        if path == "/files/uploads":
            return httpx.Response(200, json=UPLOAD)
        if path == "/files/list":
            return httpx.Response(200, json={"items": [VIEW], "next": None})
        if path.endswith("/delete"):
            return httpx.Response(200, json={"file_id": FILE_ID, "deleted": True})
        return httpx.Response(200, json=VIEW)


@pytest.fixture
def api(monkeypatch):
    platform = Platform()
    seen: list[dict] = []
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", "synthetic-test-secret")

    async def auth(request: Request):
        if request.headers.get("authorization") != "Bearer allowed":
            raise ProxyException(message="Invalid API key", type="authentication_error", param=None, code=401)
        from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

        seen.append(dict(await _read_request_body(request)))
        return UserAPIKeyAuth(
            api_key="sk-owner", user_id="user", team_id="team", metadata={"openapi_key_id": "key"}
        )

    app = FastAPI()
    app.state.moderation_transport = httpx.MockTransport(platform.handle)
    app.dependency_overrides[h3.user_api_key_auth] = auth
    app.include_router(causyn_files_endpoints.router)
    return TestClient(app), platform, seen, app


@pytest.mark.parametrize("prefix", PREFIXES)
def test_upload_proxies_to_platform_and_adds_base_resp(api, prefix):
    client, platform, seen, _ = api
    body = {"kind": "video", "filename": "clip.mp4", "size": 1234, "content_type": "video/mp4"}
    response = client.post(prefix + "/v1/causyn_files/uploads", json=body, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.json() == {**UPLOAD, "base_resp": OK}
    (path, sent), = platform.calls
    assert path == "/files/uploads"
    assert sent["principal"]["user_id"] == "user" and sent["principal"]["openapi_key_id"] == "key"
    assert {key: sent[key] for key in body} == body
    assert seen[0]["model"] == "causyn-1.1"  # the placeholder body keeps the key's model gate live


@pytest.mark.parametrize("prefix", PREFIXES)
def test_complete_get_list_delete_proxy_with_base_resp(api, prefix):
    client, platform, _, _ = api
    base = prefix + "/v1/causyn_files"
    assert client.post(f"{base}/{FILE_ID}/complete", json={}, headers=HEADERS).json() == {**VIEW, "base_resp": OK}
    assert client.get(f"{base}/{FILE_ID}", headers=HEADERS).json() == {**VIEW, "base_resp": OK}
    listed = client.get(f"{base}?limit=5&after=cf_x", headers=HEADERS)
    assert listed.json() == {"items": [VIEW], "next": None, "base_resp": OK}
    deleted = client.delete(f"{base}/{FILE_ID}", headers=HEADERS)
    assert deleted.json() == {"file_id": FILE_ID, "deleted": True, "base_resp": OK}
    assert [path for path, _ in platform.calls] == [
        f"/files/{FILE_ID}/complete",
        f"/files/{FILE_ID}/view",
        "/files/list",
        f"/files/{FILE_ID}/delete",
    ]
    assert platform.calls[2][1]["limit"] == 5 and platform.calls[2][1]["after"] == "cf_x"


def test_list_defaults_and_rejects_bad_query(api):
    client, platform, _, _ = api
    base = IR_PREFIX + "/v1/causyn_files"
    assert client.get(base, headers=HEADERS).status_code == 200
    assert platform.calls[0][1]["limit"] == 20 and platform.calls[0][1]["after"] is None
    for query in ("?limit=0", "?limit=101", "?limit=x", "?bogus=1"):
        assert client.get(base + query, headers=HEADERS).status_code == 400, query
    assert len(platform.calls) == 1


def test_requires_auth_and_validates_the_upload_body(api):
    client, platform, _, _ = api
    route = IR_PREFIX + "/v1/causyn_files/uploads"
    assert client.post(route, json={}).status_code == 401
    for bad in (
        {"kind": "pdf", "filename": "a", "size": 1, "content_type": "x/y"},
        {"kind": "image", "filename": "", "size": 1, "content_type": "image/png"},
        {"kind": "image", "filename": "a.png", "size": 0, "content_type": "image/png"},
        {"kind": "image", "filename": "a.png", "size": "5", "content_type": "image/png"},
        {"kind": "image", "filename": "a.png", "size": 5, "content_type": "image/png", "extra": 1},
    ):
        assert client.post(route, json=bad, headers=HEADERS).status_code == 400, bad
    assert platform.calls == []


def test_malformed_file_id_is_404_without_reaching_the_platform(api):
    client, platform, _, _ = api
    for bad in ("nope", "cf_SHORT", "cf_" + "A" * 26, "cf_" + "a" * 27):
        response = client.get(IR_PREFIX + "/v1/causyn_files/" + bad, headers=HEADERS)
        assert response.status_code == 404, bad
        assert response.json()["error"]["code"] == "file_not_found"
    assert platform.calls == []


def test_get_maps_404(api):
    client, platform, _, _ = api
    platform.reply = (404, {"detail": {"code": "file_not_found"}})
    response = client.get(IR_PREFIX + f"/v1/causyn_files/{FILE_ID}", headers=HEADERS)
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "file_not_found" and error["type"] == "not_found_error"


@pytest.mark.parametrize("code", ["file_rejected", "file_not_ready", "file_expired", "invalid_upload"])
def test_400_errors_carry_the_platform_reason(api, code):
    client, platform, _, _ = api
    platform.reply = (400, {"detail": {"code": code, "reason": "duration 20s exceeds the limit"}})
    response = client.post(IR_PREFIX + f"/v1/causyn_files/{FILE_ID}/complete", json={}, headers=HEADERS)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == code and "duration 20s exceeds the limit" in error["message"]


@pytest.mark.parametrize("limit", ["files", "bytes", "pending", "rate"])
def test_quota_429_maps_with_limit(api, limit):
    client, platform, _, _ = api
    platform.reply = (429, {"detail": {"code": "quota_exceeded", "limit": limit}})
    body = {"kind": "image", "filename": "a.png", "size": 10, "content_type": "image/png"}
    response = client.post(IR_PREFIX + "/v1/causyn_files/uploads", json=body, headers=HEADERS)
    assert response.status_code == 429
    error = response.json()["error"]
    assert error["code"] == "quota_exceeded" and limit in error["message"]


def test_platform_5xx_is_503(api):
    client, platform, _, _ = api
    platform.reply = (500, {"detail": "boom"})
    response = client.get(IR_PREFIX + f"/v1/causyn_files/{FILE_ID}", headers=HEADERS)
    assert response.status_code == 503 and "boom" not in response.text


def test_files_routes_not_under_test_prefix(api):
    _, _, _, app = api
    paths = [route.path for route in app.routes if isinstance(route, APIRoute)]
    assert paths and all("causyn_files" in path for path in paths)
    assert not [path for path in paths for prefix in TEST_PREFIXES if path.startswith(prefix)]
    assert {path.split("/v1/")[0] for path in paths} == set(PREFIXES)


def test_response_never_contains_oss_scheme(api):
    client, platform, _, _ = api
    base = IR_PREFIX + "/v1/causyn_files"
    body = {"kind": "video", "filename": "clip.mp4", "size": 1234, "content_type": "video/mp4"}
    responses = [
        client.post(base + "/uploads", json=body, headers=HEADERS),
        client.get(f"{base}/{FILE_ID}", headers=HEADERS),
        client.get(base, headers=HEADERS),
        client.delete(f"{base}/{FILE_ID}", headers=HEADERS),
    ]
    platform.reply = (400, {"detail": {"code": "file_rejected", "reason": "bad"}})
    responses.append(client.get(f"{base}/{FILE_ID}", headers=HEADERS))
    assert all("oss://" not in response.text for response in responses)


def test_files_routes_are_llm_api_routes_so_ordinary_keys_reach_them():
    from litellm.proxy._types import LiteLLMRoutes

    llm = set(LiteLLMRoutes.llm_api_routes.value)
    for prefix in PREFIXES:
        for suffix in ("/uploads", "/{file_id}/complete", "/{file_id}", ""):
            assert prefix + "/v1/causyn_files" + suffix in llm
    assert not [route for route in llm if "minimax-h3-test" in route and "causyn_files" in route]
