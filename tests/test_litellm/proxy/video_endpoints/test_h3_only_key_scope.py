"""Admin-issued H3-only keys: ``models=["causyn-1.1"]`` + the ``h3_only`` route profile.

The platform (OpenHands enterprise ``server.routes.openapi_keys``) provisions these
keys through LiteLLM ``/key/generate`` with ``models`` and ``allowed_routes`` only
-- there is no LiteLLM-side enforcement code for them. These tests pin the
contract that makes that safe, against the real auth chain:

* an H3-only key reaches the H3 create / query / list / Context IR routes
  (authorization passes; whatever the harness returns afterwards is not 401/403);
* the same key is rejected (403) on ``/v1/videos`` / ``/videos``: for
  ``model=seedance-2.0-fast`` by the model check (it runs first), and for the permitted
  ``causyn-1.1`` by the route allowlist -- so neither model nor route can be used to
  reach the generic video API;
* a key limited to ``causyn-1.1`` but granted *broad* routes is still rejected for
  seedance by the model check, so the model restriction is independently enforced.
"""

# ruff: noqa: F811  (the shared ``stack`` fixture is imported, then requested by name)

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from litellm.proxy._types import LiteLLM_VerificationTokenView, UserAPIKeyAuth, hash_token
from litellm.proxy.video_endpoints import minimax_h3_endpoints  # noqa: F401  (routes under test)

import litellm.proxy.proxy_server as proxy_server

from tests.test_litellm.proxy.video_endpoints.test_causyn_public_stack import (  # noqa: F401
    _auth_headers,
    stack,
)

# Mirrors OpenHands enterprise ``server.routes.openapi_keys.H3_ONLY_ROUTES``
# (namespaced create / query / list / delete on both prefixes + Context IR).
_H3_SUFFIXES = (
    "/v2/video_generation",
    "/v2/query/video_generation/{video_id}",
    "/v2/query/video_generation",
    "/v2/video_generation/{video_id}",
)
H3_ONLY_ROUTES = [
    prefix + suffix for prefix in ("/video/minimax-h3", "/video/minimax-h3/direct") for suffix in _H3_SUFFIXES
] + ["/video/minimax-h3/v2/h3_context_ir"]

# Mirrors the self-service ("all") default grant for the video routes.
BROAD_ROUTES = [
    "/v1/models",
    "/models",
    "/videos",
    "/v1/videos",
    "/videos/{video_id}",
    "/v1/videos/{video_id}",
    "/videos/{video_id}/content",
    "/v1/videos/{video_id}/content",
    *H3_ONLY_ROUTES,
]

SERVICE_SECRET = "synthetic-service-signature-secret"
H3_ONLY_KEY = "sk-h3-only"
MODEL_ONLY_KEY = "sk-causyn-model-only-broad-routes"
SEEDANCE = "seedance-2.0-fast"
OTHER_KEY = "sk-other-model"


class _ScopedStore:
    """Virtual-key rows shaped like what ``/key/generate`` stores for admin keys."""

    def __init__(self) -> None:
        self.rows = {
            hash_token(H3_ONLY_KEY): {"models": ["causyn-1.1"], "allowed_routes": list(H3_ONLY_ROUTES)},
            hash_token(MODEL_ONLY_KEY): {"models": ["causyn-1.1"], "allowed_routes": list(BROAD_ROUTES)},
            # Control: an ordinary key for a different model, never allowed on these routes.
            hash_token(OTHER_KEY): {"models": ["other-model"], "allowed_routes": []},
        }

    async def get_data(self, *, token: str, table_name: str, **_: object):
        assert table_name == "combined_view"
        row = self.rows.get(token)
        if row is None:
            return None
        # No team_id: team budget/wallet resolution needs the full proxy cache stack, which the shared
        # harness stubs out; the model and route checks under test do not depend on it.
        return LiteLLM_VerificationTokenView(
            token=token,
            user_id="owner",
            team_id="owner",
            metadata={"type": "admin_openapi", "openapi_key_id": "11111111-1111-1111-1111-111111111111"},
            **row,
        )


def _stub_owner_lookups(monkeypatch) -> None:
    """The owner's personal team / user rows live in Postgres; the public bridge needs both ids to be set."""
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable
    from litellm.proxy.auth import user_api_key_auth as auth_module

    async def team(*args, **kwargs):
        return LiteLLM_TeamTableCachedObj(team_id="owner")

    async def user(*args, **kwargs):
        return LiteLLM_UserTable(user_id="owner", max_budget=None)

    logging_obj = proxy_server.proxy_logging_obj
    monkeypatch.setattr(logging_obj, "service_logging_obj", AsyncMock(), raising=False)
    monkeypatch.setattr(logging_obj, "internal_usage_cache", SimpleNamespace(dual_cache=None), raising=False)
    monkeypatch.setattr(auth_module, "get_team_object", team)
    monkeypatch.setattr(auth_module, "get_user_object", user)


@pytest.fixture
def scoped(stack, monkeypatch):
    from litellm.proxy.video_endpoints import context_ir_endpoints

    monkeypatch.setattr(proxy_server, "prisma_client", _ScopedStore())
    _stub_owner_lookups(monkeypatch)
    stack.app.include_router(context_ir_endpoints.router)
    return stack


def _h3_create_body() -> dict[str, object]:
    return {
        "model": "minimax-h3",
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
        "content": [
            {"type": "text", "text": "Gentle waves."},
            {"type": "image_url", "role": "reference_image", "image_url": {"url": "https://source.example/0.png"}},
        ],
    }


class _Platform:
    """In-process stand-in for the moderation control plane the public bridge calls after authorization."""

    VIEW = {
        "id": "mod_video_1",
        "state": "queued",
        "model": "causyn-1.1",
        "moderation_status": "pending",
        "policy_source": "global",
        "created_at": 1_700_000_000,
    }

    def __init__(self, scoped) -> None:
        self.calls: list[dict] = []
        scoped.app.state.moderation_transport = httpx.MockTransport(self.handle)

    def handle(self, request):
        path = request.url.path.removeprefix("/internal/moderation")
        body = json.loads(request.content) if request.content else {}
        self.calls.append({"path": path, "body": body})
        if path.endswith("/cancel"):
            return httpx.Response(200, json={"action": "cancelled"})
        if path == "/intents/list":
            return httpx.Response(200, json={"items": [self.VIEW], "total": 1})
        return httpx.Response(200, json=self.VIEW)

    def only(self, path: str) -> dict:
        (call,) = [call for call in self.calls if call["path"] == path]
        return call["body"]


@pytest.fixture
def platform(scoped, monkeypatch):
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SERVICE_SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    return _Platform(scoped)


def _assert_principal(body: dict) -> None:
    principal = body["principal"]
    assert principal["user_id"] == principal["team_id"] == "owner"
    assert principal["openapi_key_id"] == "11111111-1111-1111-1111-111111111111"
    assert principal["fingerprint"] == hash_token(H3_ONLY_KEY)


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize(
    "route,namespace",
    [
        ("/video/minimax-h3/v2/video_generation", "minimax-h3"),
        ("/video/minimax-h3/direct/v2/video_generation", "minimax-h3-direct"),
    ],
)
def test_h3_only_key_creates_an_h3_task(scoped, platform, route, namespace):
    body = _h3_create_body()
    denied = scoped.client.post(route, json=body, headers=_auth_headers(OTHER_KEY))
    assert denied.status_code == 403 and "not allowed" in denied.text, denied.text  # control: the gate is live
    assert platform.calls == []
    created = scoped.client.post(route, json=body, headers=_auth_headers(H3_ONLY_KEY))
    assert created.status_code == 200, created.text
    assert created.json() == {"task_id": "mod_video_1"}
    intake = platform.only("/intents")
    _assert_principal(intake)
    assert (
        intake["namespace"] == namespace and intake["model"] == "causyn-1.1" and intake["route"] == "avideo_generation"
    )
    assert not scoped.state.enqueued  # nothing is submitted to the provider before the moderated continuation


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize(
    "prefix,namespace", [("/video/minimax-h3", "minimax-h3"), ("/video/minimax-h3/direct", "minimax-h3-direct")]
)
def test_h3_only_key_queries_lists_and_cancels_h3_tasks(scoped, platform, prefix, namespace):
    task = prefix + "/v2/query/video_generation/mod_video_1"
    listing = prefix + "/v2/query/video_generation"
    cancel = prefix + "/v2/video_generation/mod_video_1"
    for method, route in (("get", task), ("get", listing), ("delete", cancel)):
        denied = getattr(scoped.client, method)(route, headers=_auth_headers(OTHER_KEY))
        assert denied.status_code == 403 and "not allowed" in denied.text, (route, denied.text)  # control
    assert platform.calls == []

    got = scoped.client.get(task, headers=_auth_headers(H3_ONLY_KEY))
    assert got.status_code == 200, got.text
    assert got.json()["task"]["id"] == "mod_video_1" and got.json()["task"]["model"] == "MiniMax-H3"
    view = platform.only("/intents/mod_video_1/view")
    _assert_principal(view)
    assert view["namespace"] == namespace

    listed = scoped.client.get(listing, headers=_auth_headers(H3_ONLY_KEY))
    assert listed.status_code == 200, listed.text
    assert [item["id"] for item in listed.json()["items"]] == ["mod_video_1"] and listed.json()["total"] == 1
    _assert_principal(platform.only("/intents/list"))

    cancelled = scoped.client.delete(cancel, headers=_auth_headers(H3_ONLY_KEY))
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json() == {"task_id": "mod_video_1", "action": "cancelled", "status": "cancelled"}
    _assert_principal(platform.only("/intents/mod_video_1/cancel"))


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
def test_h3_only_key_passes_authorization_on_h3_context_ir(scoped, platform):
    """Context IR reserves metering budget after authorization (durable SQL/Redis, absent from this harness), so
    the strongest end-to-end observable here is: authorization passes and the request is validated (400 on a
    wrong model), whereas the control key is denied before any validation."""
    route = "/video/minimax-h3/v2/h3_context_ir"
    body = {"model": "causyn-1.1", "prompt": "kite"}
    denied = scoped.client.post(route, json=body, headers=_auth_headers(OTHER_KEY))
    assert denied.status_code == 403 and "not allowed" in denied.text, denied.text
    allowed = scoped.client.post(route, json=body, headers=_auth_headers(H3_ONLY_KEY))
    assert allowed.status_code == 400 and "model must be MiniMax-H3" in allowed.text, allowed.text
    assert platform.calls == []


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize(
    "route", ["/video/minimax-h3/v2/video_generation", "/video/minimax-h3/direct/v2/video_generation"]
)
def test_key_budget_exhaustion_is_a_public_402_insufficient_balance(scoped, platform, route):
    """The platform enforces the hard per-key credit cap and answers 402/key_budget_exceeded at intake."""

    def exhausted(request):
        platform.calls.append({"path": request.url.path})
        return httpx.Response(
            402,
            json={"detail": {"code": "key_budget_exceeded", "message": "The API key credit budget is exhausted"}},
        )

    scoped.app.state.moderation_transport = httpx.MockTransport(exhausted)
    result = scoped.client.post(route, json=_h3_create_body(), headers=_auth_headers(H3_ONLY_KEY))
    assert result.status_code == 402, result.text
    error = result.json()["error"]
    assert error["type"] == "insufficient_balance_error"
    assert error["code"] == "key_budget_exceeded"
    assert error["http_code"] == "402" and error["retryable"] is False
    assert "budget" in error["message"].lower()
    assert not scoped.state.enqueued


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
def test_key_budget_exhaustion_on_the_generic_video_api_keeps_the_402(scoped, platform):
    def exhausted(request):
        return httpx.Response(402, json={"detail": {"code": "key_budget_exceeded", "message": "budget"}})

    scoped.app.state.moderation_transport = httpx.MockTransport(exhausted)
    body = {"model": "causyn-1.1", "prompt": "kite", "seconds": "5", "size": "768p", "aspect_ratio": "16:9"}
    result = scoped.client.post("/v1/videos", json=body, headers=_auth_headers(MODEL_ONLY_KEY))
    assert result.status_code == 402, result.text
    assert "budget" in result.text.lower()


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos", "/videos"])
def test_h3_only_key_is_rejected_for_seedance_on_v1_videos(scoped, route):
    body = {"model": SEEDANCE, "prompt": "kite", "seconds": "5", "size": "720p", "aspect_ratio": "16:9"}
    result = scoped.client.post(route, json=body, headers=_auth_headers(H3_ONLY_KEY))
    assert result.status_code == 403, result.text
    assert "key not allowed to access model" in result.text and SEEDANCE in result.text
    assert not scoped.state.enqueued


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos", "/videos"])
def test_h3_only_key_is_rejected_on_v1_videos_even_for_its_own_model(scoped, route):
    """The permitted model cannot use the generic video API: the route allowlist denies it."""
    body = {"model": "causyn-1.1", "prompt": "kite", "seconds": "5", "size": "768p", "aspect_ratio": "16:9"}
    result = scoped.client.post(route, json=body, headers=_auth_headers(H3_ONLY_KEY))
    assert result.status_code == 403, result.text
    assert "not allowed to call this route" in result.text
    assert not scoped.state.enqueued


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos/some_video_id", "/v1/videos/some_video_id/content"])
def test_h3_only_key_is_rejected_on_other_video_routes(scoped, route):
    result = scoped.client.get(route, headers=_auth_headers(H3_ONLY_KEY))
    assert result.status_code == 403, result.text


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos", "/videos"])
def test_model_restricted_key_with_broad_routes_is_still_rejected_for_seedance(scoped, platform, route):
    """Routes allow /v1/videos, but ``models=["causyn-1.1"]`` blocks seedance on its own."""
    body = {"model": SEEDANCE, "prompt": "kite", "seconds": "5", "size": "720p", "aspect_ratio": "16:9"}
    result = scoped.client.post(route, json=body, headers=_auth_headers(MODEL_ONLY_KEY))
    assert result.status_code == 403, result.text
    assert "key not allowed to access model" in result.text and SEEDANCE in result.text
    assert "not allowed to call this route" not in result.text  # rejected by the model check alone
    assert not scoped.state.enqueued
    assert platform.calls == []
    # control: the same broad-route key is let through for the permitted model and reaches the platform intake
    ok = scoped.client.post(
        route,
        json={"model": "causyn-1.1", "prompt": "kite", "seconds": "5", "size": "768p", "aspect_ratio": "16:9"},
        headers=_auth_headers(MODEL_ONLY_KEY),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["id"] == "mod_video_1"
    assert platform.only("/intents")["model"] == "causyn-1.1"


# --- the moderation continuation re-authenticates the stored key on the ORIGINAL public route -------------------


from fastapi import FastAPI  # noqa: E402

from litellm.proxy.video_endpoints import moderation_bridge as bridge  # noqa: E402
from litellm.proxy.video_endpoints import moderation_execution as execution  # noqa: E402
from litellm.types.videos.main import VideoObject  # noqa: E402

TEST_POOL_KEY = "sk-h3-only-test-pool"
LEGACY_KEY = "sk-self-service-legacy"
V1_ONLY_KEY = "sk-v1-videos-only"
TEST_POOL_ROUTES = [
    prefix + suffix for prefix in ("/video/minimax-h3-test", "/video/minimax-h3-test/direct") for suffix in _H3_SUFFIXES
] + ["/video/minimax-h3-test/v2/h3_context_ir"]
LEGACY_ROUTES = ["/v1/models", "/models", "/videos", "/v1/videos", "/videos/{video_id}", "/v1/videos/{video_id}"]


class _ContinuationStore(_ScopedStore):
    def __init__(self) -> None:
        super().__init__()
        self.rows.update(
            {
                hash_token(TEST_POOL_KEY): {
                    "models": ["causyn-1.1"],
                    "allowed_routes": [*H3_ONLY_ROUTES, *TEST_POOL_ROUTES],
                },
                # Pre-profile self-service key: only the generic video API, no namespaced H3 route.
                hash_token(LEGACY_KEY): {"models": [], "allowed_routes": list(LEGACY_ROUTES)},
                hash_token(V1_ONLY_KEY): {"models": ["causyn-1.1"], "allowed_routes": ["/v1/videos"]},
            }
        )


def _service_ticket() -> str:
    import time

    import jwt

    return jwt.encode(
        {
            "intent_id": "intent",
            "request_digest": "a" * 64,
            "input_digest": "b" * 64,
            "model": "causyn-1.1",
            "policy_version": "v1",
            "policy_digest": "c" * 64,
            "purpose": "submit",
            "aud": "moderation-fork",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        SERVICE_SECRET,
        algorithm="HS256",
    )


async def _continue(
    stack, monkeypatch, *, credential, namespace, route="avideo_generation", direct=False, payload=None
):
    """POST /internal/moderation/submit with the REAL authenticate(); only the provider call is stubbed."""
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SERVICE_SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setattr(proxy_server, "prisma_client", _ContinuationStore())
    _stub_owner_lookups(monkeypatch)
    calls: list[dict] = []
    seen: dict[str, object] = {}

    async def platform(request):
        calls.append({"path": request.url.path, "body": json.loads(request.content)})
        if request.url.path.endswith("/begin"):
            body_payload = payload or {"model": "causyn-1.1", "prompt": "kite", "seconds": "5", "aspect_ratio": "16:9"}
            if direct:
                body_payload["prompt_processing"] = "direct"
            body = {
                "acquired": True,
                "token": "attempt",
                "metering": {
                    "intent_id": "intent",
                    "request_digest": "a" * 64,
                    "actor_user_id": "actor",
                    "model": "causyn-1.1",
                },
                "credential": credential,
                "route": route,
                "request": body_payload,
            }
            if namespace is not None:
                body["namespace"] = namespace
            return httpx.Response(200, json=body)
        return httpx.Response(200, json={})

    async def invoke(request, auth, payload, route_name):
        seen["path"] = request.url.path
        seen["models"] = auth.models
        return VideoObject(id="native-1", object="video", status="queued")

    monkeypatch.setattr(execution, "invoke", invoke)
    monkeypatch.setattr(bridge, "capture", AsyncMock())
    app = FastAPI()
    app.include_router(execution.router)
    app.state.moderation_transport = httpx.MockTransport(platform)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fork"
    ) as client:
        response = await client.post(
            "/internal/moderation/submit",
            json={"ticket": _service_ticket()},
            headers={"Authorization": "Bearer " + SERVICE_SECRET},
        )
    return response, calls, seen


_CASES = [
    ("minimax-h3", False, "/video/minimax-h3/v2/video_generation"),
    ("minimax-h3-direct", True, "/video/minimax-h3/direct/v2/video_generation"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("namespace,direct,expected_path", _CASES)
@pytest.mark.parametrize("key", [H3_ONLY_KEY, MODEL_ONLY_KEY, TEST_POOL_KEY])
async def test_continuation_accepts_h3_only_and_broad_keys_on_the_admitted_namespaced_route(
    stack, monkeypatch, key, namespace, direct, expected_path
):
    response, calls, seen = await _continue(stack, monkeypatch, credential=key, namespace=namespace, direct=direct)
    assert response.status_code == 200, response.text
    assert response.json() == {"accepted": True, "state": "submitted"}
    assert seen["path"] == expected_path  # authenticated and executed as the original public request
    assert seen["models"] == ["causyn-1.1"]  # the model restriction is still part of the authenticated key
    assert not any(call["path"].endswith("/not-sent") for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize(
    "namespace,expected_path",
    [
        ("minimax-h3", "/video/minimax-h3/v2/h3_context_ir"),
        ("minimax-h3-test", "/video/minimax-h3-test/v2/h3_context_ir"),
        (None, "/video/minimax-h3/v2/h3_context_ir"),  # older platform: unchanged default
    ],
)
async def test_continuation_context_ir_authenticates_on_the_namespaced_ir_route(
    stack, monkeypatch, namespace, expected_path
):
    """Context IR reserves metering budget in auth (needs durable SQL/Redis), so authenticate() is stubbed
    here and only the route it is asked to authorize is pinned."""
    from litellm.llms.causyn import context_ir as context_ir_module
    from litellm.proxy.video_endpoints import context_ir_endpoints
    from litellm.proxy.video_endpoints import moderation_metering_entry as entry

    authenticated: list[str] = []

    async def authenticate(request, credential):
        authenticated.append(request.url.path)
        return UserAPIKeyAuth(api_key="a" * 64, models=["causyn-1.1"])

    monkeypatch.setattr(execution, "authenticate", authenticate)
    monkeypatch.setattr(context_ir_endpoints, "create_context_ir", AsyncMock(return_value={"task_id": "ir-task"}))
    monkeypatch.setattr(context_ir_module, "get_context_ir_service", lambda: object())
    monkeypatch.setattr(entry, "attest", lambda request, metering, **kw: None)
    event = type("Event", (), {"model_dump": lambda self, mode=None: {}})()

    original = execution.execution_request

    def with_event(*args, **kwargs):
        request = original(*args, **kwargs)
        request.scope["moderation_context_ir_event"] = event
        return request

    monkeypatch.setattr(execution, "execution_request", with_event)
    response, calls, _ = await _continue(
        stack,
        monkeypatch,
        credential=H3_ONLY_KEY,
        namespace=namespace,
        route="context_ir",
        payload={"model": "causyn-1.1", "duration": 5, "ratio": "16:9", "content": [{"type": "text", "text": "kite"}]},
    )
    assert response.status_code == 200, response.text
    assert authenticated == [expected_path]
    assert [call["path"] for call in calls][-1].endswith("/receipt")


@pytest.mark.asyncio
@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("namespace,direct,expected_path", _CASES)
@pytest.mark.parametrize("key", [LEGACY_KEY, V1_ONLY_KEY])
async def test_continuation_rejects_keys_without_the_admitted_namespaced_route(
    stack, monkeypatch, key, namespace, direct, expected_path
):
    response, calls, seen = await _continue(stack, monkeypatch, credential=key, namespace=namespace, direct=direct)
    assert response.status_code == 403, response.text
    assert "path" not in seen  # never reached the provider
    not_sent = [call for call in calls if call["path"].endswith("/not-sent")]
    assert len(not_sent) == 1 and not_sent[0]["body"]["outcome"] == "rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize(
    "namespace,direct,key,allowed",
    [
        ("minimax-h3-test", False, TEST_POOL_KEY, True),
        ("minimax-h3-direct-test", True, TEST_POOL_KEY, True),
        # Opt-in is per key: the production-only grant must not reach the isolated test pool.
        ("minimax-h3-test", False, H3_ONLY_KEY, False),
        ("minimax-h3-direct-test", True, MODEL_ONLY_KEY, False),
    ],
)
async def test_continuation_keeps_the_test_pool_opt_in(stack, monkeypatch, namespace, direct, key, allowed):
    response, _, seen = await _continue(stack, monkeypatch, credential=key, namespace=namespace, direct=direct)
    if allowed:
        assert response.status_code == 200, response.text
        suffix = "/direct" if direct else ""
        assert seen["path"] == f"/video/minimax-h3-test{suffix}/v2/video_generation"
    else:
        assert response.status_code == 403, response.text
        assert "path" not in seen


@pytest.mark.asyncio
@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
async def test_continuation_for_the_generic_video_api_is_unchanged(stack, monkeypatch):
    # No namespace (a plain /v1/videos admission, or an older platform): still authenticated on /v1/videos.
    for key in (LEGACY_KEY, V1_ONLY_KEY, MODEL_ONLY_KEY):
        response, _, seen = await _continue(stack, monkeypatch, credential=key, namespace=None)
        assert response.status_code == 200, (key, response.text)
        assert seen["path"] == "/v1/videos"
    # An H3-only key has no business on the generic video API, even for its own model.
    response, _, seen = await _continue(stack, monkeypatch, credential=H3_ONLY_KEY, namespace=None)
    assert response.status_code == 403
    assert "path" not in seen
