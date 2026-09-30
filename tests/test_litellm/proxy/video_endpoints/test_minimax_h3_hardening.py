"""Direct-API hardening: auth before body, non-leaking 401, error envelope, 409/503 mapping, submit-time media policy."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

from litellm.llms.causyn import public_media_policy as policy
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.video_endpoints import context_ir_endpoints, endpoints
from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3
from litellm.proxy.video_endpoints import moderation_bridge as bridge
from litellm.proxy.video_endpoints.minimax_h3_paths import PREFIXES

LEAKY = (
    "Authentication Error, Invalid proxy server token passed. Received API Key = sk-...abcd, "
    "Key Hash (Token) =0123456789abcdef. Unable to find token in cache or `LiteLLM_VerificationTokenTable`"
)
FIXED_401 = "Invalid or missing API key"


def png(size=(300, 300)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(out, format="PNG")
    return out.getvalue()


def gif() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (300, 300)).save(out, format="GIF")
    return out.getvalue()


def data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def body(*images: tuple[str, str], **extra):
    content = [{"type": "text", "text": "A tea master pours glowing tea."}]
    content += [{"type": "image_url", "image_url": {"url": url}, "role": role} for url, role in images]
    return {"model": "minimax-h3", "content": content, "resolution": "768P", "duration": 5, "ratio": "16:9", **extra}


@pytest.fixture
def stack(monkeypatch):
    state = {"auth_calls": [], "platform": [], "reply": None}
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", "synthetic-test-secret")
    monkeypatch.setattr(bridge, "model_access", AsyncMock())
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "submitted", AsyncMock())
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "observe", AsyncMock())

    async def resolve(host, port):
        return ["93.184.216.34"]

    monkeypatch.setattr(policy, "resolve_host", resolve)

    async def auth(request: Request):
        authorization = request.headers.get("authorization")
        state["auth_calls"].append(dict(await _read_request_body(request)))
        if authorization == "Bearer leaky":
            raise ProxyException(message=LEAKY, type="auth_error", param=None, code=401)
        if authorization == "Bearer no-model":
            raise ProxyException(message="key not allowed to access model", type="auth_error", param=None, code=403)
        if authorization != "Bearer allowed":
            raise ProxyException(message=LEAKY, type="auth_error", param=None, code="401")
        return UserAPIKeyAuth(api_key="sk-owner", user_id="user", team_id="user", metadata={"openapi_key_id": "key"})

    async def control(request):
        data = json.loads(request.content)
        state["platform"].append((request.url.path, data))
        if state["reply"] is not None:
            return state["reply"](request, data)
        if request.url.path.endswith("/intents"):
            return httpx.Response(200, json=view("mod_video_1"))
        return httpx.Response(200, json=view(request.url.path.split("/")[-2]))

    def view(identity, **extra):
        return {
            "id": identity,
            "model": "causyn-1.1",
            "state": "input_moderation",
            "moderation_status": "pending",
            "policy_source": "model",
            "created_at": 1,
            "parameters": {"resolution": "768p", "duration": 5},
            **extra,
        }

    app = FastAPI()
    app.state.moderation_transport = httpx.MockTransport(control)
    app.dependency_overrides[h3.user_api_key_auth] = auth
    app.include_router(h3.router)
    app.include_router(context_ir_endpoints.router)
    app.include_router(endpoints.router)
    app.include_router(h3.fallback_router)
    state["view"] = view
    return TestClient(app), app, state


HEADERS = {"Authorization": "Bearer allowed"}


# ------------------------------------------------------------------ F2: auth before body


async def asgi(app, method: str, path: str, headers: dict[str, str], chunks: list[bytes]):
    reads = {"count": 0}
    queue = list(chunks)
    sent: list[dict] = []

    async def receive():
        reads["count"] += 1
        if queue:
            return {"type": "http.request", "body": queue.pop(0), "more_body": bool(queue)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("127.0.0.1", 1234),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "app": app,
    }
    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(payload), reads["count"]


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("path", ["/v2/video_generation", "/v2/h3_context_ir"])
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Bearer leaky"])
async def test_bad_key_is_rejected_without_reading_the_body(stack, prefix, path, authorization):
    _, app, state = stack
    if prefix.endswith("/direct") and path.endswith("h3_context_ir"):
        pytest.skip("Context IR create exists only on the IR namespace")
    headers = {"content-type": "application/json", **({"authorization": authorization} if authorization else {})}
    status, payload, receive_calls = await asgi(
        app, "POST", prefix + path, headers, [b'{"model": "minimax-h3", "content": ', b"[" * 1000]
    )
    assert status == 401
    assert receive_calls == 0, "the request body must never be read before authentication"
    assert payload["type"] == "error"
    assert payload["error"]["message"] == FIXED_401
    assert payload["error"]["type"] == "authorized_error"
    assert not state["platform"]


@pytest.mark.parametrize("prefix", PREFIXES)
async def test_bad_key_on_get_and_delete_routes_uses_the_same_fixed_message(stack, prefix):
    _, app, _ = stack
    for method, path in (
        ("GET", prefix + "/v2/query/video_generation/mod_video_1"),
        ("GET", prefix + "/v2/query/video_generation"),
        ("DELETE", prefix + "/v2/video_generation/mod_video_1"),
    ):
        status, payload, _ = await asgi(app, method, path, {"authorization": "Bearer leaky"}, [])
        assert status == 401, (method, path)
        assert payload["error"]["message"] == FIXED_401


def test_401_never_leaks_key_material_or_table_names(stack):
    client, _, _ = stack
    response = client.post("/video/minimax-h3/direct/v2/video_generation", json=body(), headers={"Authorization": "Bearer leaky"})
    assert response.status_code == 401
    text = response.text
    for forbidden in ("Key Hash", "LiteLLM_VerificationTokenTable", "0123456789abcdef", "sk-...", "Unable to find token"):
        assert forbidden not in text


@pytest.mark.parametrize("prefix", PREFIXES)
def test_model_authorization_still_sees_the_parsed_request_model(stack, prefix):
    client, _, state = stack
    created = client.post(prefix + "/v2/video_generation", json=body(), headers=HEADERS)
    assert created.status_code == 200, created.text
    # Identity check (placeholder body, stream untouched) then the normal dependency with the real body.
    assert [call["model"] for call in state["auth_calls"]] == ["causyn-1.1", "causyn-1.1"]
    forbidden = client.post(prefix + "/v2/video_generation", json=body(), headers={"Authorization": "Bearer no-model"})
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["type"] == "authorized_error"


async def test_valid_key_still_gets_validation_errors_after_authentication(stack):
    client, _, state = stack
    response = client.post("/video/minimax-h3/direct/v2/video_generation", json=body(duration=99), headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert len(state["auth_calls"]) == 1


async def test_pre_authentication_releases_any_budget_reservation(stack, monkeypatch):
    _, app, _ = stack
    released = []

    async def auth(request: Request):
        return UserAPIKeyAuth(api_key="sk-owner", user_id="user", team_id="user", budget_reservation={"entries": []})

    async def release(reservation):
        released.append(reservation)

    monkeypatch.setattr(h3, "release_budget_reservation", release)
    app.dependency_overrides[h3.user_api_key_auth] = auth
    await asgi(app, "GET", "/video/minimax-h3/v2/query/video_generation/mod_video_1", {"authorization": "Bearer allowed"}, [])
    assert released and released[0] == {"entries": []}


# ------------------------------------------------------------------ F3: envelope for unknown routes


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize(
    "method,path,expected",
    [
        ("GET", "/nope", 404),
        ("POST", "/v2/unknown", 404),
        ("GET", "/v2/query/video_generation/mod_video_1/extra", 404),
        ("PUT", "/v2/video_generation", 405),
        ("PATCH", "/v2/video_generation/mod_video_1", 405),
        ("GET", "/v2/video_generation", 405),
        ("POST", "/v2/query/video_generation", 405),
    ],
)
def test_unknown_paths_and_wrong_methods_use_the_error_envelope(stack, prefix, method, path, expected):
    client, _, _ = stack
    response = client.request(method, prefix + path, headers=HEADERS)
    assert response.status_code == expected, response.text
    payload = response.json()
    assert "detail" not in payload
    assert payload["type"] == "error"
    assert payload["error"]["http_code"] == str(expected)
    assert payload["error"]["type"] == ("not_found_error" if expected == 404 else "method_not_allowed_error")
    assert payload["request_id"]
    if expected == 405:
        assert {"POST", "GET", "DELETE"} & set(response.headers["allow"].replace(" ", "").split(","))


def test_fallback_does_not_shadow_real_routes_or_other_prefixes(stack):
    client, _, _ = stack
    assert client.post("/video/minimax-h3/v2/video_generation", json=body(), headers=HEADERS).status_code == 200
    assert client.get("/video/minimax-h3-other/x").json().get("detail") == "Not Found"


def test_error_types_cover_409_and_503():
    assert h3.ERROR_TYPES[409] == "conflict_error"
    assert h3.ERROR_TYPES[503] == "service_unavailable_error"
    assert h3.ERROR_TYPES[405] == "method_not_allowed_error"
    assert json.loads(h3.error_response(409, "x").body)["error"]["type"] == "conflict_error"
    assert json.loads(h3.error_response(503, "x").body)["error"]["type"] == "service_unavailable_error"


# ------------------------------------------------------------------ F4/F5: 409 and 503 mapping


@pytest.mark.parametrize("prefix", PREFIXES)
def test_idempotency_key_reuse_is_a_409_conflict(stack, prefix):
    client, _, state = stack
    state["reply"] = lambda request, data: httpx.Response(
        409, json={"detail": {"code": "idempotency_conflict", "message": "idempotency key does not match original request"}}
    )
    response = client.post(
        prefix + "/v2/video_generation", json=body(), headers={**HEADERS, "Idempotency-Key": "same-key"}
    )
    assert response.status_code == 409, response.text
    payload = response.json()
    assert payload["error"]["type"] == "conflict_error"
    assert payload["error"]["message"] == "Idempotency-Key was already used for a different request"


def test_platform_string_detail_mentioning_idempotency_is_also_a_409(stack):
    client, _, state = stack
    state["reply"] = lambda request, data: httpx.Response(409, json={"detail": "Idempotency-Key reused with a different request"})
    response = client.post(
        "/video/minimax-h3/v2/video_generation", json=body(), headers={**HEADERS, "Idempotency-Key": "k"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["message"] == "Idempotency-Key was already used for a different request"


def test_unrelated_intent_409_keeps_a_generic_client_error(stack):
    client, _, state = stack
    state["reply"] = lambda request, data: httpx.Response(409, json={"detail": "Source task is not approved"})
    response = client.post("/video/minimax-h3/v2/video_generation", json=body(), headers=HEADERS)
    assert response.status_code == 409
    assert "Idempotency" not in response.text
    assert response.json()["error"]["type"] == "conflict_error"


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize(
    "detail",
    [
        "A submitted generation cannot be cancelled; wait for its terminal state",
        "Submission outcome remains unknown; cancellation is durably requested",
    ],
)
def test_cancel_after_submission_is_a_clear_409(stack, prefix, detail):
    client, _, state = stack
    state["reply"] = lambda request, data: httpx.Response(409, json={"detail": detail})
    response = client.delete(prefix + "/v2/video_generation/mod_video_1", headers=HEADERS)
    assert response.status_code == 409, response.text
    message = response.json()["error"]["message"]
    assert response.json()["error"]["type"] == "conflict_error"
    if "submitted" in detail:
        assert message == "Task can no longer be cancelled; generation has already started"
    else:
        assert "cancel" in message.lower() and "Moderation control" not in message


@pytest.mark.parametrize("status", [500, 502, 503])
def test_genuine_platform_failures_stay_503(stack, status):
    client, _, state = stack
    state["reply"] = lambda request, data: httpx.Response(status, text="boom")
    response = client.post("/video/minimax-h3/v2/video_generation", json=body(), headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "service_unavailable_error"
    assert "boom" not in response.text


def test_other_platform_client_errors_keep_their_status(stack):
    client, _, state = stack
    state["reply"] = lambda request, data: httpx.Response(404, json={"detail": "Task not found"})
    response = client.get("/video/minimax-h3/v2/query/video_generation/mod_video_1", headers=HEADERS)
    assert response.status_code == 404


# ------------------------------------------------------------------ platform error_message passthrough


def test_view_error_message_surfaces_in_v2_task_and_video_object():
    view = bridge.View(
        id="mod_video_f",
        model="causyn-1.1",
        state="failed",
        moderation_status="pending",
        policy_source="model",
        created_at=1,
        error_message="reference image 1 could not be used: could not be retrieved",
    )
    task = bridge.v2_task(bridge.video(view), h3=True)
    assert task["status"] == "failed"
    assert task["error"]["message"] == "reference image 1 could not be used: could not be retrieved"
    assert bridge.video(view).error["message"] == "reference image 1 could not be used: could not be retrieved"


def test_old_platform_without_error_message_still_parses():
    view = bridge.View.model_validate(
        {"id": "mod_video_f", "state": "failed", "model": "causyn-1.1", "moderation_status": "pending", "policy_source": "model", "created_at": 1}
    )
    assert view.error_message is None
    assert bridge.video(view).error is None
    assert "error" not in bridge.v2_task(bridge.video(view), h3=True)


def test_error_message_reaches_the_query_response(stack):
    client, _, state = stack
    state["reply"] = lambda request, data: httpx.Response(
        200, json=state["view"]("mod_video_1", state="failed", error_message="reference image 2 could not be used: blocked")
    )
    response = client.get("/video/minimax-h3/v2/query/video_generation/mod_video_1", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["task"]["error"]["message"] == "reference image 2 could not be used: blocked"
