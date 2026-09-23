from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.video_endpoints import context_ir_endpoints
from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3
from litellm.proxy.video_endpoints import moderation_bridge as bridge
from litellm.proxy.video_endpoints.minimax_h3_models import MiniMaxH3Create
from litellm.proxy.video_endpoints.minimax_h3_paths import PREFIXES


@pytest.fixture
def api(monkeypatch):
    calls = []
    normalized = []
    owner = {"key": "sk-owner"}
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", "synthetic-test-secret")
    monkeypatch.setattr(bridge, "model_access", AsyncMock())
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "start", AsyncMock(return_value=None))
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "submitted", AsyncMock())
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "observe", AsyncMock())
    tasks = {}

    async def auth(request: Request):
        if request.headers.get("authorization") != "Bearer allowed":
            raise ProxyException(message="Invalid API key", type="authentication_error", param=None, code=401)
        normalized.append(await _read_request_body(request))
        return UserAPIKeyAuth(api_key=owner["key"], user_id="user", team_id="user", metadata={"openapi_key_id": "key"})

    async def control(request):
        data = json.loads(request.content)
        calls.append((request.url.path, data))
        if request.url.path.endswith("/intents"):
            identity = "mod_video_" + str(len(tasks))
            tasks[identity] = data
            return httpx.Response(200, json=view(identity, data))
        rows = {
            key: value
            for key, value in tasks.items()
            if value["principal"] == data["principal"] and value["namespace"] == data["namespace"]
        }
        if request.url.path.endswith("/list"):
            return httpx.Response(
                200, json={"items": [view(key, value) for key, value in rows.items()], "total": len(rows)}
            )
        identity = request.url.path.split("/")[-2]
        if identity not in rows:
            return httpx.Response(404)
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"action": "cancelled"})
        return httpx.Response(200, json=view(identity, rows[identity]))

    def view(identity, data):
        return {
            "id": identity,
            "model": "causyn-1.1",
            "state": "output_moderation",
            "moderation_status": "pending",
            "policy_source": "model",
            "created_at": 1,
            "parameters": {"resolution": "768p", "duration": 5},
            "output": {"url": "https://private.invalid/hidden.mp4"},
        }

    app = FastAPI()
    app.state.moderation_transport = httpx.MockTransport(control)
    app.dependency_overrides[h3.user_api_key_auth] = auth
    app.include_router(h3.router)
    app.include_router(context_ir_endpoints.router)
    return TestClient(app), calls, normalized, owner


def body(**extra):
    return {
        "model": "minimax-h3",
        "content": [{"type": "text", "text": "A tea master pours glowing tea."}],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
        **extra,
    }


@pytest.mark.parametrize("prefix", PREFIXES)
def test_durable_intake_query_list_cancel_and_namespace_owner_binding(api, prefix):
    client, calls, normalized, owner = api
    headers = {"Authorization": "Bearer allowed"}
    created = client.post(prefix + "/v2/video_generation", json=body(), headers=headers)
    assert created.status_code == 200, created.text
    task_id = created.json()["task_id"]
    assert normalized[0]["model"] == "causyn-1.1"
    admitted = calls[0][1]
    assert admitted["namespace"] == ("minimax-h3-direct" if prefix.endswith("/direct") else "minimax-h3")
    assert admitted["payload"].get("prompt_processing") == ("direct" if prefix.endswith("/direct") else None)
    assert admitted["payload"]["prompt"] == body()["content"][0]["text"]
    path = prefix + "/v2/query/video_generation/" + task_id
    queried = client.get(path, headers=headers)
    assert queried.status_code == 200, queried.text
    task = queried.json()["task"]
    assert (task["id"], task["model"], task["status"], task["resolution"]) == (task_id, "minimax-h3", "running", "768P")
    assert task["moderation_status"] == "pending"
    assert "content" not in task and "private.invalid" not in queried.text
    listed = client.get(prefix + "/v2/query/video_generation?filter.model=minimax-h3", headers=headers)
    assert listed.json()["items"] == [task]
    assert calls[-1][1]["model"] == "causyn-1.1"
    assert client.delete(prefix + "/v2/video_generation/" + task_id, headers=headers).json()["action"] == "cancelled"
    other = next(value for value in PREFIXES if value != prefix)
    assert client.get(other + "/v2/query/video_generation/" + task_id, headers=headers).status_code == 404
    assert client.delete(other + "/v2/video_generation/" + task_id, headers=headers).status_code == 404
    assert client.get(other + "/v2/query/video_generation", headers=headers).json()["total"] == 0
    owner["key"] = "sk-another-owner"
    assert client.get(path, headers=headers).status_code == 404


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize(
    "changes",
    [
        {"duration": True},
        {"duration": 3},
        {"duration": 16},
        {"ratio": "adaptive"},
        {"model": "MiniMax-H3"},
        {"model": "causyn-1.1"},
        {"resolution": "2K"},
        {"resolution": "480P"},
        {"api_base": "https://evil.example"},
        {"prompt_processing": "direct"},
    ],
)
def test_invalid_inputs_never_submit(api, prefix, changes):
    client, calls, _, _ = api
    response = client.post(
        prefix + "/v2/video_generation", json=body(**changes), headers={"Authorization": "Bearer allowed"}
    )
    assert response.status_code == 400, response.text
    assert response.json()["type"] == "error"
    assert not calls


@pytest.mark.parametrize("prefix", PREFIXES)
def test_auth_and_backend_rejection_precedes_admission(api, prefix):
    client, calls, _, _ = api
    assert client.post(prefix + "/v2/video_generation", json=body()).status_code == 401
    last = {"type": "image_url", "image_url": {"url": "https://media.example/last.png"}, "role": "last_frame"}
    for content in (body(callback_url="https://callback.example/status"), body(content=body()["content"] + [last])):
        response = client.post(
            prefix + "/v2/video_generation", json=content, headers={"Authorization": "Bearer allowed"}
        )
        assert response.status_code == 422, response.text
    assert not calls


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize(
    "header", ["x-litellm-model", "custom-llm-provider", "x-drama-moderation-admission", "x-drama-moderation-read"]
)
def test_client_cannot_override_routing_or_admission(api, prefix, header):
    client, calls, _, _ = api
    response = client.post(
        prefix + "/v2/video_generation", json=body(), headers={"Authorization": "Bearer allowed", header: "forged"}
    )
    assert response.status_code == 400
    assert not calls


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/v2/video_generation"),
        ("POST", "/v2/video_generation/direct"),
        ("GET", "/v2/query/video_generation"),
        ("GET", "/v2/query/video_generation/mod_video_old"),
        ("DELETE", "/v2/video_generation/mod_video_old"),
        ("POST", "/v2/h3_context_ir"),
    ],
)
def test_old_routes_are_removed(api, method, path):
    client, calls, _, _ = api
    assert client.request(method, path, json=body(), headers={"Authorization": "Bearer allowed"}).status_code == 404
    assert not calls


@pytest.mark.parametrize("roles", [("last_frame", "first_frame"), ("reference_image",) * 9])
def test_image_roles_and_full_reference_capacity(roles):
    images = [
        {"type": "image_url", "image_url": {"url": f"https://media.example/{index}.png"}, "role": role}
        for index, role in enumerate(roles)
    ]
    spec = MiniMaxH3Create.model_validate(body(content=body()["content"] + images))
    actual = spec.internal_body()
    assert actual["aspect_ratio"] == ("adaptive" if "first_frame" in roles else "16:9")
    assert [ref["role"] for ref in actual["references"]] == [
        "reference" if role == "reference_image" else role for role in roles
    ]
    assert actual["generate_audio"] is True


@pytest.mark.parametrize("prefix", PREFIXES)
def test_missing_moderation_service_never_falls_back_to_provider(api, monkeypatch, prefix):
    client, calls, _, _ = api
    monkeypatch.setattr(bridge, "configured", lambda auth: False)
    assert (
        client.post(
            prefix + "/v2/video_generation", json=body(), headers={"Authorization": "Bearer allowed"}
        ).status_code
        == 503
    )
    assert not calls


@pytest.mark.parametrize("prefix", PREFIXES)
def test_new_route_auth_budget_classification(prefix, monkeypatch):
    from litellm.proxy.auth.auth_utils import _is_video_mutation_route, _is_video_retrieval_route
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.spend_tracking import budget_reservation

    path = prefix + "/v2/video_generation"
    assert _is_video_mutation_route(path)
    assert _is_video_retrieval_route(prefix + "/v2/query/video_generation/task")
    assert not _is_video_mutation_route(path, "DELETE")
    assert RouteChecks.is_llm_api_route(path)
    auth = UserAPIKeyAuth(allowed_routes=[path], metadata={"openapi_key_id": "test"})
    assert RouteChecks.is_virtual_key_allowed_to_call_route(path, auth)
    request = Request({"type": "http", "method": "POST", "path": path, "headers": []})
    assert bridge.defer_budget(request, auth, path)
    monkeypatch.setattr(budget_reservation, "get_model_from_request", lambda *args, **kwargs: "causyn-1.1")
    monkeypatch.setattr(budget_reservation, "_estimate_request_max_cost_for_model", lambda **kwargs: 25.0)
    assert budget_reservation.estimate_request_max_cost({"model": "causyn-1.1"}, path, None) == 25.0


def test_approved_projection_has_public_model_and_usage():
    view = bridge.View(
        id="mod_video_done",
        model="causyn-1.1",
        state="completed",
        moderation_status="approved",
        policy_source="model",
        created_at=1,
        parameters={"resolution": "768p", "duration": 5},
        output={"url": "https://media.example/result.mp4"},
        usage={"total_seconds": 5, "output_seconds": 5, "input_image_count": 2},
    )
    result = bridge.v2_task(bridge.video(view), h3=True)
    assert result["model"] == "minimax-h3"
    assert result["usage"]["input_image_count"] == 2
    assert result["content"]["url"].endswith("result.mp4")
