from __future__ import annotations

import json
import time
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
from litellm.proxy.video_endpoints.minimax_h3_models import (
    DURATION_MESSAGE,
    MiniMaxH3Content,
    MiniMaxH3Create,
    MiniMaxTask,
    decode_task,
    encode_task,
)
from litellm.proxy.video_endpoints.minimax_h3_paths import ALL_PREFIXES, PREFIXES, TEST_PREFIXES


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
    assert (task["id"], task["model"], task["status"], task["resolution"]) == (task_id, "MiniMax-H3", "running", "768P")
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
        {"model": "MiniMax-H3-Max"},
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
    assert result["model"] == "MiniMax-H3"
    assert result["usage"]["input_image_count"] == 2
    assert result["updated_at"] == 1
    assert result["content"]["url"].endswith("result.mp4")


def content_body(**extra):
    return {**body(model="MiniMax-H3", duration=8), **extra}


def test_keyframe_role_order_and_h3_max_modes():
    first = {"type": "image_url", "image_url": {"url": "https://media.example/first.png"}, "role": "first_frame"}
    last = {"type": "image_url", "image_url": {"url": "https://media.example/last.png"}, "role": "last_frame"}
    content = [content_body()["content"][0], last, first]
    spec = MiniMaxH3Content.model_validate(content_body(model="MiniMax-H3-Max", content=content))
    actual = spec.internal_body()
    assert actual["model"] == "hailuo-h3-max"
    assert actual["image"] == first["image_url"]["url"]
    assert actual["last_image"] == last["image_url"]["url"]
    assert actual["aspect_ratio"] == "adaptive"
    assert actual["parameters"] == {"modeType": "frames2video"}
    with pytest.raises(ValueError):
        MiniMaxH3Content.model_validate(content_body(model="MiniMax-H3-Max", resolution="2K"))
    with pytest.raises(ValueError):
        MiniMaxH3Content.model_validate(content_body(model="MiniMax-H3-Max", duration=4))
    with pytest.raises(ValueError):
        MiniMaxH3Content.model_validate(content_body(content=content + [first]))
    assert MiniMaxH3Content.model_validate(content_body(duration=4)).duration == 4


def test_reference_media_is_shaped_for_the_libtv_backend():
    content = [
        content_body()["content"][0],
        {"type": "audio_url", "audio_url": {"url": "https://media.example/a.mp3"}},
        {"type": "image_url", "image_url": {"url": "https://media.example/ref.png"}, "role": "reference_image"},
        {"type": "video_url", "video_url": {"url": "https://media.example/ref.mp4"}},
    ]
    actual = MiniMaxH3Content.model_validate(content_body(content=content)).internal_body()
    assert "seed" not in actual
    assert actual["reference_images"] == ["https://media.example/ref.png"]
    assert actual["reference_videos"] == ["https://media.example/ref.mp4"]
    assert actual["reference_audios"] == ["https://media.example/a.mp3"]
    assert actual["parameters"] == {"modeType": "mixed2video"}
    assert "references" not in actual


@pytest.mark.parametrize("seed", [42, -1, 4294967296, 1.5])
def test_seed_is_rejected_as_an_unknown_field(seed):
    with pytest.raises(ValueError, match="extra_forbidden"):
        MiniMaxH3Content.model_validate(content_body(seed=seed))
    with pytest.raises(ValueError, match="extra_forbidden"):
        MiniMaxH3Create.model_validate(body(seed=seed))


def test_per_type_reference_caps_are_enforced_but_total_cap_and_audio_pairing_are_not():
    image_ref = {"type": "image_url", "image_url": {"url": "https://media.example/ref.png"}, "role": "reference_image"}
    video_ref = {"type": "video_url", "video_url": {"url": "https://media.example/ref.mp4"}}
    audio_ref = {"type": "audio_url", "audio_url": {"url": "https://media.example/a.mp3"}}
    text = content_body()["content"]
    with pytest.raises(ValueError, match="9 images, 3 videos or 3 audio"):
        MiniMaxH3Content.model_validate(content_body(content=text + [image_ref] * 10))
    with pytest.raises(ValueError, match="9 images, 3 videos or 3 audio"):
        MiniMaxH3Content.model_validate(content_body(content=text + [video_ref] * 4))
    MiniMaxH3Content.model_validate(content_body(content=text + [image_ref] * 9 + [video_ref] * 3 + [audio_ref]))
    MiniMaxH3Content.model_validate(content_body(content=text + [audio_ref]))


def test_task_authentication_and_expiry(monkeypatch):
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "test-secret")
    task = MiniMaxTask(
        native_id="native",
        model="MiniMax-H3",
        created_at=int(time.time()),
        duration=8,
        resolution="768P",
        ratio="16:9",
        image_count=0,
        owner="owner",
    )
    encoded = encode_task(task)
    assert decode_task(encoded) == task
    with pytest.raises(ValueError):
        decode_task(encoded[:-8] + "abcdefgh")
    with pytest.raises(ValueError):
        decode_task(encode_task(task.model_copy(update={"created_at": int(time.time()) - 8 * 86400})))


# --- test worker pool facade ---------------------------------------------------------------------------------

TEST_NAMESPACES = {
    "/video/minimax-h3": "minimax-h3",
    "/video/minimax-h3/direct": "minimax-h3-direct",
    "/video/minimax-h3-test": "minimax-h3-test",
    "/video/minimax-h3-test/direct": "minimax-h3-direct-test",
}


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_every_prefix_has_its_own_namespace_and_only_test_prefixes_carry_the_pool(api, prefix):
    client, calls, _, _ = api
    created = client.post(
        prefix + "/v2/video_generation", json=body(), headers={"Authorization": "Bearer allowed"}
    )
    assert created.status_code == 200, created.text
    admitted = calls[0][1]
    assert admitted["namespace"] == TEST_NAMESPACES[prefix]
    assert admitted["payload"].get("prompt_processing") == ("direct" if prefix.endswith("/direct") else None)
    if prefix in TEST_PREFIXES:
        assert admitted["payload"]["worker_pool"] == "test"
    else:
        assert "worker_pool" not in admitted["payload"]


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_test_and_production_tasks_are_invisible_to_each_other(api, prefix):
    client, calls, _, owner = api
    headers = {"Authorization": "Bearer allowed"}
    task_id = client.post(prefix + "/v2/video_generation", json=body(), headers=headers).json()["task_id"]
    path = prefix + "/v2/query/video_generation/" + task_id
    assert client.get(path, headers=headers).status_code == 200
    assert client.get(prefix + "/v2/query/video_generation", headers=headers).json()["total"] == 1
    assert client.delete(prefix + "/v2/video_generation/" + task_id, headers=headers).json()["action"] == "cancelled"
    for other in (value for value in ALL_PREFIXES if value != prefix):
        assert client.get(other + "/v2/query/video_generation/" + task_id, headers=headers).status_code == 404
        assert client.delete(other + "/v2/video_generation/" + task_id, headers=headers).status_code == 404
        assert client.get(other + "/v2/query/video_generation", headers=headers).json()["total"] == 0
    namespaces = {data["namespace"] for _, data in calls if "namespace" in data}
    assert namespaces == {TEST_NAMESPACES[prefix]} | {TEST_NAMESPACES[o] for o in ALL_PREFIXES if o != prefix}


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
@pytest.mark.parametrize("pool", ["test", "prod", "", None, True])
def test_a_public_body_can_never_choose_the_worker_pool(api, prefix, pool):
    client, calls, _, _ = api
    response = client.post(
        prefix + "/v2/video_generation", json=body(worker_pool=pool), headers={"Authorization": "Bearer allowed"}
    )
    assert response.status_code in (400, 422), response.text
    assert response.json()["type"] == "error"
    assert not calls


def test_test_prefix_mirrors_every_v2_route_of_the_production_facade():
    def suffixes(prefix):
        found = set()
        for router in (h3.router, context_ir_endpoints.router):
            for route in router.routes:
                if route.path.startswith(prefix + "/v2/"):
                    found |= {(method, route.path[len(prefix) :]) for method in route.methods}
        return found

    assert suffixes("/video/minimax-h3-test") == suffixes("/video/minimax-h3")
    assert suffixes("/video/minimax-h3-test/direct") == suffixes("/video/minimax-h3/direct")
    assert ("POST", "/v2/h3_context_ir") in suffixes("/video/minimax-h3-test")
    assert len(suffixes("/video/minimax-h3-test")) >= 4


# --- MiniMax official V2 parity (model alias, extra, duration, error types) ----------------------------------

HEADERS = {"Authorization": "Bearer allowed"}


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
@pytest.mark.parametrize("model", ["MiniMax-H3", "minimax-h3"])
def test_official_and_legacy_model_names_submit_the_same_internal_request(api, prefix, model):
    client, calls, normalized, _ = api
    created = client.post(prefix + "/v2/video_generation", json=body(model=model), headers=HEADERS)
    assert created.status_code == 200, created.text
    assert normalized[0]["model"] == "causyn-1.1"
    payload = calls[0][1]["payload"]
    assert payload["model"] == "causyn-1.1" and "extra" not in payload
    assert calls[0][1]["namespace"] == TEST_NAMESPACES[prefix]


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_h3_max_is_rejected_with_a_clear_message_before_admission(api, prefix):
    client, calls, _, _ = api
    response = client.post(prefix + "/v2/video_generation", json=body(model="MiniMax-H3-Max"), headers=HEADERS)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "bad_request_error"
    assert "MiniMax-H3-Max" in error["message"] and "not available" in error["message"]
    assert not calls


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_query_and_list_report_the_official_model_name_and_filter_accepts_both(api, prefix):
    client, calls, _, _ = api
    task_id = client.post(prefix + "/v2/video_generation", json=body(model="minimax-h3"), headers=HEADERS).json()[
        "task_id"
    ]
    assert client.get(prefix + "/v2/query/video_generation/" + task_id, headers=HEADERS).json()["task"]["model"] == (
        "MiniMax-H3"
    )
    for spelling in ("MiniMax-H3", "minimax-h3"):
        listed = client.get(prefix + "/v2/query/video_generation?filter.model=" + spelling, headers=HEADERS)
        assert listed.status_code == 200, listed.text
        assert [item["model"] for item in listed.json()["items"]] == ["MiniMax-H3"]
        assert calls[-1][1]["model"] == "causyn-1.1"
    rejected = client.get(prefix + "/v2/query/video_generation?filter.model=MiniMax-H3-Max", headers=HEADERS)
    assert rejected.status_code == 400


@pytest.mark.parametrize("prefix", ["/video/minimax-h3", "/video/minimax-h3-test"])
@pytest.mark.parametrize("model", ["MiniMax-H3", "minimax-h3"])
def test_standalone_context_ir_accepts_both_model_spellings(api, monkeypatch, prefix, model):
    monkeypatch.setattr(context_ir_endpoints, "get_context_ir_service", lambda: object())
    client, calls, normalized, _ = api
    data = {k: v for k, v in body(model=model).items() if k != "resolution"}
    response = client.post(prefix + "/v2/h3_context_ir", json=data, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert calls and calls[0][1]["payload"]["model"] == "MiniMax-H3"
    assert normalized[0]["model"] == "causyn-1.1"


@pytest.mark.parametrize("model", ["MiniMax-H3-Max", "causyn-1.1", "nope"])
def test_standalone_context_ir_rejects_other_models(api, model):
    client, calls, _, _ = api
    data = {k: v for k, v in body(model=model).items() if k != "resolution"}
    response = client.post("/video/minimax-h3/v2/h3_context_ir", json=data, headers=HEADERS)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "bad_request_error"
    assert not calls


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
@pytest.mark.parametrize("duration", [4])
def test_duration_4_is_rejected_up_front_on_every_flavor_with_one_message(api, prefix, duration):
    client, calls, _, _ = api
    response = client.post(prefix + "/v2/video_generation", json=body(duration=duration), headers=HEADERS)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "bad_request_error"
    assert "5" in error["message"] and "15" in error["message"] and "4" in error["message"]
    assert not calls


@pytest.mark.parametrize("prefix", ["/video/minimax-h3", "/video/minimax-h3-test"])
def test_standalone_context_ir_duration_4_matches_the_generation_message(api, prefix):
    client, calls, _, _ = api
    data = {k: v for k, v in body(duration=4).items() if k != "resolution"}
    generation = client.post(prefix + "/v2/video_generation", json=body(duration=4), headers=HEADERS)
    response = client.post(prefix + "/v2/h3_context_ir", json=data, headers=HEADERS)
    assert response.status_code == 400, response.text
    assert DURATION_MESSAGE in response.json()["error"]["message"]
    assert DURATION_MESSAGE in generation.json()["error"]["message"]
    assert not calls


@pytest.mark.parametrize("prefix", PREFIXES + TEST_PREFIXES)
@pytest.mark.parametrize("duration", [5, 15])
def test_duration_bounds_5_and_15_are_accepted(api, prefix, duration):
    client, _, _, _ = api
    assert (
        client.post(prefix + "/v2/video_generation", json=body(duration=duration), headers=HEADERS).status_code == 200
    )


def _flavor(prefix):
    return "direct" if prefix.endswith("/direct") else "ir"


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
@pytest.mark.parametrize("mode", [None, "disabled", "balanced", "quality"])
def test_extra_prompt_expansion_mode_semantics_per_flavor(api, prefix, mode):
    client, calls, _, _ = api
    extra = {} if mode is None else {"extra": {"prompt_expansion_mode": mode}}
    response = client.post(prefix + "/v2/video_generation", json=body(**extra), headers=HEADERS)
    direct = _flavor(prefix) == "direct"
    acceptable = {None, "disabled"} if direct else {None, "balanced", "quality"}
    if mode in acceptable:
        assert response.status_code == 200, response.text
        assert "extra" not in calls[0][1]["payload"]
        return
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["type"] == "unprocessable_entity_error" and error["code"] == "unsupported_parameter"
    pointer = prefix.removesuffix("/direct") if direct else prefix + "/direct"
    assert pointer in error["message"]
    assert not calls


def test_extra_allows_an_empty_object_and_rejects_everything_else(api):
    client, calls, _, _ = api
    assert client.post("/video/minimax-h3/v2/video_generation", json=body(extra={}), headers=HEADERS).status_code == 200
    calls.clear()
    for bad in (
        {"prompt_expansion_mode": "balance"},
        {"prompt_expansion_mode": ""},
        {"prompt_expansion_mode": True},
        {"prompt_expansion_mode": None},
        {"other": 1},
        {"prompt_expansion_mode": "quality", "other": 1},
        "balanced",
        ["quality"],
    ):
        response = client.post("/video/minimax-h3/v2/video_generation", json=body(extra=bad), headers=HEADERS)
        assert response.status_code == 400, (bad, response.text)
        assert response.json()["error"]["type"] == "bad_request_error"
    assert not calls


def test_extra_is_not_a_context_ir_field(api):
    client, calls, _, _ = api
    data = {k: v for k, v in body(extra={"prompt_expansion_mode": "quality"}).items() if k != "resolution"}
    assert client.post("/video/minimax-h3/v2/h3_context_ir", json=data, headers=HEADERS).status_code == 400
    assert not calls


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("resolution", ["2K", "480P"])
def test_unavailable_resolutions_keep_an_explicit_400(api, prefix, resolution):
    client, calls, _, _ = api
    response = client.post(prefix + "/v2/video_generation", json=body(resolution=resolution), headers=HEADERS)
    assert response.status_code == 400
    assert resolution in response.json()["error"]["message"] and "768P" in response.json()["error"]["message"]
    assert not calls


def test_mm_file_urls_get_an_explicit_message(api):
    client, calls, _, _ = api
    content = body()["content"] + [
        {"type": "image_url", "image_url": {"url": "mm_file://abc"}, "role": "reference_image"}
    ]
    response = client.post("/video/minimax-h3/v2/video_generation", json=body(content=content), headers=HEADERS)
    assert response.status_code == 400
    assert "mm_file" in response.json()["error"]["message"]
    assert not calls
