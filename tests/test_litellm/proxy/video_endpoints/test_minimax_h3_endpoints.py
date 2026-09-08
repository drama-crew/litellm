from __future__ import annotations

import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3
from litellm.proxy.video_endpoints.minimax_h3_models import MiniMaxH3Create, MiniMaxTask, decode_task, encode_task
from litellm.types.videos.main import VideoObject


@pytest.fixture
def api(monkeypatch):
    calls = []
    normalized = []
    owner = {"key": "sk-owner"}
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "synthetic-test-secret")

    async def auth(request: Request):
        if request.headers.get("authorization") != "Bearer allowed":
            raise ProxyException(message="Invalid API key", type="authentication_error", param=None, code=401)
        normalized.append(await _read_request_body(request))
        return UserAPIKeyAuth(api_key=owner["key"])

    async def create(request, response, input_reference, user_api_key_dict):
        body = await _read_request_body(request)
        calls.append(("create", body))
        return VideoObject(id="video-native", object="video", status="queued")

    async def query(video_id, request, response, user_api_key_dict):
        calls.append(("query", video_id))
        video = VideoObject(id=video_id, object="video", status="completed")
        video._hidden_params = {"url": "https://media.example/result.mp4"}
        return video

    monkeypatch.setattr(h3.endpoints, "video_generation", create)
    monkeypatch.setattr(h3.endpoints, "video_status", query)
    app = FastAPI()
    app.dependency_overrides[h3.user_api_key_auth] = auth
    app.include_router(h3.router)
    return TestClient(app), calls, normalized, owner


def body(**extra):
    return {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A tea master pours glowing tea."}],
        "resolution": "768P",
        "duration": 8,
        "ratio": "16:9",
        **extra,
    }


def test_create_query_and_owner_binding(api):
    client, calls, normalized, owner = api
    headers = {"Authorization": "Bearer allowed"}
    result = client.post("/v2/video_generation", json=body(), headers=headers)
    assert result.status_code == 200, result.text
    task_id = result.json()["task_id"]
    assert "video-native" not in task_id
    assert normalized[0]["model"] == "hailuo-h3"
    assert calls[0][1]["seconds"] == "8"
    query = client.get("/v2/query/video_generation/" + task_id, headers=headers)
    assert query.status_code == 200, query.text
    task = query.json()["task"]
    assert (task["id"], task["model"], task["status"]) == (task_id, "MiniMax-H3", "succeeded")
    assert task["content"]["url"] == "https://media.example/result.mp4"
    assert task["usage"]["output_seconds"] == 8
    assert calls[-1] == ("query", "video-native")
    owner["key"] = "sk-another-owner"
    count = len(calls)
    assert client.get("/v2/query/video_generation/" + task_id, headers=headers).status_code == 404
    assert len(calls) == count


def test_invalid_auth_and_inputs_never_submit(api):
    client, calls, _, _ = api
    response = client.post("/v2/video_generation", json=body())
    assert response.status_code == 401
    assert response.json()["error"]["http_code"] == "401"
    assert response.json()["error"]["type"] == "authorized_error"
    for changes in [
        {"duration": True},
        {"duration": 3},
        {"ratio": "adaptive"},
        {"model": "gpt-4"},
        {"api_base": "https://evil.example"},
    ]:
        response = client.post(
            "/v2/video_generation", json=body(**changes), headers={"Authorization": "Bearer allowed"}
        )
        assert response.status_code == 400, response.text
        assert response.json()["type"] == "error"
    assert not calls


def test_keyframe_role_order_and_h3_max_modes():
    first = {"type": "image_url", "image_url": {"url": "https://media.example/first.png"}, "role": "first_frame"}
    last = {"type": "image_url", "image_url": {"url": "https://media.example/last.png"}, "role": "last_frame"}
    content = [body()["content"][0], last, first]
    spec = MiniMaxH3Create.model_validate(body(model="MiniMax-H3-Max", content=content))
    actual = spec.internal_body()
    assert actual["model"] == "hailuo-h3-max"
    assert actual["image"] == first["image_url"]["url"]
    assert actual["last_image"] == last["image_url"]["url"]
    assert actual["aspect_ratio"] == "adaptive"
    assert actual["parameters"] == {"modeType": "frames2video"}
    with pytest.raises(ValueError):
        MiniMaxH3Create.model_validate(body(model="MiniMax-H3-Max", resolution="2K"))
    with pytest.raises(ValueError):
        MiniMaxH3Create.model_validate(body(content=content + [first]))


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


def test_real_key_allowlist_is_applied_before_create_and_query(api, monkeypatch):
    import litellm.proxy.proxy_server as proxy_server
    import litellm
    from litellm import Router
    from litellm.llms.libtv import LibTVLLM
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.types.videos.utils import encode_video_id_with_provider
    from tests.test_litellm.proxy.video_endpoints.test_causyn_public_stack import (
        _AuthStore,
        _ProxyConfig,
        _ProxyLogging,
    )

    client, calls, _, _ = api
    client.app.dependency_overrides.clear()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "libtv", "custom_handler": LibTVLLM()}])
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    litellm.utils.custom_llm_setup()
    router = Router(
        model_list=[
            {
                "model_name": "hailuo-h3",
                "litellm_params": {"model": "hailuo-h3", "custom_llm_provider": "libtv"},
                "model_info": {"id": "h3-active"},
            }
        ],
        num_retries=0,
    )
    for name, value in {
        "llm_router": router,
        "llm_model_list": router.model_list,
        "proxy_logging_obj": _ProxyLogging(),
        "proxy_config": _ProxyConfig(),
        "general_settings": {"disable_budget_reservation": True},
        "master_key": "sk-master",
        "prisma_client": _AuthStore({"sk-h3": ["hailuo-h3"], "sk-other": ["other-model"]}),
        "user_api_key_cache": UserApiKeyCache(),
        "user_custom_auth": None,
    }.items():
        monkeypatch.setattr(proxy_server, name, value)

    native_id = encode_video_id_with_provider(video_id="native", provider="libtv", model_id="h3-active")

    async def create(request, response, input_reference, user_api_key_dict):
        calls.append(("create", await _read_request_body(request)))
        return VideoObject(id=native_id, object="video", status="queued")

    monkeypatch.setattr(h3.endpoints, "video_generation", create)
    denied = client.post("/v2/video_generation", json=body(), headers={"Authorization": "Bearer sk-other"})
    assert denied.status_code in (401, 403), denied.text
    assert not calls
    created = client.post("/v2/video_generation", json=body(), headers={"Authorization": "Bearer sk-h3"})
    assert created.status_code == 200, created.text
    task_id = created.json()["task_id"]
    url = "/v2/query/video_generation/" + task_id
    denied = client.get(url, headers={"Authorization": "Bearer sk-other"})
    assert denied.status_code in (401, 403), denied.text
    assert len(calls) == 1
    assert client.get(url, headers={"Authorization": "Bearer sk-h3"}).status_code == 200
    assert calls[-1] == ("query", native_id)


def test_adaptive_keyframe_query_does_not_echo_ignored_ratio(api):
    client, _, _, _ = api
    first = {"type": "image_url", "image_url": {"url": "https://media.example/portrait.png"}}
    headers = {"Authorization": "Bearer allowed"}
    created = client.post("/v2/video_generation", json=body(content=body()["content"] + [first]), headers=headers)
    assert created.status_code == 200, created.text
    task_id = created.json()["task_id"]
    assert decode_task(task_id).ratio == "adaptive"
    result = client.get("/v2/query/video_generation/" + task_id, headers=headers)
    assert result.status_code == 200
    assert "ratio" not in result.json()["task"]


def test_unsupported_backend_features_fail_before_paid_submit(api):
    client, calls, _, _ = api
    last = {"type": "image_url", "image_url": {"url": "https://media.example/last.png"}, "role": "last_frame"}
    for content in (body(callback_url="https://callback.example/status"), body(content=body()["content"] + [last])):
        result = client.post("/v2/video_generation", json=content, headers={"Authorization": "Bearer allowed"})
        assert result.status_code == 422, result.text
        assert result.json()["error"]["type"] == "unprocessable_entity_error"
    assert not calls


def test_routing_overrides_cannot_bypass_official_auth(api):
    client, calls, _, _ = api
    for suffix, extra in (("?model=other-model", {}), ("", {"x-litellm-model": "other-model"})):
        result = client.post(
            "/v2/video_generation" + suffix, json=body(), headers={"Authorization": "Bearer allowed", **extra}
        )
        assert result.status_code == 400
    assert not calls
