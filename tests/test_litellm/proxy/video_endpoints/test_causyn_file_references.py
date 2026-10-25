"""``causyn_file://`` media references: validated publicly, resolved by the platform, logged verbatim."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import context_ir_endpoints, endpoints
from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3
from litellm.proxy.video_endpoints import moderation_bridge as bridge
from litellm.proxy.video_endpoints import openapi_log_capture
from litellm.proxy.video_endpoints.minimax_h3_models import MediaURL
from litellm.proxy.video_endpoints.minimax_h3_paths import ALL_PREFIXES, DIRECT_PREFIX, IR_PREFIX, TEST_PREFIXES

REAL_START = openapi_log_capture.start
HEADERS = {"Authorization": "Bearer allowed"}
ID1, ID2 = "cf_" + "a" * 26, "cf_" + "b0" * 13
REF1, REF2 = "causyn_file://" + ID1, "causyn_file://" + ID2
OSS1, OSS2 = f"oss://bucket/moderation/uploads/acct/user/{ID1}", f"oss://bucket/moderation/uploads/acct/user/{ID2}"


def body(*refs: tuple[str, str], **extra):
    content = [{"type": "text", "text": "A tea master pours glowing tea."}]
    content += [{"type": "image_url", "image_url": {"url": url}, "role": role} for url, role in refs]
    return {"model": "minimax-h3", "content": content, "resolution": "768P", "duration": 5, "ratio": "16:9", **extra}


def video_body(ref: str) -> dict:
    return {
        "model": "minimax-h3",
        "content": [
            {"type": "text", "text": "Dance."},
            {"type": "video_url", "video_url": {"url": ref}, "role": "reference_video"},
        ],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
    }


@pytest.fixture
def stack(monkeypatch):
    state = {"calls": [], "resolve": None}
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", "synthetic-test-secret")
    monkeypatch.setattr(bridge, "model_access", AsyncMock())
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "submitted", AsyncMock())
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "observe", AsyncMock())
    state["start"] = AsyncMock(return_value=None)
    monkeypatch.setattr(h3.endpoints.openapi_log_capture, "start", state["start"])

    async def auth(request: Request):
        return UserAPIKeyAuth(api_key="sk-owner", user_id="user", team_id="user", metadata={"openapi_key_id": "key"})

    def control(request):
        data = json.loads(request.content)
        path = request.url.path.removeprefix("/internal/moderation")
        state["calls"].append((path, data))
        if path == "/files/resolve":
            if state["resolve"] is not None:
                return state["resolve"]
            return httpx.Response(
                200,
                json={"refs": {i: f"oss://bucket/moderation/uploads/acct/user/{i}" for i in data["file_ids"]}},
            )
        return httpx.Response(
            200,
            json={
                "id": "mod_video_1",
                "model": "causyn-1.1",
                "state": "input_moderation",
                "moderation_status": "pending",
                "policy_source": "model",
                "created_at": 1,
                "parameters": {"resolution": "768p", "duration": 5},
            },
        )

    app = FastAPI()
    app.state.moderation_transport = httpx.MockTransport(control)
    app.dependency_overrides[h3.user_api_key_auth] = auth
    app.include_router(h3.router)
    app.include_router(context_ir_endpoints.router)
    app.include_router(endpoints.router)
    state["client"] = TestClient(app)
    return state


def intents(state) -> dict:
    (call,) = [data for path, data in state["calls"] if path == "/intents"]
    return call


# ------------------------------------------------------------------ public validation


def test_validate_accepts_causyn_file():
    assert MediaURL(url=REF1).url == REF1


@pytest.mark.parametrize(
    "value",
    [
        "causyn_file://../x",
        "causyn_file://" + ID1.upper(),
        "causyn_file://cf_" + "a" * 25,
        "causyn_file://cf_" + "a" * 27,
        "causyn_file://" + "a" * 26,
        "causyn_file://" + ID1 + "/extra",
        "causyn_file://" + ID1 + "\n",
        "causyn_file://",
    ],
)
def test_validate_rejects_malformed_causyn_file(value):
    with pytest.raises(ValidationError):
        MediaURL(url=value)


def test_mm_file_message_points_to_causyn_file(stack):
    with pytest.raises(ValidationError, match=r"use causyn_file://"):
        MediaURL(url="mm_file://abc")
    response = stack["client"].post(
        IR_PREFIX + "/v2/video_generation", json=body(("mm_file://abc", "reference_image")), headers=HEADERS
    )
    assert response.status_code == 400 and "use causyn_file://" in response.json()["error"]["message"]
    assert stack["calls"] == []


@pytest.mark.parametrize("raw", [f"oss://bucket/moderation/uploads/acct/user/{ID1}", "OSS://bucket/key"])
def test_oss_scheme_still_rejected(stack, raw):
    with pytest.raises(ValidationError):
        MediaURL(url=raw)
    response = stack["client"].post(
        IR_PREFIX + "/v2/video_generation", json=body((raw, "reference_image")), headers=HEADERS
    )
    assert response.status_code == 400
    assert stack["calls"] == []


# ------------------------------------------------------------------ resolution


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_direct_and_ir_both_resolve(stack, prefix):
    response = stack["client"].post(
        prefix + "/v2/video_generation", json=body((REF1, "reference_image")), headers=HEADERS
    )
    assert response.status_code == 200, response.text
    sent = intents(stack)["payload"]
    assert OSS1 in json.dumps(sent) and "causyn_file" not in json.dumps(sent)


def test_inline_resolves_all_references_in_one_platform_call(stack):
    content = body((REF1, "reference_image"), (REF2, "reference_image"), (REF1, "reference_image"))
    response = stack["client"].post(IR_PREFIX + "/v2/video_generation", json=content, headers=HEADERS)
    assert response.status_code == 200, response.text
    resolves = [data for path, data in stack["calls"] if path == "/files/resolve"]
    assert len(resolves) == 1
    assert resolves[0]["file_ids"] == [ID1, ID2]  # de-duplicated, request order
    assert resolves[0]["principal"]["user_id"] == "user"
    urls = [item["url"] for item in intents(stack)["payload"]["references"]]
    assert urls == [OSS1, OSS2, OSS1]
    assert [path for path, _ in stack["calls"]] == ["/files/resolve", "/intents"]  # resolve happens before intake


def test_video_reference_is_resolved_too(stack):
    response = stack["client"].post(DIRECT_PREFIX + "/v2/video_generation", json=video_body(REF2), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert OSS2 in json.dumps(intents(stack)["payload"])


def test_context_ir_resolves_references(stack, monkeypatch):
    monkeypatch.setattr(context_ir_endpoints, "get_context_ir_service", lambda: object())
    data = {k: v for k, v in body((REF1, "reference_image")).items() if k not in ("resolution",)}
    response = stack["client"].post(IR_PREFIX + "/v2/h3_context_ir", json=data, headers=HEADERS)
    assert response.status_code == 200, response.text
    resolves = [d for path, d in stack["calls"] if path == "/files/resolve"]
    assert len(resolves) == 1 and resolves[0]["file_ids"] == [ID1]
    assert OSS1 in json.dumps(intents(stack)["payload"])


def test_no_references_means_no_resolve_call(stack):
    response = stack["client"].post(IR_PREFIX + "/v2/video_generation", json=body(), headers=HEADERS)
    assert response.status_code == 200
    assert [path for path, _ in stack["calls"]] == ["/intents"]


@pytest.mark.parametrize(
    "reply,status,expect",
    [
        ((404, {"detail": {"code": "file_not_found"}}), 404, "file_not_found"),
        ((400, {"detail": {"code": "file_expired", "reason": "file expired"}}), 400, "file expired"),
        ((400, {"detail": {"code": "file_not_ready", "reason": "file is not ready"}}), 400, "file is not ready"),
        ((400, {"detail": {"code": "file_rejected", "reason": "rejected: too long"}}), 400, "rejected: too long"),
        (
            (400, {"detail": {"code": "invalid_upload", "reason": "causyn_file references are not enabled"}}),
            400,
            "causyn_file references are not enabled",
        ),
    ],
)
def test_resolve_404_maps_to_404_and_400_maps_to_400_with_reason(stack, reply, status, expect):
    stack["resolve"] = httpx.Response(reply[0], json=reply[1])
    response = stack["client"].post(
        IR_PREFIX + "/v2/video_generation", json=body((REF1, "reference_image")), headers=HEADERS
    )
    assert response.status_code == status, response.text
    assert expect in response.text
    assert [path for path, _ in stack["calls"]] == ["/files/resolve"]  # nothing is admitted


def test_resolve_platform_failure_is_503(stack):
    stack["resolve"] = httpx.Response(500, json={"detail": "boom"})
    response = stack["client"].post(
        IR_PREFIX + "/v2/video_generation", json=body((REF1, "reference_image")), headers=HEADERS
    )
    assert response.status_code == 503 and "boom" not in response.text


def test_resolve_omitting_a_requested_id_is_not_admitted(stack):
    stack["resolve"] = httpx.Response(200, json={"refs": {ID1: OSS1}})
    response = stack["client"].post(
        IR_PREFIX + "/v2/video_generation",
        json=body((REF1, "reference_image"), (REF2, "reference_image")),
        headers=HEADERS,
    )
    assert response.status_code == 404
    assert [path for path, _ in stack["calls"]] == ["/files/resolve"]


def test_resolver_refuses_an_unexpected_scheme_from_the_platform(stack):
    stack["resolve"] = httpx.Response(200, json={"refs": {ID1: "https://evil.example/x"}})
    response = stack["client"].post(
        IR_PREFIX + "/v2/video_generation", json=body((REF1, "reference_image")), headers=HEADERS
    )
    assert response.status_code == 503
    assert [path for path, _ in stack["calls"]] == ["/files/resolve"]


# ------------------------------------------------------------------ log capture


@pytest.fixture
def real_capture(stack, monkeypatch):
    """Run the real ``openapi_log_capture.start`` against a stubbed table, capturing what would be stored."""
    created = AsyncMock()
    monkeypatch.setattr(openapi_log_capture, "database", lambda: object())
    monkeypatch.setattr(openapi_log_capture.logs, "create_log", created)
    monkeypatch.setattr(openapi_log_capture, "start", REAL_START)
    return created


@pytest.mark.parametrize("route", ["/v2/video_generation", "/v2/h3_context_ir"])
def test_log_capture_keeps_causyn_file_not_oss(stack, real_capture, monkeypatch, route):
    monkeypatch.setattr(context_ir_endpoints, "get_context_ir_service", lambda: object())
    data = body((REF1, "reference_image"))
    if route.endswith("h3_context_ir"):
        data.pop("resolution")
    response = stack["client"].post(IR_PREFIX + route, json=data, headers=HEADERS)
    assert response.status_code == 200, response.text
    (call,) = real_capture.await_args_list
    stored = json.dumps(call.kwargs["payload"])
    assert REF1 in stored and "oss://" not in stored
    assert OSS1 in json.dumps(intents(stack)["payload"])  # the platform still receives the internal reference
