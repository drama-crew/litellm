"""Submit-time public media admission (official MiniMax limits + SSRF pre-check) on the H3 facade and /v1/videos."""

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


# ------------------------------------------------------------------ F6/F7: media policy at submit


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize(
    "image,message",
    [
        (data_url(gif(), "image/gif"), "reference image 1: unsupported format (GIF); allowed: JPEG, PNG, WEBP, HEIC, HEIF"),
        (data_url(b"%PDF-1.4" + b"0" * 200), "reference image 1: unsupported format (PDF)"),
        (data_url(png((100, 100))), "reference image 1: dimensions 100x100"),
        (data_url(png((768, 256))), "reference image 1: aspect ratio"),
        ("http://127.0.0.1/a.png", "reference image 1: media URL must point to a public internet host"),
        ("http://drama-litellm:4000/a.png", "media URL must point to a public internet host"),
        ("http://100.100.100.200/latest/meta-data", "media URL must point to a public internet host"),
    ],
)
def test_facade_rejects_bad_media_before_anything_is_submitted(stack, prefix, image, message):
    client, _, state = stack
    response = client.post(
        prefix + "/v2/video_generation", json=body((image, "reference_image")), headers=HEADERS
    )
    assert response.status_code == 400, response.text
    assert message in response.json()["error"]["message"]
    assert response.json()["error"]["type"] == "bad_request_error"
    assert not state["platform"], "nothing may reach the moderation platform"


def test_facade_accepts_valid_media_and_public_urls(stack, monkeypatch):
    client, _, state = stack
    uploaded = AsyncMock(return_value="private://reference")
    monkeypatch.setattr(bridge, "upload_media", uploaded)
    response = client.post(
        "/video/minimax-h3/direct/v2/video_generation",
        json=body((data_url(png()), "reference_image"), ("https://cdn.example.com/b.png", "reference_image")),
        headers=HEADERS,
    )
    assert response.status_code == 200, response.text
    assert state["platform"][-1][0].endswith("/intents")
    assert uploaded.await_count == 1  # only the Base64 reference; the public URL is left for the app to fetch


def test_second_reference_is_numbered_one_based(stack):
    client, _, _ = stack
    response = client.post(
        "/video/minimax-h3/v2/video_generation",
        json=body((data_url(png()), "reference_image"), (data_url(gif(), "image/png"), "reference_image")),
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("reference image 2: unsupported format (GIF)")


@pytest.mark.parametrize(
    "references",
    [
        [{"role": "reference", "media_type": "image", "url": data_url(gif(), "image/png")}],
        [{"role": "reference", "media_type": "image", "url": "http://169.254.169.254/x.png"}],
        [{"role": "first_frame", "media_type": "image", "url": "http://localhost/x.png"}],
    ],
)
def test_v1_videos_causyn_public_path_applies_the_same_policy(stack, references):
    client, _, state = stack
    response = client.post(
        "/v1/videos",
        json={"model": "causyn-1.1", "prompt": "p", "seconds": "5", "aspect_ratio": "16:9", "references": references},
        headers=HEADERS,
    )
    assert response.status_code == 400, response.text
    assert not state["platform"]


def test_v1_videos_frame_fields_are_checked(stack):
    client, _, state = stack
    response = client.post(
        "/v1/videos",
        json={"model": "causyn-1.1", "prompt": "p", "image": "http://10.0.0.8/first.png"},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert "first frame image" in response.text
    assert not state["platform"]


def test_non_causyn_models_are_not_subject_to_the_public_policy(stack):
    client, _, state = stack
    response = client.post(
        "/v1/videos",
        json={"model": "hailuo-h3", "prompt": "p", "references": [{"role": "reference", "media_type": "image", "url": "http://10.0.0.8/x.png"}]},
        headers=HEADERS,
    )
    assert response.status_code == 200, response.text


async def test_internal_producer_admission_bypasses_the_public_policy(stack, monkeypatch):
    _, app, state = stack
    called = []

    async def spy(payload):
        called.append(payload)

    monkeypatch.setattr(policy, "validate_public_payload", spy)
    client = TestClient(app)
    state["reply"] = lambda request, data: httpx.Response(
        200, json={"acquired": False, "native_id": "mod_video_existing"}
    )
    response = client.post(
        "/v1/videos",
        json={"model": "causyn-1.1", "prompt": "p", "references": [{"role": "reference", "media_type": "image", "url": "http://10.0.0.8/x.png"}]},
        headers={**HEADERS, "x-drama-moderation-admission": "ticket"},
    )
    assert response.status_code == 200, response.text
    assert called == [], "ticketed internal traffic must never be checked against the public media policy"


def test_policy_violation_is_a_400_with_no_internal_detail(stack, monkeypatch):
    client, _, _ = stack

    async def boom(payload):
        raise policy.MediaPolicyError("reference image 1: unsupported format (GIF); allowed: JPEG, PNG, WEBP, HEIC, HEIF")

    monkeypatch.setattr(policy, "validate_public_payload", boom)
    response = client.post("/video/minimax-h3/v2/video_generation", json=body(), headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("reference image 1:")
