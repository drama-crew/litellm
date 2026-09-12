import json
import time
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest
from fastapi import FastAPI

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.types.videos.main import VideoObject

SECRET = "synthetic-service-signature-secret"


def ticket(purpose="submit", **changes):
    return jwt.encode(
        {
            "intent_id": "intent",
            "request_digest": "a" * 64,
            "input_digest": "b" * 64,
            "model": "hailuo-h3",
            "policy_version": "v1",
            "policy_digest": "c" * 64,
            "purpose": purpose,
            "aud": "moderation-fork",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
            **changes,
        },
        SECRET,
        algorithm="HS256",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["receipt_ack", "provider_response"])
async def test_continuation_never_resubmits_after_unknown_or_receipt_ack_loss(monkeypatch, failure):
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    app = FastAPI()
    app.include_router(execution.router)
    state = {"state": "input_moderation", "receipts": 0}

    async def platform(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/begin"):
            if state["state"] != "input_moderation":
                return httpx.Response(
                    200, json={"acquired": False, "state": state["state"], "native_id": state.get("native_id")}
                )
            state["state"] = "submission_unknown"
            return httpx.Response(
                200,
                json={
                    "acquired": True,
                    "token": "attempt",
                    "credential": "sk-original",
                    "route": "avideo_generation",
                    "request": {"model": "hailuo-h3", "prompt": "synthetic"},
                },
            )
        assert request.url.path.endswith("/receipt")
        state.update(state="submitted", native_id=payload["native_id"], receipts=state["receipts"] + 1)
        assert payload["billing"]["request_id"] == "public-video:intent"
        raise httpx.ReadTimeout("synthetic lost receipt ACK", request=request)

    app.state.moderation_transport = httpx.MockTransport(platform)
    monkeypatch.setattr(execution, "authenticate", AsyncMock(return_value=UserAPIKeyAuth(api_key="original")))
    provider = AsyncMock(return_value=VideoObject(id="accepted-native", object="video", status="queued"))
    if failure == "provider_response":
        provider.side_effect = httpx.ReadTimeout("synthetic provider accepted then response lost")
    monkeypatch.setattr(execution, "invoke", provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fork"
    ) as client:
        first = await client.post(
            "/internal/moderation/submit", json={"ticket": ticket()}, headers={"Authorization": "Bearer " + SECRET}
        )
        assert first.status_code == 500
        second = await client.post(
            "/internal/moderation/submit", json={"ticket": ticket()}, headers={"Authorization": "Bearer " + SECRET}
        )
    assert second.status_code == 200
    assert provider.await_count == 1
    assert state["state"] == ("submitted" if failure == "receipt_ack" else "submission_unknown")
    assert state["receipts"] == (1 if failure == "receipt_ack" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [ticket(purpose="collect"), ticket(exp=1), "unsigned-client-bypass"])
async def test_invalid_ticket_never_reaches_begin(monkeypatch, value):
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    app = FastAPI()
    app.include_router(execution.router)
    control = AsyncMock()
    monkeypatch.setattr(execution.bridge, "platform", control)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fork") as client:
        response = await client.post(
            "/internal/moderation/submit", json={"ticket": value}, headers={"Authorization": "Bearer " + SECRET}
        )
    assert response.status_code == 403
    assert control.await_count == 0


@pytest.mark.asyncio
async def test_local_validation_failure_closes_not_sent_fence(monkeypatch):
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    app = FastAPI()
    app.include_router(execution.router)
    control = AsyncMock(
        side_effect=[
            {
                "acquired": True,
                "token": "attempt",
                "credential": "sk-original",
                "route": "context_ir",
                "request": {"model": "invalid"},
            },
            {"accepted": True},
        ]
    )
    monkeypatch.setattr(execution.bridge, "platform", control)
    provider = AsyncMock()
    monkeypatch.setattr(execution, "invoke", provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fork"
    ) as client:
        await client.post(
            "/internal/moderation/submit", json={"ticket": ticket()}, headers={"Authorization": "Bearer " + SECRET}
        )
    assert control.call_args.args[2].endswith("/not-sent")
    assert control.call_args.args[3]["outcome"] == "not_sent"
    assert provider.await_count == 0


@pytest.mark.asyncio
async def test_character_download_timeout_is_not_provider_submission(monkeypatch):
    from starlette.requests import Request

    from litellm.proxy.video_endpoints import endpoints

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, *args):
            raise httpx.ReadTimeout("synthetic private input download timeout")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())
    provider = AsyncMock()
    monkeypatch.setattr(endpoints, "video_create_character", provider)
    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/videos/characters", "headers": [], "query_string": b""}
    )
    with pytest.raises(Exception) as error:
        await execution.invoke(
            request,
            UserAPIKeyAuth(api_key="synthetic"),
            {"video": "https://private.test/object", "name": "synthetic"},
            "avideo_create_character",
        )
    assert provider.await_count == 0
    assert execution.failure_outcome(error.value) == "not_sent"
