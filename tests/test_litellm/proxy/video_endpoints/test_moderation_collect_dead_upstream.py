import time
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import jwt
import pytest
from fastapi import FastAPI, HTTPException

import litellm
from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent
from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts
from litellm.types.videos.utils import encode_video_id_with_provider

SECRET = "synthetic-collect-secret-long-enough-32-bytes"
NOW = datetime.now(timezone.utc)
NATIVE = encode_video_id_with_provider("task-1", "causyn", "dep-1")


def ticket():
    return jwt.encode(
        {
            "intent_id": "intent",
            "request_digest": "a" * 64,
            "input_digest": "b" * 64,
            "model": "causyn-1.1",
            "policy_version": "v1",
            "policy_digest": "c" * 64,
            "purpose": "collect",
            "aud": "moderation-fork",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        SECRET,
        algorithm="HS256",
    )


def submit_phase(native_id=NATIVE, finalized=True):
    bound = BillingBinding(
        intent_id="intent",
        request_digest="a" * 64,
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="causyn-1.1",
        expected_phases=("submit", "completion"),
    )
    return PhaseEvent(
        binding=bound,
        request_id="public-video:intent:submit",
        phase="submit",
        provider="causyn" if native_id else "",
        deployment_id="dep-1" if native_id else "",
        native_id=native_id,
        provider_task_id="task-1" if native_id else "",
        amount=Decimal(0) if native_id else None,
        finalized=finalized and bool(native_id),
        facts=BillingFacts(started_at=NOW, ended_at=NOW, route="avideo_generation") if native_id else None,
    )


class Meter:
    def __init__(self, submit):
        self.submit = submit

    async def phase(self, intent_id, phase):
        return self.submit if phase == "submit" else None


async def run_collect(monkeypatch, status_error, submit):
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setattr(runtime, "store", lambda: Meter(submit))
    posts = []

    async def platform(request, method, path, payload):
        posts.append((path, payload))
        if path.endswith("/execution"):
            return {
                "state": "submitted",
                "native_id": NATIVE,
                "route": "avideo_generation",
                "principal": {"fingerprint": "key", "user_id": "user", "team_id": "team"},
                "metering": {},
                "billing": {},
                "upload_ticket": "t",
            }
        return {}

    async def read(*args, **kwargs):
        raise status_error

    monkeypatch.setattr(execution.bridge, "platform", platform)
    monkeypatch.setattr(execution, "completion_status_read", read)
    app = FastAPI()
    app.include_router(execution.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fork"
    ) as client:
        response = await client.post(
            "/internal/moderation/collect", json={"ticket": ticket()}, headers={"Authorization": "Bearer " + SECRET}
        )
    return response, posts


def not_found():
    return litellm.NotFoundError(message="causyn video was not found", model="causyn-1.1", llm_provider="causyn")


@pytest.mark.asyncio
async def test_provider_not_found_terminalizes_as_failed(monkeypatch):
    response, posts = await run_collect(monkeypatch, not_found(), submit_phase())
    assert response.status_code == 200 and response.json() == {"accepted": True}
    (path, payload) = posts[-1]
    assert path == "/intents/intent/output"
    assert payload["native_id"] == NATIVE
    assert payload["facts"] == {"status": "failed", "url": None, "staging_key": None, "media_type": "video"}


@pytest.mark.asyncio
async def test_http_404_exception_also_terminalizes(monkeypatch):
    response, posts = await run_collect(monkeypatch, HTTPException(404, "gone"), submit_phase())
    assert response.status_code == 200
    assert posts[-1][1]["facts"]["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        litellm.InternalServerError(message="boom", model="m", llm_provider="causyn"),
        litellm.Timeout(message="slow", model="m", llm_provider="causyn"),
        httpx.ConnectError("refused"),
        HTTPException(503, "down"),
    ],
)
async def test_transient_errors_keep_non_terminal_behaviour(monkeypatch, error):
    response, posts = await run_collect(monkeypatch, error, submit_phase())
    assert response.status_code >= 500
    assert not any(path.endswith("/output") for path, _ in posts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "submit",
    [submit_phase(native_id=""), submit_phase(native_id="other"), submit_phase(finalized=False), None],
)
async def test_not_found_without_settled_submit_for_this_native_id_is_not_terminal(monkeypatch, submit):
    response, posts = await run_collect(monkeypatch, not_found(), submit)
    assert response.status_code != 200
    assert not any(path.endswith("/output") for path, _ in posts)
