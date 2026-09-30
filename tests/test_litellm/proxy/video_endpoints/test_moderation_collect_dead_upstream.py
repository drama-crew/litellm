import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import jwt
import pytest
from fastapi import FastAPI, HTTPException

import litellm
from litellm.llms.custom_llm import CustomLLMError, ProviderTaskNotFound
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


def submit_phase(native_id=NATIVE, finalized=True, age_s=0):
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
        facts=BillingFacts(
            started_at=NOW - timedelta(seconds=age_s),
            ended_at=NOW - timedelta(seconds=age_s),
            route="avideo_generation",
        )
        if native_id
        else None,
    )


class Meter:
    def __init__(self, submit, completion=None):
        self.submit = submit
        self.completion = completion
        self.persisted = []

    async def phase(self, intent_id, phase):
        return self.submit if phase == "submit" else self.completion

    async def binding(self, intent_id):
        return self.submit.binding

    async def persist(self, event):
        self.persisted.append(event)
        self.completion = event


async def run_collect(monkeypatch, status_error, submit, completion=None):
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    meter = Meter(submit, completion)
    monkeypatch.setattr(runtime, "store", lambda: meter)
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
    run_collect.meter = meter
    return response, posts


def not_found():
    # what the proxy raises for the causyn adapter's typed answer: NotFoundError chained from ProviderTaskNotFound
    error = litellm.NotFoundError(message="causyn video was not found", model="causyn-1.1", llm_provider="causyn")
    error.__cause__ = ProviderTaskNotFound("causyn video was not found")
    return error


@pytest.fixture(autouse=True)
def no_min_age(monkeypatch):
    monkeypatch.setenv("LITELLM_DEAD_UPSTREAM_MIN_AGE_S", "0")


@pytest.mark.asyncio
async def test_typed_not_found_terminalizes_and_closes_completion_at_zero(monkeypatch):
    response, posts = await run_collect(monkeypatch, not_found(), submit_phase())
    assert response.status_code == 200 and response.json() == {"accepted": True}
    (path, payload) = posts[-1]
    assert path == "/intents/intent/output"
    assert payload["native_id"] == NATIVE
    assert payload["facts"] == {"status": "failed", "url": None, "staging_key": None, "media_type": "video"}
    (event,) = run_collect.meter.persisted
    assert event.phase == "completion" and event.finalized and event.amount == Decimal(0)
    assert (event.provider, event.deployment_id, event.native_id) == ("causyn", "dep-1", NATIVE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        litellm.NotFoundError(message="model not found", model="m", llm_provider="causyn"),
        litellm.NotFoundError(message="causyn video was not found", model="m", llm_provider="other"),
        HTTPException(404, "gone"),
        CustomLLMError(404, "plain custom 404"),
    ],
)
async def test_untyped_404s_never_terminalize(monkeypatch, error):
    response, posts = await run_collect(monkeypatch, error, submit_phase())
    assert response.status_code != 200
    assert not any(path.endswith("/output") for path, _ in posts)
    assert run_collect.meter.persisted == []


@pytest.mark.asyncio
async def test_typed_not_found_on_young_task_does_not_terminalize(monkeypatch):
    monkeypatch.delenv("LITELLM_DEAD_UPSTREAM_MIN_AGE_S")
    response, posts = await run_collect(monkeypatch, not_found(), submit_phase(age_s=60))
    assert response.status_code != 200
    assert not any(path.endswith("/output") for path, _ in posts) and run_collect.meter.persisted == []
    response, posts = await run_collect(monkeypatch, not_found(), submit_phase(age_s=3600))
    assert response.status_code == 200 and posts[-1][1]["facts"]["status"] == "failed"


@pytest.mark.asyncio
async def test_typed_not_found_with_finalized_completion_is_not_terminal(monkeypatch):
    from litellm.proxy.video_endpoints.moderation_metering_projection import actual_debit

    submit = submit_phase()
    done = submit.model_copy(
        update={
            "phase": "completion",
            "request_id": "public-video:intent:completion",
            "amount": actual_debit(Decimal(5)),
        }
    )
    response, posts = await run_collect(monkeypatch, not_found(), submit, completion=done)
    assert response.status_code == 200 and response.json() == {"accepted": False}
    assert not any(path.endswith("/output") for path, _ in posts) and run_collect.meter.persisted == []


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


@pytest.mark.asyncio
async def test_delivered_close_applies_the_same_rule(monkeypatch):
    from litellm.proxy.video_endpoints.moderation_metering import BillingBinding

    submit = submit_phase()
    bound = submit.binding.model_copy(update={"actor_user_id": "actor"})
    submit = submit.model_copy(update={"binding": bound})
    meter = Meter(submit)

    async def read(*a, **k):
        raise not_found()

    monkeypatch.setattr(execution, "completion_status_read", read)
    await execution.close_completion(None, meter, bound, "delivered")
    (event,) = meter.persisted
    assert event.finalized and event.amount == Decimal(0)
    # untyped failure keeps the old behaviour: logged, nothing persisted
    meter2 = Meter(submit)

    async def read2(*a, **k):
        raise HTTPException(404, "x")

    monkeypatch.setattr(execution, "completion_status_read", read2)
    await execution.close_completion(None, meter2, bound, "delivered")
    assert meter2.persisted == []


def test_typed_error_is_still_a_plain_404_custom_llm_error():
    error = ProviderTaskNotFound("gone")
    assert isinstance(error, CustomLLMError) and error.status_code == 404 and error.message == "gone"
