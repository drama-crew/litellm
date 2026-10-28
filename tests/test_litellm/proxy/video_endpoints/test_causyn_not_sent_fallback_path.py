"""Production path for causyn admission refusals: real Router (with a fallbacks list that does not
include the model group) + real CausynVideoHandler registered as a custom provider + real metered entry
+ the real ``/internal/moderation/submit`` handler. Only Redis admission and the platform are faked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request

import litellm
from litellm import Router
from litellm.llms.causyn import CausynVideoHandler
from litellm.llms.causyn import handler as causyn_module
from litellm.llms.libtv.video_generate import VideoGenerateError, VideoGenerateSettings
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.proxy.video_endpoints import moderation_metering_entry as entry
from tests.test_litellm.proxy.video_endpoints.test_moderation_execution import _post_submit


def _router(monkeypatch, code):
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", "true")
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "s" * 32)

    async def refuse(*_args, **_kwargs):
        raise VideoGenerateError(code, "refused before any write")

    monkeypatch.setattr(causyn_module, "enqueue_video_generate", refuse)
    provider = CausynVideoHandler(
        redis_factory=lambda: MagicMock(),
        settings_factory=lambda: VideoGenerateSettings(
            source_hosts=frozenset({"source.example"}), target_hosts=frozenset({"target.example"})
        ),
        task_id_factory=lambda: "0123456789abcdef0123456789abcdef",
        clock=lambda: 2_000_000_000.0,
    )
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "causyn", "custom_handler": provider}])
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    from litellm.utils import custom_llm_setup

    custom_llm_setup()
    return Router(
        model_list=[
            {
                "model_name": "causyn-1.1",
                "litellm_params": {"model": "causyn/causyn-1.0"},
                "model_info": {"id": "causyn-price-v1", "output_cost_per_second_768x512": 0.1},
            }
        ],
        num_retries=2,
        retry_after=0,
        fallbacks=[{"seedance-3": ["seedance-2.5"]}],  # production: fallbacks exist, but not for causyn-1.1
        enable_pre_call_checks=True,
    )


def _call(router):
    return router.avideo_generation(
        model="causyn-1.1",
        prompt="animate",
        seconds="5",
        size="768x512",
        aspect_ratio="3:2",
        generate_audio=True,
        seed=7,
        reference_images=["https://source.example/reference.png"],
    )


async def _scoped(monkeypatch):
    from litellm.proxy.spend_tracking import budget_reservation

    scope = MagicMock()
    scope.store.void_unsent = AsyncMock()
    scope.binding.intent_id = "intent"
    scope.binding.request_digest = "d" * 64
    scope.phase = "submit"
    monkeypatch.setattr(entry, "scope_for", AsyncMock(return_value=scope))
    monkeypatch.setattr(budget_reservation, "release_budget_reservation", AsyncMock())
    return scope


def _request():
    return Request({"type": "http", "method": "POST", "path": "/v1/videos", "headers": [], "app": FastAPI()})


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["no_capacity_available", "no_worker_available"])
async def test_router_path_admission_refusal_is_503_not_sent(monkeypatch, code):
    router = _router(monkeypatch, code)
    await _scoped(monkeypatch)
    with pytest.raises(Exception) as caught:
        await entry.execute(_request(), UserAPIKeyAuth(api_key="k", user_id="u", team_id="t"), "avideo_generation", _call(router))

    error = caught.value
    assert isinstance(error, HTTPException) and error.status_code == 503
    assert error.headers == {"x-drama-submission": "not_sent"}
    assert execution.failure_outcome(error) == "not_sent"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["no_capacity_available", "no_worker_available"])
async def test_submit_handler_marks_not_sent_and_never_parks_on_router_path(monkeypatch, code):
    router = _router(monkeypatch, code)
    scope = await _scoped(monkeypatch)

    async def invoke(request, auth, payload, route):
        return await entry.execute(request, auth, "avideo_generation", _call(router))

    meter, calls = await _post_submit(monkeypatch, invoke)
    assert meter.marks == []
    assert [b for p, b in meter.bodies if p.endswith("/not-sent")] == [{"token": "attempt", "outcome": "not_sent"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["enqueue_ambiguous", "redis_error"])
async def test_submit_handler_stays_ambiguous_for_post_write_errors_on_router_path(monkeypatch, code):
    router = _router(monkeypatch, code)
    await _scoped(monkeypatch)

    async def invoke(request, auth, payload, route):
        return await entry.execute(request, auth, "avideo_generation", _call(router))

    meter, calls = await _post_submit(monkeypatch, invoke)
    assert [m[1] for m in meter.marks] == ["provider_submission_ambiguous"]
    assert not any(p.endswith("/not-sent") for p, _ in meter.bodies)


async def _through_endpoint_exception_wrapper(router, request, auth):
    """endpoints.video_generation converts whatever the processor raised with _handle_llm_api_exception."""
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    processor = ProxyBaseLLMRequestProcessing(data={})
    try:
        return await entry.execute(request, auth, "avideo_generation", _call(router))
    except Exception as exc:
        raise await processor._handle_llm_api_exception(
            e=exc, user_api_key_dict=auth, proxy_logging_obj=_logging(), version="t"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["no_capacity_available", "no_worker_available"])
async def test_submit_handler_marks_not_sent_after_endpoint_exception_conversion(monkeypatch, code):
    """Production: the public endpoint rewraps the metered entry's HTTPException into a ProxyException, which
    must keep the not-sent verdict, or the submit handler parks the task as provider_submission_ambiguous."""
    router = _router(monkeypatch, code)
    await _scoped(monkeypatch)

    async def invoke(request, auth, payload, route):
        return await _through_endpoint_exception_wrapper(router, request, auth)

    meter, calls = await _post_submit(monkeypatch, invoke)
    assert meter.marks == []
    assert [b for p, b in meter.bodies if p.endswith("/not-sent")] == [{"token": "attempt", "outcome": "not_sent"}]


@pytest.mark.asyncio
async def test_submit_handler_stays_ambiguous_after_endpoint_exception_conversion(monkeypatch):
    router = _router(monkeypatch, "enqueue_ambiguous")
    await _scoped(monkeypatch)

    async def invoke(request, auth, payload, route):
        return await _through_endpoint_exception_wrapper(router, request, auth)

    meter, calls = await _post_submit(monkeypatch, invoke)
    assert [m[1] for m in meter.marks] == ["provider_submission_ambiguous"]
    assert not any(p.endswith("/not-sent") for p, _ in meter.bodies)


def _logging():
    logging_obj = AsyncMock()
    logging_obj.post_call_response_headers_hook.return_value = {}
    logging_obj.post_call_failure_hook.return_value = None
    return logging_obj
