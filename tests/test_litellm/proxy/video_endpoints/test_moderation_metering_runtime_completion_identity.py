from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent
from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider

NOW = datetime.now(timezone.utc)


def _scope(authority):
    binding = BillingBinding(
        intent_id="i",
        request_digest="d",
        fingerprint="k",
        user_id="u",
        team_id="t",
        model="seedance-2.0",
        expected_phases=("submit", "completion"),
    )
    nid = encode_video_id_with_provider("tid1", "wavespeed", "dep-1")
    facts = BillingFacts(started_at=NOW, ended_at=NOW, route="avideo_generation")
    submit = PhaseEvent(
        binding=binding,
        request_id="public-video:i:submit",
        phase="submit",
        provider="wavespeed",
        deployment_id="dep-1",
        native_id=nid,
        provider_task_id="tid1",
        amount=Decimal(300),
        finalized=True,
        facts=facts,
    )
    placeholder = PhaseEvent(
        binding=binding,
        request_id="public-video:i:completion",
        phase="completion",
        provider="",
        deployment_id="",
        native_id="",
        provider_task_id="",
    )
    return runtime.Scope(binding, "completion", authority, placeholder, submit), nid


def _logging_obj(scope):
    logging_obj = MagicMock()
    setattr(logging_obj, runtime.SCOPE_KEY, scope)
    setattr(logging_obj, runtime.CALL_TYPE_KEY, "avideo_status")
    logging_obj.model_call_details = {}
    logging_obj.custom_llm_provider = "wavespeed"
    return logging_obj


def _result(task_id):
    return VideoObject(id=encode_video_id_with_provider(task_id, "wavespeed", None), object="video", status="completed")


@pytest.mark.asyncio
async def test_completion_persists_under_submit_identity_when_status_id_is_model_less():
    authority = AsyncMock()
    scope, nid = _scope(authority)
    await runtime._handoff(_logging_obj(scope), _result("tid1"), NOW, NOW)
    assert authority.persist.await_count == 1
    event = authority.persist.await_args.args[0]
    assert (event.native_id, event.deployment_id, event.provider, event.provider_task_id) == (
        nid,
        "dep-1",
        "wavespeed",
        "tid1",
    )
    assert event.phase == "completion"
    assert event.amount == Decimal(0)
    assert event.finalized is True


@pytest.mark.asyncio
async def test_completion_for_different_task_is_rejected_and_not_persisted():
    authority = AsyncMock()
    scope, _ = _scope(authority)
    with pytest.raises(ValueError):
        await runtime._handoff(_logging_obj(scope), _result("other"), NOW, NOW)
    authority.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_avideo_status_persists_wavespeed_completion_end_to_end():
    import httpx

    import litellm
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    class FakeClient(AsyncHTTPHandler):
        def __init__(self):
            pass

        async def get(self, url, headers=None, **kw):
            body = {"code": 200, "data": {"id": "tid1", "status": "completed", "outputs": ["http://x/y.mp4"]}}
            return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    authority = AsyncMock()
    scope, nid = _scope(authority)
    token = runtime.CONTEXT.set(scope)
    try:
        result = await litellm.avideo_status(
            video_id=nid, custom_llm_provider="wavespeed", api_key="x", client=FakeClient(), model="dep"
        )
    finally:
        runtime.CONTEXT.reset(token)
    assert result.id == nid
    events = [call.args[0] for call in authority.persist.await_args_list]
    assert [(e.phase, e.native_id, e.deployment_id, e.finalized) for e in events] == [
        ("completion", nid, "dep-1", True)
    ]
