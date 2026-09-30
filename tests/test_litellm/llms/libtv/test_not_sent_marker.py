"""Contract §9: LibTV failures that provably happen before a generation exists."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import litellm.llms.causyn  # noqa: F401  (pre-existing import cycle, see test_ref2va_enqueue_stream)
from litellm.exceptions import BadGatewayError
from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.handler import _video_usage
from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.router_utils import attempt_outcomes


class _Resp:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._body


def _client() -> LibTVClient:
    return object.__new__(LibTVClient)


def _check(step, resp):
    with pytest.raises(LibTVError) as caught:
        _client()._check(resp, step)
    return caught.value


def test_no_login_during_user_resolution_is_marked_not_sent():
    error = _check("getUserInfo", _Resp(body={"code": 401, "msg": "No login"}))
    assert error.status_code == 502
    assert error.submission == "not_sent"


def test_create_business_refusal_is_marked_not_sent():
    error = _check("generation/create", _Resp(body={"code": 1200000136, "msg": "算力不足"}))
    assert error.status_code == 429
    assert error.submission == "not_sent"


def test_create_http_5xx_is_not_marked():
    assert _check("generation/create", _Resp(status_code=502, text="bad gateway")).submission is None


def test_create_http_4xx_is_not_marked():
    assert _check("generation/create", _Resp(status_code=400, text="bad")).submission is None


def test_status_step_business_error_is_not_marked():
    assert _check("generation/progress", _Resp(body={"code": 500, "msg": "x"})).submission is None


def test_pre_create_http_failure_is_marked():
    assert _check("project/create", _Resp(status_code=503, text="x")).submission == "not_sent"


def test_create_body_with_task_id_and_nonzero_code_is_not_marked():
    body = {"code": 1200000136, "msg": "x", "data": {"taskId": "t1"}}
    assert _check("generation/create", _Resp(body=body)).submission is None


def _boundary(exc):
    """A LibTV provider call that fails with ``exc`` at the real provider boundary."""
    from litellm.llms.libtv.handler import normalize_libtv_errors

    @normalize_libtv_errors
    async def call(*args, **kwargs):
        raise exc

    return call


def _not_sent():
    error = LibTVError(502, "libtv getUserInfo code=401 msg=No login")
    error.submission = "not_sent"
    return error


async def _attempt(router, exc):
    """One router attempt through make_call; ``exc`` None means a non-LibTV failure."""
    try:
        if exc is None:

            async def plain(**_):
                raise RuntimeError("routing failure")

            await router.make_call(plain, model="m")
        else:
            await router.make_call(_boundary(exc), model="m")
    except Exception:  # noqa: BLE001
        pass


def _router():
    from litellm import Router

    return Router(model_list=[{"model_name": "m", "litellm_params": {"model": "openai/x", "api_key": "k"}}])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exceptions, expected",
    [
        ([_not_sent, _not_sent], True),
        ([_not_sent, lambda: httpx.ReadTimeout("slow")], False),
        ([lambda: httpx.ReadTimeout("slow"), _not_sent], False),
        ([_not_sent, lambda: LibTVError(502, "5xx after create")], False),
        ([_not_sent, lambda: None], False),  # non-LibTV attempt raised through the router
    ],
)
async def test_ledger_is_all_not_sent_only_when_every_attempt_is_marked(exceptions, expected):
    router = _router()
    token = attempt_outcomes.start()
    try:
        for factory in exceptions:
            await _attempt(router, factory())
        assert attempt_outcomes.all_not_sent() is expected
    finally:
        attempt_outcomes.stop(token)


@pytest.mark.asyncio
async def test_ledger_ignores_exception_context_chains():
    """The 2nd attempt raised inside the 1st attempt's except block has it as __context__."""
    router = _router()
    token = attempt_outcomes.start()
    try:
        try:
            await router.make_call(_boundary(_not_sent()), model="m")
        except Exception:  # noqa: BLE001
            await router.make_call(_boundary(httpx.ReadTimeout("slow")), model="m")
    except Exception:  # noqa: BLE001
        pass
    else:
        raise AssertionError("expected failure")
    try:
        assert attempt_outcomes.all_not_sent() is False
    finally:
        attempt_outcomes.stop(token)


@pytest.mark.asyncio
async def test_success_then_failure_inside_one_attempt_is_unknown(monkeypatch):
    router = _router()

    async def ok(**_):
        return object()

    async def boom(**_):
        raise RuntimeError("after provider success")

    monkeypatch.setattr(router, "set_response_headers", boom)
    token = attempt_outcomes.start()
    try:
        await _attempt_plain(router, ok)
        assert attempt_outcomes.all_not_sent() is False
    finally:
        attempt_outcomes.stop(token)


async def _attempt_plain(router, function):
    try:
        await router.make_call(function, model="m")
    except Exception:  # noqa: BLE001
        pass


def test_boundary_and_record_are_noops_without_a_ledger():
    attempt_outcomes.note(attempt_outcomes.NOT_SENT)
    attempt_outcomes.fail_attempt(attempt_outcomes.begin_attempt())
    assert attempt_outcomes.all_not_sent() is False


def test_empty_ledger_is_not_not_sent():
    token = attempt_outcomes.start()
    try:
        assert attempt_outcomes.all_not_sent() is False
    finally:
        attempt_outcomes.stop(token)


def test_failure_outcome_ignores_cause_chain_and_needs_explicit_verdict():
    marked_looking = LibTVError(502, "x")
    marked_looking.submission = "not_sent"
    try:
        raise marked_looking
    except LibTVError as error:
        wrapped = RuntimeError("wrapped")
        wrapped.__cause__ = error
    assert execution.failure_outcome(wrapped) == "ambiguous"
    verdict = BadGatewayError(message="m", model="m", llm_provider="libtv")
    attempt_outcomes.mark_verdict(verdict, True)
    assert execution.failure_outcome(verdict) == "not_sent"


def test_recreate_failure_after_a_first_create_is_not_marked():
    """A first create already issued a taskId, so a later refusal proves nothing."""
    from litellm.llms.libtv.handler import LibTVLLM

    refusal = LibTVError(429, "refused")
    refusal.submission = "not_sent"

    class _Lt:
        calls = 0

        async def acreate(self, *a, **k):
            _Lt.calls += 1
            if _Lt.calls == 1:
                return {"task_id": "t1"}
            raise refusal

    handler = LibTVLLM.__new__(LibTVLLM)
    handler.fresh_asset_retry_attempts = 2
    handler.fresh_asset_retry_wait = 0

    async def guard(lt, task_id):
        return {"status": 3}

    handler._guard_poll_async = guard
    import litellm.llms.libtv.handler as mod

    original = mod._is_fresh_asset_aging_failure
    mod._is_fresh_asset_aging_failure = lambda state: True
    try:
        with pytest.raises(LibTVError) as caught:
            asyncio.run(handler._acreate_with_fresh_asset_retry(_Lt(), "m", "v", {}, "p"))
    finally:
        mod._is_fresh_asset_aging_failure = original
    assert caught.value.submission is None


def test_create_time_usage_uses_merged_generation_params(monkeypatch):
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "s" * 32)
    from litellm.llms.libtv.handler import LibTVLLM

    handler = LibTVLLM.__new__(LibTVLLM)
    created = {"task_id": "t1"}
    vo = handler._build_video_object("m", created, {}, {"duration": 5, "resolution_720": "720p"})
    assert vo.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}
    # explicit optional_params still win
    vo = handler._build_video_object("m", created, {"seconds": 8}, {"duration": 5})
    assert vo.usage["duration_seconds"] == 8.0
    assert _video_usage({}) is None
