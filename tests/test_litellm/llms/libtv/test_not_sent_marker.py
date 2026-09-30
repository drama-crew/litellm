"""Contract §9: LibTV failures that provably happen before a generation exists."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import litellm.llms.causyn  # noqa: F401  (pre-existing import cycle, see test_ref2va_enqueue_stream)
from litellm.exceptions import BadGatewayError, RateLimitError
from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.handler import _raise_normalized_libtv_error, _video_usage
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


def test_normalized_exception_carries_marker_and_unmarked_stays_unmarked():
    marked = LibTVError(429, "refused")
    marked.submission = "not_sent"
    with pytest.raises(RateLimitError) as caught:
        _raise_normalized_libtv_error(marked, "seedance")
    assert caught.value.drama_submission == "not_sent"
    with pytest.raises(BadGatewayError) as caught:
        _raise_normalized_libtv_error(LibTVError(502, "boom"), "seedance")
    assert getattr(caught.value, "drama_submission", None) is None


def _marked(exc_type=BadGatewayError):
    marked = LibTVError(502, "no login")
    marked.submission = "not_sent"
    with pytest.raises(exc_type) as caught:
        _raise_normalized_libtv_error(marked, "seedance")
    return caught.value


def _unmarked():
    with pytest.raises(BadGatewayError) as caught:
        _raise_normalized_libtv_error(LibTVError(502, "boom"), "seedance")
    return caught.value


def test_failure_outcome_not_sent_when_every_attempt_is_marked():
    token = attempt_outcomes.start()
    try:
        first, second = _marked(), _marked(RateLimitError if False else BadGatewayError)
        attempt_outcomes.record(first)
        attempt_outcomes.record(second)
        assert execution.failure_outcome(second) == "not_sent"
    finally:
        attempt_outcomes.stop(token)


def test_failure_outcome_ambiguous_when_any_attempt_is_unmarked():
    token = attempt_outcomes.start()
    try:
        first, last = _marked(), _marked()
        attempt_outcomes.record(first)
        attempt_outcomes.record(httpx.ReadTimeout("slow"))
        attempt_outcomes.record(last)
        assert execution.failure_outcome(last) == "ambiguous"
    finally:
        attempt_outcomes.stop(token)


def test_failure_outcome_ambiguous_when_final_error_is_unmarked_even_if_attempts_marked():
    token = attempt_outcomes.start()
    try:
        attempt_outcomes.record(_marked())
        assert execution.failure_outcome(_unmarked()) == "ambiguous"
    finally:
        attempt_outcomes.stop(token)


def test_unmarked_error_stays_ambiguous_without_attempt_tracking():
    assert execution.failure_outcome(_unmarked()) == "ambiguous"


def test_record_is_a_noop_without_a_tracking_scope():
    attempt_outcomes.record(_marked())  # must not raise


@pytest.mark.asyncio
async def test_router_make_call_records_each_attempt():
    from litellm import Router

    router = Router(model_list=[{"model_name": "m", "litellm_params": {"model": "openai/x", "api_key": "k"}}])
    token = attempt_outcomes.start()
    try:
        first, second = _marked(), httpx.ReadTimeout("slow")

        async def fail(exc):
            raise exc

        for exc in (first, second):
            with pytest.raises(type(exc)):
                await router.make_call(lambda **_: fail(exc), model="m")
        assert attempt_outcomes.snapshot() == [True, False]
    finally:
        attempt_outcomes.stop(token)


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
