import litellm

from litellm.proxy.video_endpoints import moderation_bridge

OLD_DETAIL = {
    "error": {
        "code": "provider_not_sent",
        "message": "The provider did not create the generation; no submission was sent",
    }
}


def _rate_limit(**kw):
    return litellm.RateLimitError(message="upstream secret text", model="m", llm_provider="libtv", **kw)


def _assert_capacity(error):
    assert error.status_code == 503
    assert error.headers["x-drama-submission"] == "not_sent"
    body = error.detail["error"]
    assert body["code"] == "provider_not_sent"
    assert body["reason"] == "provider_capacity"
    assert "rate limit" in body["message"] and "算力不足" in body["message"]
    assert "upstream secret text" not in str(error.detail)


def test_no_cause_is_unchanged():
    error = moderation_bridge.provider_not_sent()
    assert error.status_code == 503
    assert error.detail == OLD_DETAIL
    assert error.headers == {"x-drama-submission": "not_sent"}


def test_non_capacity_cause_is_unchanged():
    error = moderation_bridge.provider_not_sent(RuntimeError("boom"))
    assert error.detail == OLD_DETAIL
    assert error.headers == {"x-drama-submission": "not_sent"}


def test_rate_limit_error_is_capacity():
    _assert_capacity(moderation_bridge.provider_not_sent(_rate_limit()))
    assert moderation_bridge.provider_not_sent(_rate_limit()).headers == {"x-drama-submission": "not_sent"}


def test_status_429_exception_is_capacity():
    class E(Exception):
        status_code = 429

    _assert_capacity(moderation_bridge.provider_not_sent(E("x")))


def test_chained_cause_detected():
    try:
        try:
            raise _rate_limit()
        except Exception as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        _assert_capacity(moderation_bridge.provider_not_sent(outer))


def test_chained_context_detected():
    try:
        try:
            raise _rate_limit()
        except Exception:
            raise ValueError("during handling")
    except ValueError as outer:
        _assert_capacity(moderation_bridge.provider_not_sent(outer))


def test_cycle_is_safe_and_depth_bounded():
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__, b.__cause__ = b, a
    assert moderation_bridge.provider_not_sent(a).detail == OLD_DETAIL
    head = _rate_limit()
    for _ in range(50):
        wrapper = RuntimeError("w")
        wrapper.__cause__ = head
        head = wrapper
    assert moderation_bridge.provider_not_sent(head).detail == OLD_DETAIL


def test_retry_after_only_when_exposed():
    error = moderation_bridge.provider_not_sent(_rate_limit())
    assert "retry-after" not in {k.lower() for k in error.headers}

    class E(Exception):
        status_code = 429
        headers = {"Retry-After": "7"}

    error = moderation_bridge.provider_not_sent(E())
    assert error.headers["Retry-After"] == "7"
    assert error.headers["x-drama-submission"] == "not_sent"
    bad = E()
    bad.headers = {"Retry-After": "soon\r\nX: y"}
    assert "Retry-After" not in moderation_bridge.provider_not_sent(bad).headers
