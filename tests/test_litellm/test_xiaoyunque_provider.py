import asyncio
import base64
import json
import random
import time
from typing import Any, Dict, Optional

import pytest

from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    ContentPolicyViolationError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.llms.xiaoyunque.client import (
    XiaoyunqueClient,
    _submit_rate_limit_delay,
    decode_composite_task_id,
    encode_composite_task_id,
    parse_query_result,
    parse_submit_run,
    parse_upload_asset_id,
)
from litellm.llms.xiaoyunque.common import (
    AGENT_NAME,
    XiaoyunqueContentPolicyError,
    XiaoyunqueError,
    build_xiaoyunque_headers,
    build_xiaoyunque_upload_headers,
    is_ak_error,
    is_compliance_ret,
    is_submit_rate_limit_retryable,
    resolve_xiaoyunque_credentials,
)
from litellm.llms.xiaoyunque.handler import (
    FB3_PROVIDER,
    XiaoyunqueLLM,
    _collect_reference_groups,
    _raise_normalized_xiaoyunque_error,
    _wants_frames2video,
)
from litellm.llms.xiaoyunque.transform import (
    build_video_part_tool_param,
    resolution_from_size,
    resolve_resolution,
    size_to_ratio,
)
from litellm.types.videos.utils import (
    VIDEO_ID_PREFIX,
    _add_base64_padding,
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

# ---------------------------------------------------------------------------
# fakes: dependency-injected fake HTTP clients keyed by request path, mirroring
# tests/local_testing/test_libtv_provider.py's style. No monkeypatching of classes.
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSyncClient:
    def __init__(self, post_by_path=None):
        self.post_by_path = post_by_path or {}
        self.calls = []

    def _path(self, url):
        return url.split("xyq.jianying.com", 1)[-1]

    def post(self, url, json=None, headers=None, timeout=None, files=None):
        path = self._path(url)
        self.calls.append((path, json, files))
        queue = self.post_by_path[path]
        item = queue.pop(0) if isinstance(queue, list) else queue
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append(("GET", url, None))
        return FakeResponse({})


class FakeAsyncClient:
    def __init__(self, post_by_path=None):
        self.post_by_path = post_by_path or {}
        self.calls = []

    def _path(self, url):
        return url.split("xyq.jianying.com", 1)[-1]

    async def post(self, url, json=None, headers=None, timeout=None, files=None):
        path = self._path(url)
        self.calls.append((path, json, files))
        queue = self.post_by_path[path]
        item = queue.pop(0) if isinstance(queue, list) else queue
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)

    async def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append(("GET", url, None))
        return FakeResponse({})


class FakeBillingPersistence:
    def __init__(self, billed: bool = True, raises: bool = False, stored_usage=None):
        self.billed = billed
        self.raises = raises
        self.stored_usage = stored_usage
        self.calls = []
        self.store_usage_calls = []
        self.get_usage_calls = []

    async def mark_video_billed(self, billing_key, duration_seconds, response_cost):
        self.calls.append((billing_key, duration_seconds, response_cost))
        if self.raises:
            raise RuntimeError("db unreachable")
        return self.billed

    async def store_video_task_usage(self, billing_key, duration_seconds, video_resolution):
        self.store_usage_calls.append((billing_key, duration_seconds, video_resolution))
        if self.raises:
            raise RuntimeError("db unreachable")

    async def get_video_task_usage(self, billing_key):
        self.get_usage_calls.append(billing_key)
        if self.raises:
            raise RuntimeError("db unreachable")
        return self.stored_usage


class FakeCachePersistence:
    def __init__(self):
        self.store: Dict[tuple, str] = {}
        self.store_calls = []
        self.lookup_calls = []

    async def cached_upload(self, account_key, source_key):
        self.lookup_calls.append((account_key, source_key))
        return self.store.get((account_key, source_key))

    async def store_upload(self, account_key, source_key, cdn_url, size_bytes):
        self.store_calls.append((account_key, source_key, cdn_url, size_bytes))
        self.store[(account_key, source_key)] = cdn_url


def _upload_ok(asset_id: str) -> Dict[str, Any]:
    return {"ret": "0", "errmsg": "", "data": {"pippit_asset_id": asset_id}}


def _submit_ok(run_id: str = "run-1", thread_id: str = "thread-1") -> Dict[str, Any]:
    return {"ret": "0", "errmsg": "", "data": {"run": {"run_id": run_id, "thread_id": thread_id, "state": 1}}}


def _rate_limited_16010() -> Dict[str, Any]:
    return {"ret": "16010", "errmsg": "操作过于频繁，请一分钟后再试", "data": {}}


def _no_jitter(_low: float, _high: float) -> float:
    return 0.0


_EXPECTED_RETRY_DELAY = _submit_rate_limit_delay(_no_jitter)


class _RecordingSleep:
    def __init__(self):
        self.calls: list = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class _RecordingAsleep:
    def __init__(self):
        self.calls: list = []

    async def __call__(self, seconds):
        self.calls.append(seconds)


def _query_result_route(run_state, video_urls=None, image_urls=None, fail_reason=None) -> Dict[str, Any]:
    data: Dict[str, Any] = {"run_state": run_state}
    if video_urls is not None:
        data["video_urls"] = video_urls
    if image_urls is not None:
        data["image_urls"] = image_urls
    if fail_reason is not None:
        data["fail_reason"] = fail_reason
    return {"/api/biz/v1/agent/query_generate_video_result": {"ret": "0", "errmsg": "", "data": data}}


def _vid(thread_id="thread-1", run_id="run-1", model_info_id: Optional[str] = None) -> str:
    op = {"model_info": {"id": model_info_id}} if model_info_id else {}
    return XiaoyunqueLLM()._build_video_object("m", thread_id, run_id, op).id


_720P_MODEL_INFO = {"output_cost_per_second_720p": 0.5}


# ---------------------------------------------------------------------------
# headers / credentials
# ---------------------------------------------------------------------------


def test_build_xiaoyunque_headers_carries_bearer_token():
    h = build_xiaoyunque_headers("tok-123")
    assert h["Authorization"] == "Bearer tok-123"
    assert h["Content-Type"] == "application/json"
    assert h["Accept"] == "application/json"


def test_build_xiaoyunque_upload_headers_has_no_content_type():
    h = build_xiaoyunque_upload_headers("tok-123")
    assert h["Authorization"] == "Bearer tok-123"
    assert "Content-Type" not in h


def test_explicit_empty_credential_fails_closed(monkeypatch):
    monkeypatch.setenv("XIAOYUNQUE_TOKEN", "global-token")
    with pytest.raises(XiaoyunqueError) as exc:
        resolve_xiaoyunque_credentials(token="")
    assert exc.value.status_code == 401
    assert resolve_xiaoyunque_credentials(None) == "global-token"


def test_none_credential_falls_back_to_env_and_missing_env_still_fails_closed(monkeypatch):
    monkeypatch.delenv("XIAOYUNQUE_TOKEN", raising=False)
    with pytest.raises(XiaoyunqueError) as exc:
        resolve_xiaoyunque_credentials(None)
    assert exc.value.status_code == 401


class _NetworkForbiddenClient:
    """Injected in place of a real HTTPHandler/AsyncHTTPHandler for tests that claim
    the credential guard raises BEFORE any client method runs. Any call here proves
    that claim false -- and does so hermetically (AssertionError) instead of the
    test actually reaching https://xyq.jianying.com and failing on a live upstream
    body, which is what happened once the guard regressed under mutation."""

    def post(self, *args, **kwargs):
        raise AssertionError("credential guard regressed: reached network POST")

    def get(self, *args, **kwargs):
        raise AssertionError("credential guard regressed: reached network GET")


def test_pool_deployment_missing_explicit_slot_never_borrows_global_account(monkeypatch):
    monkeypatch.setenv("XIAOYUNQUE_TOKEN", "global-token")
    with pytest.raises(AuthenticationError):
        XiaoyunqueLLM().video_generation(
            model="seedance2.0_vision",
            prompt="x",
            api_key=None,
            api_base=None,
            optional_params={"xiaoyunque_require_explicit_credentials": True},
            logging_obj=None,
            client=_NetworkForbiddenClient(),
        )


def test_video_status_missing_credentials_raises_auth_error_before_reaching_client(monkeypatch):
    monkeypatch.delenv("XIAOYUNQUE_TOKEN", raising=False)
    with pytest.raises(AuthenticationError):
        XiaoyunqueLLM().video_status("plain-id", None, None, {}, None, client=_NetworkForbiddenClient())


def test_xiaoyunque_llm_constructor_matches_custom_handlers_zero_arg_contract():
    """Pins the exact construction openhands-multi-acp-image/litellm_custom_handlers.py
    performs at proxy boot: `xiaoyunque_proxy_handler = XiaoyunqueLLM()`. Unlike
    LibTVLLM (which needs poll_interval/poll_max_attempts because it also serves
    synchronous image generation), XiaoyunqueLLM is submit-only and takes no
    constructor arguments at all -- reverting to the libtv-style call signature
    would raise TypeError at proxy import time and take down every LLM call on the
    platform. Deliberately mirror the handlers-file call rather than merely
    checking XiaoyunqueLLM() succeeds, so a signature drift on either side of that
    seam fails this test."""
    XiaoyunqueLLM()
    with pytest.raises(TypeError):
        XiaoyunqueLLM(poll_interval=3.0, poll_max_attempts=160)


# ---------------------------------------------------------------------------
# ret-based error classification (client._check) -- the single biggest deviation
# from every other provider: HTTP status is always 200, errors live in `ret`.
# ---------------------------------------------------------------------------


def test_check_success_returns_payload():
    client = XiaoyunqueClient(token="t")
    payload = {"ret": "0", "errmsg": "", "data": {"pippit_asset_id": "a1"}}
    assert client._check(FakeResponse(payload), "step") == payload


def test_check_coerces_non_string_ret():
    client = XiaoyunqueClient(token="t")
    with pytest.raises(XiaoyunqueError):
        client._check(FakeResponse({"ret": 2, "errmsg": "Ak无效"}), "step")
    assert client._check(FakeResponse({"ret": 0, "errmsg": ""}), "step") == {"ret": 0, "errmsg": ""}


def test_http_200_with_ret_2_raises_regression_pin():
    # Live-verified 2026-08-01: POST query_generate_video_result {} with no auth
    # returns HTTP 200 + this exact body. Pins the deviation: `_check` must read
    # `ret`, not response.status_code, or this silently "succeeds".
    payload = {"ret": "2", "errmsg": "thread_id或run_id不能为空", "svr_time": 1730000000, "log_id": "abc123"}
    client = XiaoyunqueClient(token="t")
    with pytest.raises(XiaoyunqueError) as exc:
        client._check(FakeResponse(payload, status_code=200), "query_result")
    assert exc.value.status_code == 400  # not an AK errmsg -> terminal BadRequestError, not AuthenticationError


@pytest.mark.asyncio
async def test_avideo_status_http_200_ret_2_raises_through_full_handler():
    payload = {"ret": "2", "errmsg": "thread_id或run_id不能为空", "svr_time": 1730000000, "log_id": "abc123"}
    client = FakeAsyncClient(post_by_path={"/api/biz/v1/agent/query_generate_video_result": payload})
    with pytest.raises(BadRequestError):
        await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, {}, None, client=client)


def test_check_passes_through_non_200_http_status():
    client = XiaoyunqueClient(token="t")
    with pytest.raises(XiaoyunqueError) as exc:
        client._check(FakeResponse({}, status_code=503), "step")
    assert exc.value.status_code == 503
    with pytest.raises(ServiceUnavailableError):
        _raise_normalized_xiaoyunque_error(exc.value, "seedance2.0_vision")


@pytest.mark.parametrize(
    ("ret", "errmsg", "expected_exception", "expected_status"),
    [
        ("2", "Ak无效", AuthenticationError, 401),
        ("2", "Ak为空", AuthenticationError, 401),
        ("2", "未查询到有效的Ak明细", AuthenticationError, 401),
        ("2", "Ak明细为空", AuthenticationError, 401),
        ("2", "该Ak未启用", AuthenticationError, 401),
        ("2", "Ak已过期,请更换", AuthenticationError, 401),
        ("2", "模型参数配置不合法：resolution=1080p 仅支持 model=seedance2.0_vision", BadRequestError, 400),
        ("2", "unrecognized param error", BadRequestError, 400),
        ("2", "当前模型参考图片/视频/音频最多支持 9 个，当前传入 12 个", BadRequestError, 400),
        ("2", "音频素材不能单独提交，请同时上传图片或视频素材", BadRequestError, 400),
        ("5", "获取资产信息失败", BadGatewayError, 502),
        ("5", "提交Run任务失败", BadGatewayError, 502),
        ("10", "服务器太火爆了，请稍后再试", RateLimitError, 429),
        ("16010", "操作过于频繁，请一分钟后再试", RateLimitError, 429),
        ("15", "达到当日生成上限", RateLimitError, 429),
        ("12001", "发起安全审核失败", BadGatewayError, 502),
        ("12004", "输入文本内容审核未通过", ContentPolicyViolationError, None),
        ("12005", "上传图片审核未通过", ContentPolicyViolationError, None),
        ("12006", "上传视频审核未通过", ContentPolicyViolationError, None),
        ("12015", "上传音频审核未通过", ContentPolicyViolationError, None),
        ("99", "unexpected upstream code", BadGatewayError, 502),
    ],
)
def test_check_maps_every_documented_ret_to_expected_exception(ret, errmsg, expected_exception, expected_status):
    client = XiaoyunqueClient(token="t")
    with pytest.raises(XiaoyunqueError) as exc_info:
        client._check(FakeResponse({"ret": ret, "errmsg": errmsg, "data": {}}), "step")
    error = exc_info.value
    if expected_status is not None:
        assert error.status_code == expected_status
    if expected_exception is ContentPolicyViolationError:
        assert isinstance(error, XiaoyunqueContentPolicyError)
    with pytest.raises(expected_exception):
        _raise_normalized_xiaoyunque_error(error, "seedance2.0_vision")


def test_is_ak_error_positive_whitelist():
    assert is_ak_error("Ak无效")
    assert is_ak_error("未查询到有效的Ak明细")
    assert is_ak_error("该Ak未启用")
    assert is_ak_error("Ak已过期,请更换")
    assert not is_ak_error("模型参数配置不合法：resolution=1080p 仅支持 model=seedance2.0_vision")
    assert not is_ak_error(None)
    assert not is_ak_error("")


def test_is_compliance_ret_positive_whitelist():
    assert is_compliance_ret("12004")
    assert is_compliance_ret("12005")
    assert is_compliance_ret("12006")
    assert is_compliance_ret("12015")
    assert not is_compliance_ret("12001")
    assert not is_compliance_ret("2")
    assert not is_compliance_ret("0")


# ---------------------------------------------------------------------------
# video_id: composite thread_id~run_id encoding, and the round trip through
# litellm's own base64 video_id envelope.
# ---------------------------------------------------------------------------


def test_composite_task_id_roundtrip():
    thread_id, run_id = "marketing_11111111-2222", "marketing_66666666-7777"
    composite = encode_composite_task_id(thread_id, run_id)
    assert "~" in composite
    assert decode_composite_task_id(composite) == (thread_id, run_id)


def test_composite_task_id_survives_video_id_encode_decode_with_provider():
    thread_id, run_id = "marketing_thread", "marketing_run"
    composite = encode_composite_task_id(thread_id, run_id)
    video_id = encode_video_id_with_provider(composite, FB3_PROVIDER, "fb3-seedance-2-standard-account-1")
    decoded = decode_video_id_with_provider(video_id)
    assert decoded["video_id"] == composite  # not truncated at '~'
    assert decode_composite_task_id(decoded["video_id"]) == (thread_id, run_id)
    assert decoded["model_id"] == "fb3-seedance-2-standard-account-1"
    assert decoded["custom_llm_provider"] == FB3_PROVIDER


def test_decode_composite_task_id_rejects_malformed():
    with pytest.raises(XiaoyunqueError):
        decode_composite_task_id("no-tilde-here")
    with pytest.raises(XiaoyunqueError):
        decode_composite_task_id("~missing-thread")


def test_build_video_object_prefers_model_info_id_over_status_model():
    vo = XiaoyunqueLLM()._build_video_object(
        "seedance2.0_vision",
        "thread-1",
        "run-1",
        {"model_info": {"id": "fb3-seedance-2-standard-account-2"}, "xiaoyunque_status_model": "legacy-name"},
    )
    decoded = decode_video_id_with_provider(vo.id)
    assert decoded["model_id"] == "fb3-seedance-2-standard-account-2"


def test_build_video_object_falls_back_to_status_model():
    vo = XiaoyunqueLLM()._build_video_object("m", "thread-1", "run-1", {"xiaoyunque_status_model": "seedance-2.0"})
    decoded = decode_video_id_with_provider(vo.id)
    assert decoded["model_id"] == "seedance-2.0"


_FORBIDDEN_VENDOR_SUBSTRINGS = ("xiaoyunque", "jianying", "pippit", "xyq", "剪映", "capcut")


def test_video_id_produced_by_provider_leaks_no_vendor_identity():
    """A video_id built by the real production path (_build_video_object, the same
    call video_generation/avideo_generation make) must decode to a blob carrying
    none of the vendor's names, anywhere in the string -- not merely equal "fb3".
    An equality assertion on custom_llm_provider alone would pass just as happily
    if a vendor token slipped back in through some other segment (e.g. a
    deployment id or a future field); only a forbidden-substring scan over the
    full decoded blob actually holds the property this test exists to pin."""
    vo = XiaoyunqueLLM()._build_video_object(
        "seedance2.0_vision",
        "thread-real-123",
        "run-real-456",
        {"model_info": {"id": "fb3-seedance-2-standard-account-1"}},
    )

    cleaned = _add_base64_padding(vo.id.replace(VIDEO_ID_PREFIX, ""))
    decoded_blob = base64.b64decode(cleaned).decode("utf-8")

    for forbidden in _FORBIDDEN_VENDOR_SUBSTRINGS:
        assert forbidden.lower() not in decoded_blob.lower(), (
            f"video id leaks vendor identity via {forbidden!r} in decoded blob: {decoded_blob!r}"
        )


# ---------------------------------------------------------------------------
# submit-only semantics + run_state mapping + failure-is-data-not-exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_avideo_generation_is_submit_only_no_polling():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    vo = await XiaoyunqueLLM().avideo_generation(
        "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5}, None, client=fake
    )
    assert vo.status == "queued"
    assert [c[0] for c in fake.calls] == ["/api/biz/v1/skill/submit_run"]


def test_video_generation_sync_submit_only():
    client = FakeSyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    vo = XiaoyunqueLLM().video_generation(
        "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5}, None, client=client
    )
    assert vo.status == "queued"
    assert [c[0] for c in client.calls] == ["/api/biz/v1/skill/submit_run"]
    _, submit_body, _ = client.calls[0]
    assert submit_body["agent_name"] == AGENT_NAME


@pytest.mark.parametrize(
    ("run_state", "expected"), [(1, "queued"), (2, "in_progress"), (3, "completed"), (4, "failed"), (5, "failed")]
)
def test_video_status_maps_run_state_int(run_state, expected):
    client = FakeSyncClient(
        post_by_path=_query_result_route(run_state, video_urls=["https://x/v.mp4"] if run_state == 3 else None)
    )
    status = XiaoyunqueLLM().video_status(_vid(), "tok", None, {}, None, client=client)
    assert status.status == expected


@pytest.mark.parametrize(
    ("run_state", "expected"),
    [("1", "queued"), ("2", "in_progress"), ("3", "completed"), ("4", "failed"), ("5", "failed")],
)
def test_video_status_maps_run_state_string(run_state, expected):
    client = FakeSyncClient(
        post_by_path=_query_result_route(run_state, video_urls=["https://x/v.mp4"] if run_state == "3" else None)
    )
    status = XiaoyunqueLLM().video_status(_vid(), "tok", None, {}, None, client=client)
    assert status.status == expected


def test_video_status_completed_hidden_params_carry_url():
    client = FakeSyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))
    status = XiaoyunqueLLM().video_status(_vid(), "tok", None, {}, None, client=client)
    assert status._hidden_params["url"] == "https://x/v.mp4"


def test_video_status_completed_falls_back_to_image_urls_when_no_video_urls():
    # image_urls is a real upstream shape, not a hypothetical: some completed runs
    # (e.g. frames2video / non-video outputs) report through image_urls with
    # video_urls absent entirely. Without the `or state.get("image_urls")` fallback
    # this reports completed with url=None, silently losing the result.
    client = FakeSyncClient(post_by_path=_query_result_route(3, image_urls=["https://x/frame.png"]))
    status = XiaoyunqueLLM().video_status(_vid(), "tok", None, {}, None, client=client)
    assert status.status == "completed"
    assert status._hidden_params["url"] == "https://x/frame.png"


def test_video_status_failed_sets_error_data_not_exception():
    # A poll-time generation failure must surface as vo.error data, never a raised
    # exception -- raising here would break router fallback semantics (the router
    # has already returned the submit-time response by the time status is polled).
    fail_reason = {"code": "11001", "message": "积分不足"}
    client = FakeSyncClient(post_by_path=_query_result_route(4, fail_reason=fail_reason))
    status = XiaoyunqueLLM().video_status(_vid(), "tok", None, {}, None, client=client)
    assert status.status == "failed"
    assert status.error["message"] == "积分不足"


def test_video_status_failed_without_fail_reason_has_default_message():
    client = FakeSyncClient(post_by_path=_query_result_route(5))
    status = XiaoyunqueLLM().video_status(_vid(), "tok", None, {}, None, client=client)
    assert status.status == "failed"
    assert status.error["message"]


# ---------------------------------------------------------------------------
# video_content
# ---------------------------------------------------------------------------


class _DownloadResp:
    status_code = 200
    content = b"MP4BYTES"


class _PollAndDownloadSyncClient(FakeSyncClient):
    def __init__(self, run_state, url=None):
        super().__init__(post_by_path=_query_result_route(run_state, video_urls=[url] if url else None))
        self.got = None

    def get(self, url, headers=None, timeout=None, params=None):
        self.got = url
        return _DownloadResp()


class _PollAndDownloadAsyncClient(FakeAsyncClient):
    def __init__(self, run_state, url=None):
        super().__init__(post_by_path=_query_result_route(run_state, video_urls=[url] if url else None))
        self.got = None

    async def get(self, url, headers=None, timeout=None, params=None):
        self.got = url
        return _DownloadResp()


def test_video_content_polls_then_downloads():
    c = _PollAndDownloadSyncClient(3, url="https://x/v.mp4")
    data = XiaoyunqueLLM().video_content(_vid(), "tok", None, {}, None, client=c)
    assert data == b"MP4BYTES"
    assert c.got == "https://x/v.mp4"


@pytest.mark.asyncio
async def test_avideo_content_polls_then_downloads():
    c = _PollAndDownloadAsyncClient(3, url="https://x/v.mp4")
    data = await XiaoyunqueLLM().avideo_content(_vid(), "tok", None, {}, None, client=c)
    assert data == b"MP4BYTES"
    assert c.got == "https://x/v.mp4"


@pytest.mark.asyncio
async def test_avideo_content_raises_while_still_processing():
    c = _PollAndDownloadAsyncClient(2)
    with pytest.raises(APIError) as exc:
        await XiaoyunqueLLM().avideo_content(_vid(), "tok", None, {}, None, client=c)
    assert exc.value.status_code == 409


def test_video_content_completed_without_url_raises_instead_of_empty_bytes():
    # Upstream reported run_state=3 (completed) but no video_urls/image_urls -- this
    # must be a hard 502, never a silent b"". This repo has a live incident class
    # for exactly this shape (silent download truncation corrupted 4/19 artifacts);
    # an empty MP4 reaching the backend is the same failure mode, just at ret=0.
    c = _PollAndDownloadSyncClient(3, url=None)
    with pytest.raises(BadGatewayError) as exc:
        XiaoyunqueLLM().video_content(_vid(), "tok", None, {}, None, client=c)
    assert exc.value.status_code == 502
    assert c.got is None  # must fail before ever attempting a download GET


@pytest.mark.asyncio
async def test_avideo_content_completed_without_url_raises_instead_of_empty_bytes():
    c = _PollAndDownloadAsyncClient(3, url=None)
    with pytest.raises(BadGatewayError) as exc:
        await XiaoyunqueLLM().avideo_content(_vid(), "tok", None, {}, None, client=c)
    assert exc.value.status_code == 502
    assert c.got is None  # must fail before ever attempting a download GET


# ---------------------------------------------------------------------------
# reference collection / mode inference: frames2video, modeType handling,
# reference-intent guard, audio-only guard
# ---------------------------------------------------------------------------


def test_wants_frames2video_from_image_and_last_image_pair():
    assert _wants_frames2video({"image": "https://x/a.png", "last_image": "https://x/b.png"}) is True


def test_wants_frames2video_false_when_only_image():
    assert _wants_frames2video({"image": "https://x/a.png"}) is False


def test_wants_frames2video_explicit_mode_type():
    assert _wants_frames2video({"modeType": "frames2video"}) is True


def test_wants_frames2video_ignores_unrelated_mode_type():
    assert _wants_frames2video({"modeType": "mixed2video", "reference_images": ["a"]}) is False


def test_collect_reference_groups_frames_mode_uses_image_pair_only():
    images, videos, audios, wants_frames = _collect_reference_groups(
        {
            "image": "https://x/first.png",
            "last_image": "https://x/last.png",
            "reference_images": ["https://x/ignored.png"],
        }
    )
    assert wants_frames is True
    assert images == ["https://x/first.png", "https://x/last.png"]


def test_collect_reference_groups_single_image_folds_into_general_bucket():
    images, videos, audios, wants_frames = _collect_reference_groups({"image": "https://x/a.png"})
    assert wants_frames is False
    assert images == ["https://x/a.png"]


def test_collect_reference_groups_mixed2video_mode_type_does_not_trigger_frames():
    images, videos, audios, wants_frames = _collect_reference_groups(
        {"modeType": "mixed2video", "reference_images": ["https://x/a.png", "https://x/b.png"]}
    )
    assert wants_frames is False
    assert images == ["https://x/a.png", "https://x/b.png"]


def test_collect_reference_groups_merges_aliases():
    images, videos, audios, wants_frames = _collect_reference_groups(
        {
            "input_reference": "https://x/a.png",
            "image_references": ["https://x/b.png"],
            "reference_images": ["https://x/c.png"],
            "video_references": ["https://x/v.mp4"],
            "audio_references": ["https://x/s.mp3"],
        }
    )
    assert images == ["https://x/a.png", "https://x/b.png", "https://x/c.png"]
    assert videos == ["https://x/v.mp4"]
    assert audios == ["https://x/s.mp3"]
    assert wants_frames is False


@pytest.mark.asyncio
async def test_avideo_generation_skips_none_entry_in_reference_list_without_crashing():
    # A None entry mid-list must be dropped, not unpacked into ensure_uploaded(*None, ...)
    # (which raises a bare TypeError instead of a clean, user-facing error).
    fake = FakeAsyncClient(
        post_by_path={
            "/api/biz/v1/skill/upload_file": [_upload_ok("asset-a"), _upload_ok("asset-b")],
            "/api/biz/v1/skill/submit_run": _submit_ok(),
        }
    )
    vo = await XiaoyunqueLLM(http_get=lambda url: b"bytes").avideo_generation(
        "seedance2.0_vision",
        "a cat",
        "tok",
        None,
        # None must sit after index 1: _as_list's own tuple-vs-list heuristic (str at [0],
        # non-str at [1] => treat as a single (name, bytes) tuple) is a separate, pre-existing
        # ambiguity inherited from libtv and out of scope here.
        {"reference_images": ["https://x/a.png", "https://x/b.png", None]},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    submit_body = next(body for path, body, _files in fake.calls if path == "/api/biz/v1/skill/submit_run")
    assert submit_body["asset_ids"] == ["asset-a", "asset-b"]


@pytest.mark.asyncio
async def test_avideo_generation_reference_key_present_but_empty_raises_not_text2video():
    fake = FakeAsyncClient(post_by_path={})
    with pytest.raises(BadRequestError):
        await XiaoyunqueLLM().avideo_generation(
            "seedance2.0_vision", "a cat", "tok", None, {"reference_images": []}, None, client=fake
        )
    assert fake.calls == []  # never even tried to submit


@pytest.mark.asyncio
async def test_avideo_generation_text_only_never_triggers_guard():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    vo = await XiaoyunqueLLM().avideo_generation(
        "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5}, None, client=fake
    )
    assert vo.status == "queued"


@pytest.mark.asyncio
async def test_avideo_generation_audio_only_fails_fast():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/upload_file": _upload_ok("asset-aud")})
    with pytest.raises(BadRequestError):
        await XiaoyunqueLLM(http_get=lambda url: b"bytes").avideo_generation(
            "seedance2.0_vision", "a cat", "tok", None, {"reference_audios": ["https://x/a.mp3"]}, None, client=fake
        )
    assert fake.calls == []  # guard runs before any upload


@pytest.mark.asyncio
async def test_avideo_generation_audio_with_image_passes_guard():
    fake = FakeAsyncClient(
        post_by_path={
            "/api/biz/v1/skill/upload_file": [_upload_ok("asset-img"), _upload_ok("asset-aud")],
            "/api/biz/v1/skill/submit_run": _submit_ok(),
        }
    )
    vo = await XiaoyunqueLLM(http_get=lambda url: b"bytes").avideo_generation(
        "seedance2.0_vision",
        "a cat",
        "tok",
        None,
        {"reference_images": ["https://x/a.png"], "reference_audios": ["https://x/s.mp3"]},
        None,
        client=fake,
    )
    assert vo.status == "queued"


# ---------------------------------------------------------------------------
# reference upload -> asset_id -> payload shape (images/videos/audios + top-level
# asset_ids), frames2video generate_type, and coordinator-flagged real payload
# quirks: string `seconds`, ignored `generate_audio`/`seed`/unrelated `modeType`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_avideo_generation_full_payload_shape_and_ignores_unknown_params():
    routes = {
        "/api/biz/v1/skill/upload_file": [
            _upload_ok("asset-img"),
            _upload_ok("asset-vid"),
            _upload_ok("asset-aud"),
        ],
        "/api/biz/v1/skill/submit_run": _submit_ok(),
    }
    fake = FakeAsyncClient(post_by_path=routes)
    vo = await XiaoyunqueLLM(http_get=lambda url: b"bytes").avideo_generation(
        "seedance2.0_vision",
        "a cat",
        "tok",
        None,
        {
            "reference_images": ["https://x/a.png"],
            "reference_videos": ["https://x/v.mp4"],
            "reference_audios": ["https://x/s.mp3"],
            "seconds": "8",  # the drama backend sends duration as a STRING
            "generate_audio": True,
            "seed": 12345,
            "modeType": "mixed2video",
        },
        None,
        client=fake,
    )
    assert vo.status == "queued"
    submit_body = next(body for path, body, _files in fake.calls if path == "/api/biz/v1/skill/submit_run")
    assert submit_body["agent_name"] == AGENT_NAME
    assert set(submit_body["asset_ids"]) == {"asset-img", "asset-vid", "asset-aud"}
    vptp = submit_body["video_part_tool_param"]
    assert vptp["images"] == [{"pippit_asset_id": "asset-img"}]
    assert vptp["videos"] == [{"pippit_asset_id": "asset-vid"}]
    assert vptp["audios"] == [{"pippit_asset_id": "asset-aud"}]
    assert vptp["duration_sec"] == 8
    assert isinstance(vptp["duration_sec"], int)
    assert "generate_audio" not in vptp
    assert "seed" not in vptp
    assert "modeType" not in vptp
    assert "generate_type" not in vptp  # mixed2video has no xiaoyunque equivalent -- must not leak through


@pytest.mark.asyncio
async def test_avideo_generation_frames2video_sets_generate_type_and_image_pair():
    routes = {
        "/api/biz/v1/skill/upload_file": [_upload_ok("asset-first"), _upload_ok("asset-last")],
        "/api/biz/v1/skill/submit_run": _submit_ok(run_id="run-2", thread_id="thread-2"),
    }
    fake = FakeAsyncClient(post_by_path=routes)
    vo = await XiaoyunqueLLM(http_get=lambda url: b"bytes").avideo_generation(
        "seedance2.0_fast_vision",
        "a cat",
        "tok",
        None,
        {"image": "https://x/first.png", "last_image": "https://x/last.png", "seconds": 5},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    submit_body = next(body for path, body, _files in fake.calls if path == "/api/biz/v1/skill/submit_run")
    vptp = submit_body["video_part_tool_param"]
    assert vptp["generate_type"] == 1
    assert vptp["images"] == [{"pippit_asset_id": "asset-first"}, {"pippit_asset_id": "asset-last"}]


@pytest.mark.asyncio
async def test_avideo_generation_text_only_omits_asset_ids_and_reference_fields():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    await XiaoyunqueLLM().avideo_generation(
        "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5}, None, client=fake
    )
    _, body, _ = fake.calls[0]
    assert "asset_ids" not in body
    vptp = body["video_part_tool_param"]
    assert "images" not in vptp and "videos" not in vptp and "audios" not in vptp and "generate_type" not in vptp


@pytest.mark.asyncio
async def test_aupload_file_uses_field_name_file():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/upload_file": _upload_ok("asset-1")})
    client = XiaoyunqueClient(token="t", async_client=fake)
    asset_id = await client.aupload_file(b"bytes", "ref.png")
    assert asset_id == "asset-1"
    _, _, files = fake.calls[0]
    assert files == {"file": ("ref.png", b"bytes")}


# ---------------------------------------------------------------------------
# 1080p downgrade guard
# ---------------------------------------------------------------------------


def test_resolve_resolution_downgrades_non_vision_model():
    warnings = []
    result = resolve_resolution("seedance2.0_fast_vision", "1080p", warn=lambda *a: warnings.append(a))
    assert result == "720p"
    assert warnings


def test_resolve_resolution_keeps_1080p_for_vision_model():
    def _fail_if_called(*a):
        raise AssertionError("must not warn for the model that actually supports 1080p")

    assert resolve_resolution("seedance2.0_vision", "1080p", warn=_fail_if_called) == "1080p"


def test_resolve_resolution_passes_through_non_1080p_untouched():
    assert resolve_resolution("seedance2.0_fast_vision", "720p", warn=lambda *a: None) == "720p"
    assert resolve_resolution("seedance2.0_fast_vision", None, warn=lambda *a: None) is None


@pytest.mark.asyncio
async def test_avideo_generation_downgrades_1080p_before_sending_upstream():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    await XiaoyunqueLLM().avideo_generation(
        "seedance2.0_fast_vision", "a cat", "tok", None, {"seconds": 5, "resolution": "1080p"}, None, client=fake
    )
    _, body, _ = fake.calls[0]
    assert body["video_part_tool_param"]["resolution"] == "720p"


@pytest.mark.asyncio
async def test_avideo_generation_keeps_1080p_for_vision_model():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    await XiaoyunqueLLM().avideo_generation(
        "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5, "resolution": "1080p"}, None, client=fake
    )
    _, body, _ = fake.calls[0]
    assert body["video_part_tool_param"]["resolution"] == "1080p"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["Seedance_2.0_mini", "Seedance_2.5"])
async def test_avideo_generation_accepts_new_model_slug(model):
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    vo = await XiaoyunqueLLM().avideo_generation(model, "a cat", "tok", None, {"seconds": 5}, None, client=fake)
    assert vo.status == "queued"
    assert vo.model == model
    _, body, _ = fake.calls[0]
    assert body["video_part_tool_param"]["model"] == model


@pytest.mark.parametrize("model", ["Seedance_2.0_mini", "Seedance_2.5"])
def test_resolve_resolution_downgrades_new_slugs_at_1080p(model):
    warnings = []
    assert resolve_resolution(model, "1080p", warn=lambda *a: warnings.append(a)) == "720p"
    assert warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["Seedance_2.0_mini", "Seedance_2.5"])
async def test_avideo_generation_downgrades_1080p_for_new_slugs(model):
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    await XiaoyunqueLLM().avideo_generation(
        model, "a cat", "tok", None, {"seconds": 5, "resolution": "1080p"}, None, client=fake
    )
    _, body, _ = fake.calls[0]
    assert body["video_part_tool_param"]["resolution"] == "720p"


# ---------------------------------------------------------------------------
# transform.py pure functions
# ---------------------------------------------------------------------------


def test_size_to_ratio():
    assert size_to_ratio("1280x720") == "16:9"
    assert size_to_ratio("1080x1920") == "9:16"
    assert size_to_ratio("1024x1024") == "1:1"
    assert size_to_ratio(None) is None
    assert size_to_ratio("garbage") is None


def test_resolution_from_size():
    assert resolution_from_size("1920x1080") == "1080p"
    assert resolution_from_size("1280x720") == "720p"
    assert resolution_from_size("640x480") == "480p"
    assert resolution_from_size(None) is None


def test_build_video_part_tool_param_coerces_string_duration():
    params = build_video_part_tool_param(
        "p", "seedance2.0_vision", {"seconds": "8"}, [], [], [], None, warn=lambda *a: None
    )
    assert params["duration_sec"] == 8
    assert isinstance(params["duration_sec"], int)


def test_build_video_part_tool_param_default_duration_when_absent():
    params = build_video_part_tool_param("p", "seedance2.0_vision", {}, [], [], [], None, warn=lambda *a: None)
    assert params["duration_sec"] == 5


def test_build_video_part_tool_param_ratio_precedence():
    warn = lambda *a: None  # noqa: E731
    assert (
        build_video_part_tool_param("x", "seedance2.0_vision", {"aspect_ratio": "9:16"}, [], [], [], None, warn)[
            "ratio"
        ]
        == "9:16"
    )
    assert (
        build_video_part_tool_param(
            "x", "seedance2.0_vision", {"ratio": "1:1", "aspect_ratio": "9:16"}, [], [], [], None, warn
        )["ratio"]
        == "1:1"
    )
    assert (
        build_video_part_tool_param("x", "seedance2.0_vision", {"size": "1280x720"}, [], [], [], None, warn)["ratio"]
        == "16:9"
    )


def test_build_video_part_tool_param_asset_ref_shape():
    params = build_video_part_tool_param(
        "x", "seedance2.0_vision", {}, ["a1", "a2"], ["v1"], ["s1"], None, warn=lambda *a: None
    )
    assert params["images"] == [{"pippit_asset_id": "a1"}, {"pippit_asset_id": "a2"}]
    assert params["videos"] == [{"pippit_asset_id": "v1"}]
    assert params["audios"] == [{"pippit_asset_id": "s1"}]


# ---------------------------------------------------------------------------
# parse_* pure functions
# ---------------------------------------------------------------------------


def test_parse_upload_asset_id():
    assert parse_upload_asset_id({"data": {"pippit_asset_id": "a1"}}) == "a1"
    with pytest.raises(XiaoyunqueError):
        parse_upload_asset_id({"data": {}})


def test_parse_submit_run():
    assert parse_submit_run({"data": {"run": {"run_id": "r1", "thread_id": "t1", "state": 1}}}) == {
        "thread_id": "t1",
        "run_id": "r1",
    }
    with pytest.raises(XiaoyunqueError):
        parse_submit_run({"data": {"run": {}}})


def test_parse_query_result_defaults():
    assert parse_query_result({"data": {}}) == {
        "run_state": None,
        "video_urls": [],
        "image_urls": [],
        "fail_reason": None,
    }


def test_parse_query_result_coerces_string_run_state():
    assert parse_query_result({"data": {"run_state": "3", "video_urls": ["https://x/v.mp4"]}})["run_state"] == 3


# ---------------------------------------------------------------------------
# upload cache: hit / miss / disabled, and dedup across differing presigned URLs
# for the same object path (query/fragment stripped -- presigned URLs re-sign on
# every call but the underlying bytes are unchanged).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aensure_uploaded_cache_hit_skips_reupload():
    persistence = FakeCachePersistence()
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/upload_file": [_upload_ok("asset-1")]})
    client = XiaoyunqueClient(token="t", async_client=fake, http_get=lambda url: b"bytes", persistence=persistence)

    first = await client.aensure_uploaded("url", "https://oss.example.com/path/obj.png?sig=aaa", None, "ref.png")
    second = await client.aensure_uploaded("url", "https://oss.example.com/path/obj.png?sig=bbb", None, "ref.png")

    assert first == second == "asset-1"
    assert len(fake.calls) == 1  # only the first call actually uploaded
    assert persistence.store_calls == [(client._account_key, "oss.example.com/path/obj.png", "asset-1", 0)]
    assert client._account_key.startswith("xiaoyunque:")


@pytest.mark.asyncio
async def test_aensure_uploaded_cache_miss_for_different_object_uploads_twice():
    persistence = FakeCachePersistence()
    fake = FakeAsyncClient(
        post_by_path={"/api/biz/v1/skill/upload_file": [_upload_ok("asset-1"), _upload_ok("asset-2")]}
    )
    client = XiaoyunqueClient(token="t", async_client=fake, http_get=lambda url: b"bytes", persistence=persistence)

    first = await client.aensure_uploaded("url", "https://oss.example.com/a.png", None, "ref.png")
    second = await client.aensure_uploaded("url", "https://oss.example.com/b.png", None, "ref.png")

    assert (first, second) == ("asset-1", "asset-2")
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_aensure_uploaded_cache_disabled_env_always_reuploads(monkeypatch):
    monkeypatch.setenv("XIAOYUNQUE_UPLOAD_CACHE_DISABLED", "1")
    persistence = FakeCachePersistence()
    fake = FakeAsyncClient(
        post_by_path={"/api/biz/v1/skill/upload_file": [_upload_ok("asset-1"), _upload_ok("asset-2")]}
    )
    client = XiaoyunqueClient(token="t", async_client=fake, http_get=lambda url: b"bytes", persistence=persistence)

    first = await client.aensure_uploaded("url", "https://oss.example.com/path/obj.png?sig=aaa", None, "ref.png")
    second = await client.aensure_uploaded("url", "https://oss.example.com/path/obj.png?sig=bbb", None, "ref.png")

    assert (first, second) == ("asset-1", "asset-2")  # cache bypassed -> reuploaded despite same object path
    assert len(fake.calls) == 2
    assert persistence.store_calls == []


def test_ensure_uploaded_sync_has_no_cache_by_design():
    # LibTVPersistence is async-only; the sync path (unused by the production proxy,
    # which routes exclusively through avideo_generation) never touches persistence.
    fake = FakeSyncClient(
        post_by_path={"/api/biz/v1/skill/upload_file": [_upload_ok("asset-1"), _upload_ok("asset-2")]}
    )
    persistence = FakeCachePersistence()
    client = XiaoyunqueClient(token="t", sync_client=fake, http_get=lambda url: b"bytes", persistence=persistence)

    first = client.ensure_uploaded("url", "https://oss.example.com/a.png", None, "ref.png")
    second = client.ensure_uploaded("url", "https://oss.example.com/a.png", None, "ref.png")

    assert (first, second) == ("asset-1", "asset-2")
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# billing: create-time record, completion-time bill, idempotency, no-usage /
# no-price gaps warn-and-skip.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_avideo_generation_records_task_usage_for_status_billing(monkeypatch):
    fake_persistence = FakeBillingPersistence()
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})

    vo = await XiaoyunqueLLM().avideo_generation(
        "seedance2.0_vision", "a fox", "tok", None, {"seconds": 8, "resolution": "720p"}, None, client=fake
    )

    assert vo.status == "queued"
    assert fake_persistence.store_usage_calls == [("xiaoyunque:thread-1~run-1", 8.0, "720p")]


@pytest.mark.asyncio
async def test_avideo_generation_records_default_duration_when_request_has_none(monkeypatch):
    fake_persistence = FakeBillingPersistence()
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})

    await XiaoyunqueLLM().avideo_generation("seedance2.0_vision", "a fox", "tok", None, {}, None, client=fake)

    assert fake_persistence.store_usage_calls == [("xiaoyunque:thread-1~run-1", 5.0, None)]


@pytest.mark.asyncio
async def test_avideo_generation_store_usage_error_does_not_break_create(monkeypatch):
    fake_persistence = FakeBillingPersistence(raises=True)
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})

    vo = await XiaoyunqueLLM().avideo_generation(
        "seedance2.0_vision", "a fox", "tok", None, {"seconds": 8}, None, client=fake
    )
    assert vo.status == "queued"


@pytest.mark.asyncio
async def test_avideo_status_completed_bills_via_persistence_on_first_poll(monkeypatch):
    fake_persistence = FakeBillingPersistence(billed=True)
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    client = FakeAsyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))
    optional_params = {"seconds": 5, "resolution": "720p", "model_info": _720P_MODEL_INFO}

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, optional_params, None, client=client)

    assert status.status == "completed"
    assert status.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}
    assert status._hidden_params["response_cost"] == pytest.approx(2.5)
    assert fake_persistence.calls == [("xiaoyunque:thread-1~run-1", 5.0, pytest.approx(2.5))]


@pytest.mark.asyncio
async def test_avideo_status_completed_repeat_poll_does_not_double_bill(monkeypatch):
    fake_persistence = FakeBillingPersistence(billed=False)
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    client = FakeAsyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))
    optional_params = {"seconds": 5, "resolution": "720p", "model_info": _720P_MODEL_INFO}

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, optional_params, None, client=client)

    assert status.status == "completed"
    assert status._hidden_params["response_cost"] == 0.0


@pytest.mark.asyncio
async def test_avideo_status_completed_without_duration_skips_billing(monkeypatch):
    fake_persistence = FakeBillingPersistence(billed=True)
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    client = FakeAsyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, {}, None, client=client)

    assert status.status == "completed"
    assert "response_cost" not in status._hidden_params
    assert fake_persistence.calls == []


@pytest.mark.asyncio
async def test_avideo_status_completed_no_price_for_resolution_skips_billing(monkeypatch):
    fake_persistence = FakeBillingPersistence(billed=True)
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    client = FakeAsyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))
    optional_params = {"seconds": 5, "resolution": "720p", "model_info": {}}

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, optional_params, None, client=client)

    assert status.status == "completed"
    assert "response_cost" not in status._hidden_params
    assert fake_persistence.calls == []


@pytest.mark.asyncio
async def test_avideo_status_persistence_unavailable_skips_billing(monkeypatch):
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: None)
    client = FakeAsyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))
    optional_params = {"seconds": 5, "resolution": "720p", "model_info": _720P_MODEL_INFO}

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, optional_params, None, client=client)

    assert status.status == "completed"
    assert "response_cost" not in status._hidden_params


@pytest.mark.asyncio
async def test_avideo_status_persistence_error_does_not_raise_and_skips_billing(monkeypatch):
    fake_persistence = FakeBillingPersistence(raises=True)
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    client = FakeAsyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))
    optional_params = {"seconds": 5, "resolution": "720p", "model_info": _720P_MODEL_INFO}

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, optional_params, None, client=client)

    assert status.status == "completed"
    assert "response_cost" not in status._hidden_params


@pytest.mark.asyncio
async def test_avideo_status_in_progress_never_calls_persistence(monkeypatch):
    fake_persistence = FakeBillingPersistence(billed=True)
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    client = FakeAsyncClient(post_by_path=_query_result_route(1))
    optional_params = {"seconds": 5, "resolution": "720p", "model_info": _720P_MODEL_INFO}

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, optional_params, None, client=client)

    assert status.status == "queued"
    assert fake_persistence.calls == []


@pytest.mark.asyncio
async def test_avideo_status_production_shape_bills_from_persisted_usage(monkeypatch):
    fake_persistence = FakeBillingPersistence(
        billed=True, stored_usage={"duration_seconds": 5.0, "video_resolution": "720p"}
    )
    monkeypatch.setattr("litellm.llms.xiaoyunque.handler.get_persistence", lambda: fake_persistence)
    client = FakeAsyncClient(post_by_path=_query_result_route(3, video_urls=["https://x/v.mp4"]))
    optional_params = {"model": "seedance-2.0", "custom_llm_provider": "fb3", "model_info": _720P_MODEL_INFO}

    status = await XiaoyunqueLLM().avideo_status(_vid(), "tok", None, optional_params, None, client=client)

    assert status.status == "completed"
    assert status.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}
    assert status._hidden_params["response_cost"] == pytest.approx(2.5)
    assert fake_persistence.get_usage_calls == ["xiaoyunque:thread-1~run-1"]
    assert fake_persistence.calls == [("xiaoyunque:thread-1~run-1", 5.0, pytest.approx(2.5))]


# ---------------------------------------------------------------------------
# malformed video id through the full handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_avideo_status_malformed_composite_id_raises_bad_request(monkeypatch):
    monkeypatch.setenv("XIAOYUNQUE_TOKEN", "tok")
    video_id = encode_video_id_with_provider("no-tilde-here", FB3_PROVIDER, "some-model")
    with pytest.raises(BadRequestError):
        await XiaoyunqueLLM().avideo_status(video_id, None, None, {}, None, client=FakeAsyncClient())


def test_client_defaults_sleep_functions_when_not_injected():
    client = XiaoyunqueClient(token="t")
    assert client._sleep is time.sleep
    assert client._asleep is asyncio.sleep


def test_make_client_forwards_retry_dependencies():
    sleep = _RecordingSleep()
    asleep = _RecordingAsleep()
    llm = XiaoyunqueLLM(sleep=sleep, asleep=asleep, jitter=_no_jitter)
    client = llm._make_client(api_key="tok", optional_params={})
    assert client._sleep is sleep
    assert client._asleep is asleep
    assert client._jitter is _no_jitter


def test_is_submit_rate_limit_retryable():
    assert is_submit_rate_limit_retryable("16010") is True
    assert is_submit_rate_limit_retryable("10") is False
    assert is_submit_rate_limit_retryable("15") is False
    assert is_submit_rate_limit_retryable("2") is False
    assert is_submit_rate_limit_retryable(None) is False


def test_check_sets_ret_on_nonzero_ret_error():
    client = XiaoyunqueClient(token="t")
    with pytest.raises(XiaoyunqueError) as exc:
        client._check(FakeResponse({"ret": "16010", "errmsg": "操作过于频繁，请一分钟后再试"}), "submit_run")
    assert exc.value.ret == "16010"


def test_check_leaves_ret_none_for_non_200_http_status():
    client = XiaoyunqueClient(token="t")
    with pytest.raises(XiaoyunqueError) as exc:
        client._check(FakeResponse({}, status_code=503), "submit_run")
    assert exc.value.ret is None


@pytest.mark.asyncio
async def test_asubmit_run_succeeds_first_try_never_sleeps():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": _submit_ok()})
    asleep = _RecordingAsleep()
    client = XiaoyunqueClient(token="t", async_client=fake, asleep=asleep, jitter=_no_jitter)
    result = await client.asubmit_run(message="a cat", asset_ids=[], video_part_tool_param={"model": "m"})
    assert result == {"thread_id": "thread-1", "run_id": "run-1"}
    assert len(fake.calls) == 1
    assert asleep.calls == []


@pytest.mark.asyncio
async def test_asubmit_run_retries_on_16010_then_succeeds():
    fake = FakeAsyncClient(
        post_by_path={"/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _rate_limited_16010(), _submit_ok()]}
    )
    asleep = _RecordingAsleep()
    client = XiaoyunqueClient(token="t", async_client=fake, asleep=asleep, jitter=_no_jitter)
    result = await client.asubmit_run(message="a cat", asset_ids=[], video_part_tool_param={"model": "m"})
    assert result == {"thread_id": "thread-1", "run_id": "run-1"}
    assert len(fake.calls) == 3
    assert asleep.calls == [_EXPECTED_RETRY_DELAY, _EXPECTED_RETRY_DELAY]


@pytest.mark.asyncio
async def test_asubmit_run_exhausts_retries_then_raises_rate_limit():
    fake = FakeAsyncClient(
        post_by_path={
            "/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _rate_limited_16010(), _rate_limited_16010()]
        }
    )
    asleep = _RecordingAsleep()
    client = XiaoyunqueClient(token="t", async_client=fake, asleep=asleep, jitter=_no_jitter)
    with pytest.raises(XiaoyunqueError) as exc:
        await client.asubmit_run(message="a cat", asset_ids=[], video_part_tool_param={"model": "m"})
    assert exc.value.status_code == 429
    assert exc.value.ret == "16010"
    assert len(fake.calls) == 3
    assert asleep.calls == [_EXPECTED_RETRY_DELAY, _EXPECTED_RETRY_DELAY]


@pytest.mark.asyncio
async def test_asubmit_run_does_not_retry_other_rate_limit_ret():
    fake = FakeAsyncClient(
        post_by_path={"/api/biz/v1/skill/submit_run": {"ret": "10", "errmsg": "服务器太火爆了，请稍后再试", "data": {}}}
    )
    asleep = _RecordingAsleep()
    client = XiaoyunqueClient(token="t", async_client=fake, asleep=asleep, jitter=_no_jitter)
    with pytest.raises(XiaoyunqueError) as exc:
        await client.asubmit_run(message="a cat", asset_ids=[], video_part_tool_param={"model": "m"})
    assert exc.value.status_code == 429
    assert len(fake.calls) == 1
    assert asleep.calls == []


@pytest.mark.asyncio
async def test_asubmit_run_does_not_retry_bad_request_ret():
    fake = FakeAsyncClient(
        post_by_path={"/api/biz/v1/skill/submit_run": {"ret": "2", "errmsg": "unrecognized param error", "data": {}}}
    )
    asleep = _RecordingAsleep()
    client = XiaoyunqueClient(token="t", async_client=fake, asleep=asleep, jitter=_no_jitter)
    with pytest.raises(XiaoyunqueError) as exc:
        await client.asubmit_run(message="a cat", asset_ids=[], video_part_tool_param={"model": "m"})
    assert exc.value.status_code == 400
    assert len(fake.calls) == 1
    assert asleep.calls == []


def test_submit_run_sync_retries_on_16010_then_succeeds():
    fake = FakeSyncClient(post_by_path={"/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _submit_ok()]})
    sleep = _RecordingSleep()
    client = XiaoyunqueClient(token="t", sync_client=fake, sleep=sleep, jitter=_no_jitter)
    result = client.submit_run(message="a cat", asset_ids=[], video_part_tool_param={"model": "m"})
    assert result == {"thread_id": "thread-1", "run_id": "run-1"}
    assert len(fake.calls) == 2
    assert sleep.calls == [_EXPECTED_RETRY_DELAY]


def test_submit_run_sync_exhausts_retries_then_raises():
    fake = FakeSyncClient(
        post_by_path={
            "/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _rate_limited_16010(), _rate_limited_16010()]
        }
    )
    sleep = _RecordingSleep()
    client = XiaoyunqueClient(token="t", sync_client=fake, sleep=sleep, jitter=_no_jitter)
    with pytest.raises(XiaoyunqueError) as exc:
        client.submit_run(message="a cat", asset_ids=[], video_part_tool_param={"model": "m"})
    assert exc.value.status_code == 429
    assert len(fake.calls) == 3
    assert sleep.calls == [_EXPECTED_RETRY_DELAY, _EXPECTED_RETRY_DELAY]


@pytest.mark.asyncio
async def test_avideo_generation_16010_retries_then_succeeds_end_to_end():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _submit_ok()]})
    asleep = _RecordingAsleep()
    vo = await XiaoyunqueLLM(asleep=asleep, jitter=_no_jitter).avideo_generation(
        "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5}, None, client=fake
    )
    assert vo.status == "queued"
    assert len(fake.calls) == 2
    assert asleep.calls == [_EXPECTED_RETRY_DELAY]


@pytest.mark.asyncio
async def test_avideo_generation_16010_exhausted_raises_existing_rate_limit_mapping():
    fake = FakeAsyncClient(
        post_by_path={
            "/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _rate_limited_16010(), _rate_limited_16010()]
        }
    )
    asleep = _RecordingAsleep()
    with pytest.raises(RateLimitError):
        await XiaoyunqueLLM(asleep=asleep, jitter=_no_jitter).avideo_generation(
            "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5}, None, client=fake
        )
    assert len(fake.calls) == 3
    assert asleep.calls == [_EXPECTED_RETRY_DELAY, _EXPECTED_RETRY_DELAY]


def test_video_generation_sync_16010_retries_then_succeeds_end_to_end():
    fake = FakeSyncClient(post_by_path={"/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _submit_ok()]})
    sleep = _RecordingSleep()
    vo = XiaoyunqueLLM(sleep=sleep, jitter=_no_jitter).video_generation(
        "seedance2.0_vision", "a cat", "tok", None, {"seconds": 5}, None, client=fake
    )
    assert vo.status == "queued"
    assert len(fake.calls) == 2
    assert sleep.calls == [_EXPECTED_RETRY_DELAY]


@pytest.mark.asyncio
async def test_avideo_status_16010_does_not_retry():
    fake = FakeAsyncClient(
        post_by_path={
            "/api/biz/v1/agent/query_generate_video_result": {
                "ret": "16010",
                "errmsg": "操作过于频繁，请一分钟后再试",
                "data": {},
            }
        }
    )
    asleep = _RecordingAsleep()
    with pytest.raises(RateLimitError):
        await XiaoyunqueLLM(asleep=asleep, jitter=_no_jitter).avideo_status(_vid(), "tok", None, {}, None, client=fake)
    assert len(fake.calls) == 1
    assert asleep.calls == []


def test_retry_delay_is_at_least_the_upstream_requested_minute():
    assert _submit_rate_limit_delay(_no_jitter) >= 60.0


def test_retry_delay_uses_the_injected_jitter():
    def upper_bound(low: float, high: float) -> float:
        assert (low, high) == (0.0, 10.0)
        return high

    assert _submit_rate_limit_delay(upper_bound) == _EXPECTED_RETRY_DELAY + 10.0


def test_client_defaults_to_random_uniform_jitter():
    assert XiaoyunqueClient(token="t")._jitter is random.uniform


@pytest.mark.asyncio
async def test_retry_resubmits_a_byte_identical_body():
    fake = FakeAsyncClient(post_by_path={"/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _submit_ok()]})
    client = XiaoyunqueClient(token="t", async_client=fake, asleep=_RecordingAsleep(), jitter=_no_jitter)
    await client.asubmit_run(
        message="a cat on a windowsill",
        asset_ids=["asset-1", "asset-2"],
        video_part_tool_param={"model": "Seedance_2.5", "duration_sec": 30, "prompt": "a cat on a windowsill"},
        thread_id="thread-existing",
    )

    assert len(fake.calls) == 2, "expected exactly one retry"
    first_body, second_body = fake.calls[0][1], fake.calls[1][1]
    assert second_body == first_body, (
        "the retried submit body differs from the original — a retry that "
        "changes the model, prompt or reference set would generate a "
        f"different video at full cost.\nfirst:  {first_body!r}\nsecond: {second_body!r}"
    )
    assert first_body["asset_ids"] == ["asset-1", "asset-2"]
    assert first_body["thread_id"] == "thread-existing"
    assert first_body["video_part_tool_param"]["model"] == "Seedance_2.5"


def test_sync_retry_resubmits_a_byte_identical_body():
    fake = FakeSyncClient(post_by_path={"/api/biz/v1/skill/submit_run": [_rate_limited_16010(), _submit_ok()]})
    client = XiaoyunqueClient(token="t", sync_client=fake, sleep=_RecordingSleep(), jitter=_no_jitter)
    client.submit_run(
        message="a cat on a windowsill",
        asset_ids=["asset-1", "asset-2"],
        video_part_tool_param={"model": "Seedance_2.5", "duration_sec": 30, "prompt": "a cat on a windowsill"},
        thread_id="thread-existing",
    )

    assert len(fake.calls) == 2, "expected exactly one retry"
    first_body, second_body = fake.calls[0][1], fake.calls[1][1]
    assert second_body == first_body
    assert first_body["asset_ids"] == ["asset-1", "asset-2"]
    assert first_body["thread_id"] == "thread-existing"
    assert first_body["video_part_tool_param"]["model"] == "Seedance_2.5"
