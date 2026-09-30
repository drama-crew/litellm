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
                    "metering": {
                        "intent_id": "intent",
                        "request_digest": "a" * 64,
                        "actor_user_id": "actor",
                        "model": "hailuo-h3",
                    },
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
        assert first.status_code == (200 if failure == "receipt_ack" else 500)
        if failure == "receipt_ack":
            assert first.json() == {"accepted": True, "state": "submitted"}
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


@pytest.mark.asyncio
async def test_authenticated_resume_replaces_body_cache_before_real_processor(monkeypatch):
    import importlib

    from starlette.requests import Request

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

    auth_module = importlib.import_module("litellm.proxy.auth.user_api_key_auth")
    monkeypatch.setattr(
        auth_module,
        "_user_api_key_auth_builder",
        AsyncMock(return_value=UserAPIKeyAuth(api_key="key", end_user_id="owner")),
    )
    monkeypatch.setattr(auth_module, "_run_centralized_common_checks", AsyncMock())
    parent = Request(
        {"type": "http", "method": "POST", "path": "/internal/moderation/submit", "headers": [], "query_string": b""}
    )
    original = execution.execution_request(parent, {"model": "synthetic-model", "prompt": "approved"}, "/v1/videos")
    auth = await execution.authenticate(original, "sk-synthetic")
    cached = await _read_request_body(original)
    assert cached == {"model": "synthetic-model", "prompt": "approved"}
    seen = []

    async def process(self, **kwargs):
        seen.append(dict(self.data))
        return VideoObject(id="native", object="video", status="queued")

    monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "base_process_llm_request", process)
    await execution.invoke(original, auth, {"model": "synthetic-model", "prompt": "sealed-new"}, "avideo_generation")
    assert seen[0]["num_retries"] == 0
    assert seen[0]["prompt"] == "sealed-new"
    assert await _read_request_body(original) == cached


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,expected",
    [
        ("http-status", "rejected"),
        ("http-exception", "rejected"),
        ("proxy-exception", "rejected"),
        ("connect", "not_sent"),
        ("read", "ambiguous"),
        ("write", "ambiguous"),
        ("server", "ambiguous"),
        ("rewritten-read", "ambiguous"),
    ],
)
async def test_actual_endpoint_error_wrapping_preserves_submission_evidence(monkeypatch, kind, expected):
    from fastapi import HTTPException
    from starlette.requests import Request

    from litellm.proxy import proxy_server
    from litellm.proxy._types import ProxyException
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    request = httpx.Request("POST", "https://synthetic.invalid/videos")
    errors = {
        "http-status": httpx.HTTPStatusError(
            "rejected", request=request, response=httpx.Response(422, request=request)
        ),
        "http-exception": HTTPException(422, "rejected"),
        "proxy-exception": ProxyException(message="rejected", type="invalid_request_error", param=None, code=422),
        "connect": httpx.ConnectError("not sent", request=request),
        "rewritten-read": httpx.ReadTimeout("accepted then callback rewrites error", request=request),
        "read": httpx.ReadTimeout("accepted response lost", request=request),
        "write": httpx.WriteTimeout("partial transmission", request=request),
        "server": httpx.HTTPStatusError(
            "server unknown", request=request, response=httpx.Response(503, request=request)
        ),
    }

    async def process(self, **kwargs):
        raise errors[kind]

    monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "base_process_llm_request", process)
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "post_call_failure_hook",
        AsyncMock(return_value=HTTPException(422, "redacted") if kind == "rewritten-read" else None),
    )
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "post_call_response_headers_hook", AsyncMock(return_value=None))
    parent = Request({"type": "http", "method": "POST", "path": "/v1/videos", "headers": [], "query_string": b""})
    with pytest.raises(Exception) as wrapped:
        await execution.invoke(
            parent,
            UserAPIKeyAuth(api_key="synthetic"),
            {"model": "synthetic", "prompt": "synthetic"},
            "avideo_generation",
        )
    assert isinstance(wrapped.value, (HTTPException, ProxyException))
    assert execution.failure_outcome(wrapped.value) == expected


@pytest.mark.asyncio
async def test_settlement_without_admission_is_typed_404(monkeypatch):
    import time
    import jwt
    import httpx
    from types import SimpleNamespace
    from fastapi import FastAPI
    from litellm.proxy.video_endpoints import moderation_execution as execution, moderation_metering_runtime as runtime
    from litellm.proxy.video_endpoints.moderation_metering import MeteringStore

    class EmptyDb:
        async def query_raw(self, *args):
            return []

    meter = MeteringStore.__new__(MeteringStore)
    meter.db = EmptyDb()
    monkeypatch.setattr(runtime, "store", lambda: meter)
    secret = "synthetic-settlement-secret-long-enough-32-bytes"
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", secret)

    async def platform(request, method, path, payload):
        return dict(binding=None, native_id="native", events={})

    monkeypatch.setattr(execution.bridge, "platform", platform)
    app = FastAPI()
    app.include_router(execution.router)
    token = jwt.encode(
        dict(
            intent_id="never-admitted",
            request_digest="digest",
            input_digest="input",
            model="video",
            policy_version="v1",
            policy_digest="policy",
            purpose="settlement",
            iat=int(time.time()),
            exp=int(time.time()) + 60,
            aud="moderation-fork",
        ),
        secret,
        algorithm="HS256",
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://synthetic") as client:
        response = await client.post(
            "/internal/moderation/settlement", headers={"Authorization": "Bearer " + secret}, json={"ticket": token}
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "admission_missing"


def _bad_gateway():
    from litellm.exceptions import BadGatewayError

    response = httpx.Response(502, request=httpx.Request("POST", "https://provider.invalid/video"))
    return BadGatewayError(message="provider request failed", model="m", llm_provider="libtv", response=response)


def _timeout():
    from litellm.exceptions import Timeout

    return Timeout(message="provider request failed", model="m", llm_provider="libtv", exception_status_code=504)


def _http_status(code):
    return httpx.HTTPStatusError("boom", request=httpx.Request("POST", "https://p"), response=httpx.Response(code))


def _openai_status(code):
    import openai

    response = httpx.Response(code, request=httpx.Request("POST", "https://p"))
    return openai.APIStatusError("boom", response=response, body=None)


@pytest.mark.parametrize("code", [405, 406, 410, 411, 413, 414, 415, 431, 451])
def test_provider_status_proof_accepts_only_refused_before_processing_statuses(code):
    assert execution.provider_status_proof(_http_status(code)) == code
    assert execution.provider_status_proof(_openai_status(code)) == code


@pytest.mark.parametrize("code", [408, 409, 425, 429, 500, 502, 503, 504])
def test_provider_status_proof_rejects_statuses_that_may_follow_acceptance(code):
    assert execution.provider_status_proof(_http_status(code)) is None
    assert execution.provider_status_proof(_openai_status(code)) is None


@pytest.mark.parametrize(
    "error",
    [_bad_gateway(), _timeout(), httpx.ReadTimeout("lost"), httpx.ConnectError("refused"), RuntimeError("bug")],
)
def test_provider_status_proof_never_trusts_synthesized_or_transport_errors(error):
    assert execution.provider_status_proof(error) is None


def test_litellm_synthesized_status_is_not_proof_even_when_refusal_like():
    from litellm.exceptions import APIError
    from litellm.llms.custom_llm import CustomLLMError

    assert execution.provider_status_proof(APIError(415, "m", "libtv", "m")) is None
    assert execution.provider_status_proof(CustomLLMError(status_code=413, message="x")) is None


def test_provider_status_proof_follows_the_cause_chain():
    try:
        try:
            raise _http_status(413)
        except httpx.HTTPStatusError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert execution.provider_status_proof(outer) == 413


class _RecordingMeter:
    def __init__(self):
        self.marks = []
        self.manual = None
        self.not_submitted = None
        self.events = {}

    async def record_submission_failure(self, intent_id, code, status, message, attempt):
        self.marks.append((intent_id, code, status, message, attempt))
        return True

    async def clear_submission_failure(self, intent_id):
        self.cleared = getattr(self, "cleared", 0) + 1

    async def binding(self, intent_id):
        from litellm.proxy.video_endpoints.moderation_metering import BillingBinding

        return BillingBinding(
            intent_id=intent_id,
            request_digest="a" * 64,
            fingerprint="key",
            user_id="user",
            team_id="team",
            model="video",
            expected_phases=("submit",),
        )

    async def manual_settlement_reason(self, intent_id):
        return self.manual

    async def submission_failure(self, intent_id):
        return self.not_submitted

    async def phase(self, intent_id, phase):
        return self.events.get(phase)

    async def persist(self, event):
        raise AssertionError("no events expected")

    async def settlement(self, binding):
        raise AssertionError("must not settle")


async def _post_submit(monkeypatch, error):
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    meter = _RecordingMeter()
    monkeypatch.setattr(runtime, "store", lambda: meter)
    app = FastAPI()
    app.include_router(execution.router)
    calls = []

    async def platform(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/begin"):
            return httpx.Response(
                200,
                json={
                    "acquired": True,
                    "token": "attempt",
                    "metering": {
                        "intent_id": "intent",
                        "request_digest": "a" * 64,
                        "actor_user_id": "actor",
                        "model": "hailuo-h3",
                    },
                    "credential": "sk-original",
                    "route": "avideo_generation",
                    "request": {"model": "hailuo-h3", "prompt": "synthetic"},
                },
            )
        return httpx.Response(200, json={})

    app.state.moderation_transport = httpx.MockTransport(platform)
    monkeypatch.setattr(execution, "authenticate", AsyncMock(return_value=UserAPIKeyAuth(api_key="original")))
    monkeypatch.setattr(execution, "invoke", AsyncMock(side_effect=error))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fork"
    ) as client:
        await client.post(
            "/internal/moderation/submit", json={"ticket": ticket()}, headers={"Authorization": "Bearer " + SECRET}
        )
    return meter, calls


@pytest.mark.asyncio
async def test_refused_before_processing_records_provider_not_submitted_bound_to_the_attempt(monkeypatch):
    meter, calls = await _post_submit(monkeypatch, _http_status(413))
    assert meter.marks == [("intent", "provider_not_submitted", 413, "provider request failed", "attempt")]
    assert meter.cleared == 1
    assert not any(path.endswith("/not-sent") for path in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"),
    [
        (_bad_gateway(), 502),
        (_http_status(503), 503),
        (httpx.ReadTimeout("lost reply"), None),
        (RuntimeError("unknown"), None),
    ],
)
async def test_other_post_admission_failures_record_provider_submission_ambiguous(monkeypatch, error, status):
    meter, _ = await _post_submit(monkeypatch, error)
    assert meter.marks == [("intent", "provider_submission_ambiguous", status, "provider request failed", "attempt")]


async def _settle(monkeypatch, meter):
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    monkeypatch.setattr(runtime, "store", lambda: meter)

    async def platform(request, method, path, payload):
        return dict(binding=None, native_id="native", events={}, recovery_binding={"intent_id": "intent"})

    monkeypatch.setattr(execution.bridge, "platform", platform)
    app = FastAPI()
    app.include_router(execution.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fork") as client:
        return await client.post(
            "/internal/moderation/settlement",
            headers={"Authorization": "Bearer " + SECRET},
            json={"ticket": ticket("settlement")},
        )


@pytest.mark.asyncio
async def test_settlement_surfaces_key_identity_mismatch_as_typed_409(monkeypatch):
    meter = _RecordingMeter()
    meter.manual = "key_identity_mismatch: moderation metering key identity mismatch"
    response = await _settle(monkeypatch, meter)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "key_identity_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "status"), [("provider_not_submitted", 413), ("provider_submission_ambiguous", None)])
async def test_settlement_surfaces_submission_failures_as_typed_409_with_the_attempt(monkeypatch, code, status):
    meter = _RecordingMeter()
    meter.not_submitted = (code, status, "provider request failed", "attempt-7")
    response = await _settle(monkeypatch, meter)
    assert response.status_code == 409
    error = response.json()["error"]
    assert (error["code"], error["provider_status"], error["attempt"]) == (code, status, "attempt-7")


def test_provider_status_proof_ignores_the_implicit_context_chain():
    try:
        try:
            raise _http_status(413)
        except httpx.HTTPStatusError:
            raise RuntimeError("unrelated failure raised while handling another error")
    except RuntimeError as outer:
        assert outer.__context__ is not None and outer.__cause__ is None
        assert execution.provider_status_proof(outer) is None


def _boundary_failure(exc):
    from litellm.llms.libtv.handler import normalize_libtv_errors

    @normalize_libtv_errors
    async def call(*args, **kwargs):
        raise exc

    return call


def _no_login():
    from litellm.llms.libtv.common import LibTVError

    error = LibTVError(502, "libtv getUserInfo code=401 msg=No login")
    error.submission = "not_sent"
    return error


def _timeout():
    return httpx.ReadTimeout("slow")


def _metered_request(method="POST"):
    from fastapi import Request

    return Request({"type": "http", "method": method, "path": "/v1/videos", "headers": [], "app": FastAPI()})


async def _run_execute(monkeypatch, sequence, route="avideo_generation", scoped=True, method="POST"):
    """Drive ``execute`` with a router that attempts one deployment per factory in ``sequence``."""
    from unittest.mock import MagicMock

    from litellm import Router
    from litellm.proxy.spend_tracking import budget_reservation
    from litellm.proxy.video_endpoints import moderation_metering_entry as entry

    scope = MagicMock()
    scope.store.void_unsent = AsyncMock()
    monkeypatch.setattr(entry, "scope_for", AsyncMock(return_value=scope if scoped else None))
    monkeypatch.setattr(budget_reservation, "release_budget_reservation", AsyncMock())
    router = Router(model_list=[{"model_name": "m", "litellm_params": {"model": "openai/x", "api_key": "k"}}])
    final = []

    async def attempts():
        error = None
        for factory in sequence:
            try:
                await router.make_call(_boundary_failure(factory()), model="m")
            except Exception as caught:  # noqa: BLE001  # the router moves to the next attempt
                error = caught
        final.append(error)
        raise error

    auth = UserAPIKeyAuth(api_key="k", user_id="u", team_id="t")
    with pytest.raises(Exception) as caught:
        await entry.execute(_metered_request(method), auth, route, attempts())
    return caught.value, final[0], scope


@pytest.mark.asyncio
async def test_execute_503_not_sent_when_every_attempt_is_not_sent(monkeypatch):
    from fastapi import HTTPException

    error, _, scope = await _run_execute(monkeypatch, [_no_login, _no_login])
    assert isinstance(error, HTTPException) and error.status_code == 503
    assert error.headers == {"x-drama-submission": "not_sent"}
    scope.store.void_unsent.assert_awaited_once()
    # the public submit path classifies the same exception after execute has returned
    assert execution.failure_outcome(error) == "not_sent"


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [[_no_login, _timeout], [_timeout, _no_login]])
async def test_execute_and_public_path_agree_ambiguous_when_any_attempt_is_unknown(monkeypatch, sequence):
    error, final, scope = await _run_execute(monkeypatch, sequence)
    assert error is final  # no rewrite
    assert execution.failure_outcome(error) == "ambiguous"
    scope.store.void_unsent.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_deployment_retry_not_sent_then_timeout_is_ambiguous(monkeypatch):
    error, final, _ = await _run_execute(monkeypatch, [_no_login, _timeout])
    assert error is final and execution.failure_outcome(error) == "ambiguous"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, scoped, method",
    [("aimage_generation", True, "POST"), ("avideo_generation", False, "POST"), ("avideo_status", True, "GET")],
)
async def test_execute_leaves_non_metered_image_and_status_errors_unchanged(monkeypatch, route, scoped, method):
    error, final, _ = await _run_execute(monkeypatch, [_no_login, _no_login], route, scoped, method)
    assert error is final


@pytest.mark.asyncio
@pytest.mark.parametrize("entries", [["not_sent", "sent"], ["sent"]])
async def test_execute_never_answers_not_sent_once_a_provider_attempt_succeeded(monkeypatch, entries):
    """Hard short-circuit: even a ledger that (wrongly) still reads all-not-sent cannot win."""
    from litellm.router_utils import attempt_outcomes
    from litellm.proxy.video_endpoints import moderation_metering_entry as entry
    from litellm.proxy.spend_tracking import budget_reservation
    from unittest.mock import MagicMock

    scope = MagicMock()
    scope.store.void_unsent = AsyncMock()
    monkeypatch.setattr(entry, "scope_for", AsyncMock(return_value=scope))
    monkeypatch.setattr(budget_reservation, "release_budget_reservation", AsyncMock())
    failure = RuntimeError("after provider success")

    async def call():
        for value in entries:
            attempt_outcomes.note(value)
        raise failure

    # simulate a ledger predicate that would say not_sent: the short-circuit must still hold
    monkeypatch.setattr(attempt_outcomes, "all_not_sent", lambda: True)
    auth = UserAPIKeyAuth(api_key="k", user_id="u", team_id="t")
    with pytest.raises(RuntimeError) as caught:
        await entry.execute(_metered_request(), auth, "avideo_generation", call())
    assert caught.value is failure
    assert execution.failure_outcome(failure) == "ambiguous"
    scope.store.void_unsent.assert_not_awaited()
