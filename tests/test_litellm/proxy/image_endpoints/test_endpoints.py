import asyncio
import copy
from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any, Dict

import orjson
import pytest
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.image_endpoints import endpoints
from litellm.llms.libtv.handler import LibTVLLM
from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.receipts import ReceiptClaim, StoredReceipt, request_fingerprint
from litellm.llms.libtv.image_upscale import (
    ImageUpscaleReceipt,
    make_resume_token,
    normalize_image_upscale_receipt,
    verify_resume_token,
)
from litellm.types.utils import ImageResponse


def _request(body: bytes) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/libtv/image-upscale/submit",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _action_request(path: str, body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": path, "headers": []}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


class _EndpointReceiptStore:
    def __init__(self):
        self.receipt = None
        self.receipt_key = "endpoint-receipt"

    async def claim(self, team_id, model, request_id, fingerprint, deployment_id, **kwargs):
        if self.receipt is not None:
            if self.receipt.fingerprint != fingerprint:
                return ReceiptClaim("mismatch", self.receipt_key, None)
            return ReceiptClaim("existing", self.receipt_key, self.receipt)
        self.receipt = StoredReceipt(
            team_id=team_id,
            model=model,
            request_id=request_id,
            fingerprint=fingerprint,
            submission_state="submitting",
            deployment_id=deployment_id,
        )
        return ReceiptClaim("owner", self.receipt_key, self.receipt)

    async def transition(self, receipt, receipt_key, submission_state, **kwargs):
        self.receipt = StoredReceipt(
            team_id=receipt.team_id,
            model=receipt.model,
            request_id=receipt.request_id,
            fingerprint=receipt.fingerprint,
            submission_state=submission_state,
            deployment_id=receipt.deployment_id,
            provider_task_id=kwargs.get("provider_task_id"),
            resume_token=kwargs.get("resume_token"),
            provider_code=kwargs.get("provider_code"),
            message=kwargs.get("message"),
        )
        return self.receipt

    async def wait(self, claim):
        return self.receipt

    async def readiness(self):
        return True


class _EndpointRouter:
    model_names = ["topaz-image-upscaler"]
    model_group_alias = None
    default_deployment = None
    deployment_names = []
    router_general_settings = SimpleNamespace(pass_through_all_models=False)
    pattern_router = SimpleNamespace(patterns=[])

    def has_model_id(self, model):
        return False

    def map_team_model(self, model, team_id):
        return None

    async def aimage_generation(self, **data):
        data.setdefault("output_cost_per_image_2x", 0.25)
        return await LibTVLLM(poll_interval=0).aimage_generation(
            model=data["model"],
            prompt=data.get("prompt", ""),
            model_response=ImageResponse(),
            api_key=data.get("api_key"),
            api_base=None,
            optional_params=data,
            logging_obj=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity",
    [
        {"api_key": "key-1", "user_id": "user-1", "org_id": "org-1"},
        {"team_id": "team-1", "user_id": "user-1", "org_id": "org-1"},
        {"team_id": "team-1", "api_key": "key-1", "org_id": "org-1"},
        {"team_id": "team-1", "api_key": "key-1", "user_id": "user-1"},
    ],
)
async def test_image_upscale_submit_fails_closed_before_all_side_effects_for_incomplete_billing_identity(
    monkeypatch, identity
):
    calls = []

    async def unexpected_add_litellm_data_to_request(**kwargs):
        calls.append("add_litellm_data")
        raise AssertionError("incomplete billing identity must not enter request processing")

    async def unexpected_route_request(**kwargs):
        calls.append("route")
        raise AssertionError("incomplete billing identity must not reach provider routing")

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request", unexpected_add_litellm_data_to_request
    )
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", unexpected_route_request)

    result = await endpoints.libtv_image_upscale_submit(
        request=_request(
            orjson.dumps(
                {
                    "request_id": "identity-gate-request",
                    "source_url": "https://source.example/input.png",
                    "source_bytes": 3,
                    "source_sha256": "a" * 64,
                }
            )
        ),
        user_api_key_dict=UserAPIKeyAuth(**identity),
    )

    body = json.loads(result.body)
    assert result.status_code == 503
    assert body["error"]["code"] == "libtv_submission_not_submitted"
    assert body["error"]["metadata"]["submission_receipt"]["submission_state"] == "not_submitted"
    assert calls == []


@pytest.mark.asyncio
async def test_image_upscale_endpoint_replay_returns_durable_receipt_without_second_provider_create(
    monkeypatch,
):
    store = _EndpointReceiptStore()
    provider_calls = 0

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):
        return data

    async def fake_post_call_success_hook(**kwargs):
        return kwargs["response"]

    async def fake_route_request(*, data, **kwargs):
        return _EndpointRouter().aimage_generation(**data)

    async def fake_resolve_model_spec(self, model):
        return {"model_key": model, "vendor": "topazlabs", "task_type": "image"}

    async def fake_ensure_libtv_url(self, kind, url, data, *args, **kwargs):
        return url

    async def fake_acreate(self, model_key, vendor, task_type, params, project_name, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return {"task_id": "task-1"}

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        post_call_success_hook=fake_post_call_success_hook,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)
    monkeypatch.setattr("litellm.llms.libtv.persistence.get_receipt_store", lambda: store)
    monkeypatch.setattr("litellm.llms.libtv.client.get_receipt_store", lambda: store)
    monkeypatch.setenv("LIBTV_RECEIPTS_REDIS_URL", "redis://receipt-test")
    monkeypatch.setenv("LIBTV_TOKEN", "test-token")
    monkeypatch.setenv("LIBTV_WEBID", "test-webid")
    monkeypatch.setattr("litellm.llms.libtv.client.LibTVClient.aresolve_model_spec", fake_resolve_model_spec)
    monkeypatch.setattr("litellm.llms.libtv.client.LibTVClient.aensure_libtv_url", fake_ensure_libtv_url)
    monkeypatch.setattr("litellm.llms.libtv.client.LibTVClient.acreate", fake_acreate)

    request_body = orjson.dumps(
        {
            "request_id": "endpoint-request-1",
            "source_url": "https://source.example/input.png",
            "source_bytes": 3,
            "source_sha256": "a" * 64,
            "model_info": {"id": "primary"},
        }
    )
    user_api_key = UserAPIKeyAuth(team_id="team-1", api_key="key-1", user_id="user-1", org_id="org-1")
    first = await endpoints.libtv_image_upscale_submit(request=_request(request_body), user_api_key_dict=user_api_key)
    second = await endpoints.libtv_image_upscale_submit(request=_request(request_body), user_api_key_dict=user_api_key)

    first_body = json.loads(first.body)
    second_body = json.loads(second.body)
    assert first.status_code == 202
    assert second.status_code == 202
    assert provider_calls == 1
    assert first_body["receipt"] == second_body["receipt"]
    assert second_body["receipt"]["provider_task_id"] == "task-1"
    assert verify_resume_token(
        second_body["receipt"]["resume_token"],
        "test-token",
        deployment_id="primary",
        provider_task_id="task-1",
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="endpoint-request-1",
        fingerprint=request_fingerprint(
            {"source_sha256": "a" * 64, "style": "Standard V2", "scale": 2},
            "topaz-image-upscaler",
        ),
    )


@pytest.mark.asyncio
async def test_image_upscale_endpoint_fingerprint_mismatch_returns_stable_conflict(monkeypatch):
    store = _EndpointReceiptStore()

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):
        return data

    async def fake_post_call_success_hook(**kwargs):
        return kwargs["response"]

    async def fake_route_request(*, data, **kwargs):
        return _EndpointRouter().aimage_generation(**data)

    async def fake_resolve_model_spec(self, model):
        return {"model_key": model, "vendor": "topazlabs", "task_type": "image"}

    async def fake_ensure_libtv_url(self, kind, url, data, *args, **kwargs):
        return url

    async def fake_acreate(self, model_key, vendor, task_type, params, project_name, **kwargs):
        return {"task_id": "task-1"}

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        post_call_success_hook=fake_post_call_success_hook,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)
    monkeypatch.setattr("litellm.llms.libtv.client.get_receipt_store", lambda: store)
    monkeypatch.setenv("LIBTV_RECEIPTS_REDIS_URL", "redis://receipt-test")
    monkeypatch.setenv("LIBTV_TOKEN", "test-token")
    monkeypatch.setenv("LIBTV_WEBID", "test-webid")
    monkeypatch.setattr("litellm.llms.libtv.client.LibTVClient.aresolve_model_spec", fake_resolve_model_spec)
    monkeypatch.setattr("litellm.llms.libtv.client.LibTVClient.aensure_libtv_url", fake_ensure_libtv_url)
    monkeypatch.setattr("litellm.llms.libtv.client.LibTVClient.acreate", fake_acreate)

    def body(style):
        return orjson.dumps(
            {
                "request_id": "endpoint-mismatch-1",
                "source_url": "https://source.example/input.png",
                "source_bytes": 3,
                "source_sha256": "a" * 64,
                "style": style,
                "model_info": {"id": "primary"},
            }
        )

    user_api_key = UserAPIKeyAuth(team_id="team-1", api_key="key-1", user_id="user-1", org_id="org-1")
    first = await endpoints.libtv_image_upscale_submit(
        request=_request(body("Standard V2")), user_api_key_dict=user_api_key
    )
    second = await endpoints.libtv_image_upscale_submit(request=_request(body("CGI")), user_api_key_dict=user_api_key)

    assert first.status_code == 202
    assert second.status_code == 409
    second_body = json.loads(second.body)
    assert second_body["error"]["code"] == "idempotency_fingerprint_mismatch"
    assert second_body["error"]["metadata"]["submission_receipt"]["request_id"] == "endpoint-mismatch-1"


@pytest.mark.asyncio
async def test_image_generation_prompt_rerouting(monkeypatch):
    """Ensure image prompts are exposed to guardrails and restored afterwards."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: Dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        modified = {
            **data,
            "messages": [
                {
                    "role": "user",
                    "content": "sanitized prompt",
                }
            ],
        }
        return modified

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    async def fake_post_call_success_hook(*, data, user_api_key_dict, response):
        return response

    async def fake_post_call_response_headers_hook(**kwargs):
        return {"x-callback-test": "value"}

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
        post_call_success_hook=fake_post_call_success_hook,
        post_call_response_headers_hook=fake_post_call_response_headers_hook,
    )

    captured_route_request_data: Dict[str, Any] = {}

    async def fake_route_request(*, data, **kwargs):  # type: ignore[override]
        captured_route_request_data.update(data)

        async def _inner():
            class FakeResponse(dict):
                _hidden_params = {}

            return FakeResponse(result="ok")

        return _inner()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/generations",
        "headers": [],
    }
    body = orjson.dumps({"prompt": "original prompt"})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)

    result = await endpoints.image_generation(
        request=request,
        fastapi_response=response,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert pre_call_input["messages"][0]["content"] == "original prompt"
    assert captured_route_request_data["prompt"] == "sanitized prompt"
    assert "messages" not in captured_route_request_data
    assert response.headers.get("x-callback-test") == "value"


@pytest.mark.asyncio
async def test_image_upscale_submit_maps_source_url_and_runs_proxy_pipeline(monkeypatch):
    captured: Dict[str, Any] = {}
    hooks: list[str] = []

    async def fake_add_litellm_data_to_request(**kwargs):
        captured["added"] = True
        return {**kwargs["data"], "injected_budget": "budget-1"}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):
        hooks.append(call_type)
        return {
            **data,
            "messages": [{"role": "user", "content": "sanitized"}],
        }

    async def fake_post_call_success_hook(*, data, user_api_key_dict, response):
        hooks.append("success")
        return response

    async def fake_post_call_failure_hook(**kwargs):
        hooks.append("failure")

    async def fake_route_request(*, data, **kwargs):
        captured["route"] = data

        async def _inner():
            class FakeResponse:
                _hidden_params = {
                    "submission_receipt": {
                        "request_id": "r1",
                        "submission_state": "submitted",
                        "deployment_id": "dep-1",
                        "provider_task_id": "task-1",
                        "resume_token": "opaque",
                    }
                }

            return FakeResponse()

        return _inner()

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        post_call_success_hook=fake_post_call_success_hook,
        post_call_failure_hook=fake_post_call_failure_hook,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)

    result = await endpoints.libtv_image_upscale_submit(
        request=_request(
            orjson.dumps(
                {
                    "request_id": "r1",
                    "source_url": "https://source.example/input.png",
                    "source_bytes": 3,
                    "source_sha256": "a" * 64,
                }
            )
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1", api_key="key-1", user_id="user-1", org_id="org-1"),
    )

    assert result.status_code == 202
    assert captured["added"] is True
    assert captured["route"]["prompt"] == "sanitized"
    assert captured["route"]["input_reference"] == "https://source.example/input.png"
    assert "source_url" not in captured["route"]
    assert captured["route"]["injected_budget"] == "budget-1"
    assert hooks == ["image_generation", "success"]


@pytest.mark.asyncio
async def test_image_upscale_submit_hook_error_preserves_unknown_submission_state(monkeypatch):
    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_pre_call_hook(**kwargs):
        return kwargs["data"]

    async def fake_route_request(*, data, **kwargs):
        async def _inner():
            class FakeResponse:
                _hidden_params = {
                    "submission_receipt": {
                        "request_id": "r1",
                        "submission_state": "submitted",
                        "deployment_id": "dep-1",
                        "provider_task_id": "task-1",
                        "resume_token": "opaque",
                    }
                }

            return FakeResponse()

        return _inner()

    async def fake_post_call_success_hook(**kwargs):
        raise ValueError("success hook failed after provider submission")

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        post_call_success_hook=fake_post_call_success_hook,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)

    result = await endpoints.libtv_image_upscale_submit(
        request=_request(
            orjson.dumps(
                {
                    "request_id": "r1",
                    "source_url": "https://source.example/input.png",
                    "source_bytes": 3,
                    "source_sha256": "a" * 64,
                }
            )
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1", api_key="key-1", user_id="user-1", org_id="org-1"),
    )

    body = json.loads(result.body)
    assert result.status_code == 409
    assert body["error"]["metadata"]["submission_receipt"]["submission_state"] == "unknown"


@pytest.mark.asyncio
async def test_image_upscale_submit_malformed_json_returns_not_submitted_receipt(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)

    result = await endpoints.libtv_image_upscale_submit(
        request=_request(b"{"),
        user_api_key_dict=UserAPIKeyAuth(),
    )

    body = json.loads(result.body)
    assert result.status_code == 503
    assert body["error"]["metadata"]["submission_receipt"]["submission_state"] == "not_submitted"


@pytest.mark.asyncio
async def test_image_upscale_submit_rejects_conflicting_source_aliases(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)

    result = await endpoints.libtv_image_upscale_submit(
        request=_request(
            orjson.dumps(
                {
                    "request_id": "r1",
                    "source_url": "https://source.example/one.png",
                    "input_reference": "https://source.example/two.png",
                    "source_bytes": 3,
                    "source_sha256": "a" * 64,
                }
            )
        ),
        user_api_key_dict=UserAPIKeyAuth(),
    )

    body = json.loads(result.body)
    assert result.status_code == 503
    assert body["error"]["metadata"]["submission_receipt"]["submission_state"] == "not_submitted"


def test_image_upscale_openapi_contract_requires_source_digest_and_exposes_receipt_schema():
    route = next(
        route for route in endpoints.router.routes if getattr(route, "path", None) == "/v1/libtv/image-upscale/submit"
    )
    request_schema = route.openapi_extra["requestBody"]["content"]["application/json"]["schema"]
    assert {"source_bytes", "source_sha256"}.issubset(set(request_schema["required"]))

    error_schema = route.responses[409]["model"].model_json_schema()
    receipt_schema = error_schema["$defs"]["ImageUpscaleReceiptResponse"]
    assert "submission_state" in receipt_schema["properties"]
    assert "task_state" in receipt_schema["properties"]


@pytest.mark.asyncio
async def test_image_upscale_submit_rejects_inline_media_body():
    result = await endpoints.libtv_image_upscale_submit(
        request=_request(
            orjson.dumps(
                {
                    "request_id": "r1",
                    "source_url": "https://source.example/input.png",
                    "source_bytes": 3,
                    "source_sha256": "a" * 64,
                    "source_body": "base64-media-must-not-reach-libtv",
                }
            )
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert result.status_code == 503
    assert "source_body" not in result.body.decode()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "data:image/png;base64,not-allowed"),
        ("data", {"image": "not-allowed"}),
        ("source_base64", "not-allowed"),
        ("unexpected", "not-allowed"),
    ],
)
def test_image_upscale_submit_request_forbids_unknown_top_level_fields(field, value):
    payload = {
        "request_id": "r1",
        "source_url": "https://source.example/input.png",
        "source_bytes": 3,
        "source_sha256": "a" * 64,
        field: value,
    }

    with pytest.raises(ValidationError) as error:
        endpoints.ImageUpscaleSubmitRequest.model_validate(payload)

    assert any(item["type"] == "extra_forbidden" and item["loc"] == (field,) for item in error.value.errors())


def test_image_upscale_submit_request_accepts_declared_receipt_and_source_metadata():
    request = endpoints.ImageUpscaleSubmitRequest.model_validate(
        {
            "model": "topaz-image-upscaler",
            "request_id": "receipt-r1",
            "input_reference": "https://source.example/input.png",
            "source_bytes": 3,
            "source_sha256": "a" * 64,
            "source_hard_cap": 64,
            "style": "Standard V2",
            "scale": 2,
            "model_info": {"id": "primary"},
        }
    )

    assert request.request_id == "receipt-r1"
    assert request.input_reference == "https://source.example/input.png"
    assert request.source_sha256 == "a" * 64
    assert request.model_info == {"id": "primary"}


@pytest.mark.asyncio
async def test_image_upscale_submit_returns_422_for_unknown_top_level_field():
    result = await endpoints.libtv_image_upscale_submit(
        request=_request(
            orjson.dumps(
                {
                    "request_id": "r1",
                    "source_url": "https://source.example/input.png",
                    "source_bytes": 3,
                    "source_sha256": "a" * 64,
                    "image": "data:image/png;base64,not-allowed",
                }
            )
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert result.status_code == 422
    assert any(
        item["type"] == "extra_forbidden" and item["loc"] == ["image"] for item in json.loads(result.body)["detail"]
    )


@pytest.mark.asyncio
async def test_image_upscale_submit_requires_stable_request_id_with_422():
    result = await endpoints.libtv_image_upscale_submit(
        request=_request(
            orjson.dumps(
                {
                    "source_url": "https://source.example/input.png",
                    "source_bytes": 3,
                    "source_sha256": "a" * 64,
                }
            )
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert result.status_code == 422
    assert "request_id" in json.dumps(json.loads(result.body)["detail"])


def test_image_upscale_openapi_requires_request_id():
    route = next(
        route for route in endpoints.router.routes if getattr(route, "path", None) == "/v1/libtv/image-upscale/submit"
    )
    schema = route.openapi_extra["requestBody"]["content"]["application/json"]["schema"]
    assert "request_id" in schema["required"]


def test_image_upscale_submit_collects_all_libtv_deployments_for_pool_failover():
    router = SimpleNamespace(
        get_model_list=lambda **kwargs: [
            {
                "model_info": {"id": "primary"},
                "litellm_params": {"custom_llm_provider": "libtv", "model": "libtv/topaz-image-upscaler"},
            },
            {
                "model_info": {"id": "secondary"},
                "litellm_params": {"custom_llm_provider": "libtv", "model": "libtv/topaz-image-upscaler"},
            },
        ]
    )
    pool = endpoints._image_upscale_deployment_pool(router, "topaz-image-upscaler")
    assert [entry["id"] for entry in pool] == ["primary", "secondary"]


def test_image_upscale_recovery_resolves_exact_deployment_credential(monkeypatch):
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="submitted",
        deployment_id="secondary",
        provider_task_id="task-1",
        resume_token="signed",
        task_state="succeeded",
        terminal_result={"url": "https://provider.example/result.png", "provider_task_id": "task-1"},
    )
    router = SimpleNamespace(
        get_model_list=lambda **kwargs: [
            {
                "model_info": {"id": "primary"},
                "litellm_params": {
                    "custom_llm_provider": "libtv",
                    "model": "libtv/topaz-image-upscaler",
                    "api_key": "os.environ/LIBTV_TOKEN",
                    "webid": "os.environ/LIBTV_WEBID",
                },
            },
            {
                "model_info": {"id": "secondary"},
                "litellm_params": {
                    "custom_llm_provider": "libtv",
                    "model": "libtv/topaz-image-upscaler",
                    "api_key": "os.environ/LIBTV_TOKEN_2",
                    "webid": "os.environ/LIBTV_WEBID_2",
                },
            },
        ]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setenv("LIBTV_TOKEN", "primary-token")
    monkeypatch.setenv("LIBTV_WEBID", "primary-webid")
    monkeypatch.setenv("LIBTV_TOKEN_2", "secondary-token")
    monkeypatch.setenv("LIBTV_WEBID_2", "secondary-webid")

    client = endpoints._client_for_receipt(receipt)

    assert client.token == "secondary-token"
    assert client.webid == "secondary-webid"


@pytest.mark.asyncio
async def test_image_upscale_receipt_lookup_is_authenticated_and_team_scoped(monkeypatch):
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token="signed",
        task_state="succeeded",
        terminal_result={
            "url": "https://provider.example/result.png",
            "provider_task_id": "task-1",
        },
    )

    class Store:
        async def get(self, team_id, model, request_id):
            return receipt if (team_id, model, request_id) == ("team-1", "topaz-image-upscaler", "request-1") else None

    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: Store())

    response = await endpoints.libtv_image_upscale_receipt(
        request_id="request-1",
        model="topaz-image-upscaler",
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )
    assert response.status_code == 200
    assert json.loads(response.body)["receipt"]["provider_task_id"] == "task-1"
    assert json.loads(response.body)["receipt"]["result"] == {
        "url": "https://provider.example/result.png",
        "provider_task_id": "task-1",
    }


def test_not_submitted_receipt_serialization_round_trip_retains_deployment():
    receipt = ImageUpscaleReceipt(
        request_id="request-1",
        submission_state="not_submitted",
        deployment_id="primary",
        message="source transfer failed",
    )

    response = endpoints._image_upscale_response(receipt.to_dict())
    payload = json.loads(response.body)
    restored = normalize_image_upscale_receipt(payload, "request-1")

    assert response.status_code == 503
    assert restored.submission_state == "not_submitted"
    assert restored.deployment_id == "primary"


@pytest.mark.asyncio
async def test_image_upscale_resolution_rejects_non_operator_before_store_lookup(monkeypatch):
    calls = []
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: calls.append(True) or None)

    response = await endpoints.libtv_image_upscale_resolve(
        request=_action_request(
            "/v1/libtv/image-upscale/resolve",
            orjson.dumps(
                {
                    "request_id": "request-1",
                    "reason": "duplicate risk",
                    "confirm_submission_risk": True,
                }
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1", user_id="user-1", user_role="internal_user"),
    )

    assert response.status_code == 403
    assert calls == []


@pytest.mark.asyncio
async def test_image_upscale_resolution_records_authenticated_operator_only(monkeypatch):
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="unknown",
        deployment_id="primary",
    )

    class Store:
        async def get(self, team_id, model, request_id):
            return receipt

        async def transition(self, stored, receipt_key, state, **kwargs):
            return StoredReceipt(
                team_id=stored.team_id,
                model=stored.model,
                request_id=stored.request_id,
                fingerprint=stored.fingerprint,
                submission_state=state,
                deployment_id=stored.deployment_id,
                resolution_tombstone=kwargs["resolution_tombstone"],
                task_state=kwargs["task_state"],
            )

    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: Store())

    response = await endpoints.libtv_image_upscale_resolve(
        request=_action_request(
            "/v1/libtv/image-upscale/resolve",
            orjson.dumps(
                {
                    "request_id": "request-1",
                    "reason": "duplicate risk",
                    "confirm_submission_risk": True,
                }
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1", user_id="operator-1", user_role="proxy_admin"),
    )

    tombstone = json.loads(response.body)["receipt"]["resolution_tombstone"]
    assert response.status_code == 200
    assert tombstone["operator_id"] == "operator-1"
    assert tombstone["reason"] == "duplicate risk"
    assert tombstone["confirmed_submission_risk"] is True
    assert "resolved_at" in tombstone
    assert json.loads(response.body)["receipt"]["task_state"] == "resolved"


@pytest.mark.asyncio
async def test_image_upscale_resolution_rejects_client_supplied_audit_fields(monkeypatch):
    response = await endpoints.libtv_image_upscale_resolve(
        request=_action_request(
            "/v1/libtv/image-upscale/resolve",
            orjson.dumps(
                {
                    "request_id": "request-1",
                    "reason": "duplicate risk",
                    "confirm_submission_risk": True,
                    "operator_id": "spoofed",
                }
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1", user_id="operator-1", user_role="proxy_admin"),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_image_upscale_poll_terminal_transition_appends_billing_event(monkeypatch):
    fingerprint = "f" * 64
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint=fingerprint,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token=make_resume_token(
            "dep-1",
            "task-1",
            "secret",
            team_id="team-1",
            model="topaz-image-upscaler",
            request_id="request-1",
            fingerprint=fingerprint,
        ),
        response_cost=0.25,
        api_key="hashed-key-id",
        user_id="receipt-user",
        organization_id="receipt-org",
        scale=4,
        project_id="project-1",
        artifact_id="artifact-1",
        attribution_user_id="owner-1",
    )

    class Store:
        def __init__(self):
            self.transition_calls = []

        async def get(self, team_id, model, request_id):
            return receipt

        async def transition(self, *args, **kwargs):
            self.transition_calls.append(kwargs)
            return replace(
                receipt,
                task_state=kwargs["task_state"],
                billing_event_id=kwargs["billing_event"].event_id,
                terminal_result=kwargs["terminal_result"],
            )

    store = Store()

    class Client:
        async def apoll_image_upscale(self, provider_task_id):
            assert provider_task_id == "task-1"
            return {
                "status": 2,
                "urls": ["https://libtv.example/result.png"],
            }

    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: store)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints._client_for_receipt", lambda _: Client())
    monkeypatch.setenv("LIBTV_IMAGE_UPSCALE_RESUME_SECRET", "secret")

    response = await endpoints.libtv_image_upscale_poll(
        request=_action_request(
            "/v1/libtv/image-upscale/poll",
            orjson.dumps(
                {
                    "request_id": "request-1",
                    "model": "topaz-image-upscaler",
                    "resume_token": receipt.resume_token,
                }
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(
            team_id="team-1", api_key="sk-request-key", user_id="request-user", org_id="request-org"
        ),
    )

    assert response.status_code == 200
    assert json.loads(response.body)["result"] == {
        "url": "https://libtv.example/result.png",
        "provider_task_id": "task-1",
    }
    event = store.transition_calls[0]["billing_event"]
    assert event.provider_task_id == "task-1"
    assert event.response_cost == 0.25
    assert event.api_key == "hashed-key-id"
    assert event.user_id == "receipt-user"
    assert event.organization_id == "receipt-org"
    assert event.team_id == "team-1"
    assert event.model == "topaz-image-upscaler"
    assert (event.scale, event.project_id, event.artifact_id, event.attribution_user_id) == (
        4,
        "project-1",
        "artifact-1",
        "owner-1",
    )
    assert store.transition_calls[0]["task_state"] == "succeeded"


@pytest.mark.asyncio
async def test_image_upscale_poll_provider_terminal_failure_marks_failed_without_billing(monkeypatch):
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token="token",
    )

    class Store:
        def __init__(self):
            self.transition_calls = []

        async def get(self, team_id, model, request_id):
            return receipt

        async def transition(self, *args, **kwargs):
            self.transition_calls.append(kwargs)
            return receipt

    class Client:
        async def apoll_image_upscale(self, provider_task_id):
            return {"status": 3, "failed_reason": "provider failed", "urls": []}

    store = Store()
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: store)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints._client_for_receipt", lambda _: Client())
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.verify_resume_token", lambda *args, **kwargs: True)

    response = await endpoints.libtv_image_upscale_poll(
        request=_action_request(
            "/v1/libtv/image-upscale/poll",
            orjson.dumps({"request_id": "request-1", "model": "topaz-image-upscaler", "resume_token": "token"}),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert response.status_code == 200
    assert store.transition_calls[0]["task_state"] == "failed"
    assert "billing_event" not in store.transition_calls[0]


@pytest.mark.asyncio
async def test_image_upscale_poll_transient_provider_state_remains_active(monkeypatch):
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token="token",
    )

    class Store:
        async def get(self, team_id, model, request_id):
            return receipt

        async def transition(self, *args, **kwargs):
            raise AssertionError("transient poll must not transition the receipt")

    class Client:
        async def apoll_image_upscale(self, provider_task_id):
            return {"status": 1, "urls": []}

    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: Store())
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints._client_for_receipt", lambda _: Client())
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.verify_resume_token", lambda *args, **kwargs: True)

    response = await endpoints.libtv_image_upscale_poll(
        request=_action_request(
            "/v1/libtv/image-upscale/poll",
            orjson.dumps({"request_id": "request-1", "model": "topaz-image-upscaler", "resume_token": "token"}),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert response.status_code == 200
    assert json.loads(response.body)["receipt"]["task_state"] == "active"


@pytest.mark.asyncio
async def test_image_upscale_poll_is_idempotent_after_terminal_success(monkeypatch):
    initial = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token="token",
        response_cost=0.25,
        api_key="key-1",
        user_id="user-1",
        organization_id="org-1",
    )

    class Store:
        def __init__(self):
            self.receipt = initial
            self.transition_calls = []

        async def get(self, team_id, model, request_id):
            return self.receipt

        async def transition(self, receipt, receipt_key, submission_state, **kwargs):
            self.transition_calls.append(kwargs)
            self.receipt = replace(
                receipt,
                task_state=kwargs["task_state"],
                billing_event_id="event-1",
                terminal_result=kwargs["terminal_result"],
            )
            return self.receipt

    class Client:
        def __init__(self):
            self.poll_calls = 0

        async def apoll_image_upscale(self, provider_task_id):
            self.poll_calls += 1
            return {
                "status": 2,
                "urls": ["https://libtv.example/result.png"],
            }

    store = Store()
    client = Client()
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: store)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints._client_for_receipt", lambda _: client)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.verify_resume_token", lambda *args, **kwargs: True)
    request = _action_request(
        "/v1/libtv/image-upscale/poll",
        orjson.dumps({"request_id": "request-1", "model": "topaz-image-upscaler", "resume_token": "token"}),
    )

    first = await endpoints.libtv_image_upscale_poll(
        request=request, user_api_key_dict=UserAPIKeyAuth(team_id="team-1")
    )
    second = await endpoints.libtv_image_upscale_poll(
        request=_action_request(
            "/v1/libtv/image-upscale/poll",
            orjson.dumps({"request_id": "request-1", "model": "topaz-image-upscaler", "resume_token": "token"}),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert json.loads(second.body)["result"] == {
        "url": "https://libtv.example/result.png",
        "provider_task_id": "task-1",
    }
    assert client.poll_calls == 1
    assert len(store.transition_calls) == 1
    assert store.transition_calls[0]["task_state"] == "succeeded"
    assert "billing_event" in store.transition_calls[0]


@pytest.mark.asyncio
async def test_image_upscale_poll_without_durable_identity_does_not_bill(monkeypatch):
    fingerprint = "f" * 64
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint=fingerprint,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token=make_resume_token(
            "dep-1",
            "task-1",
            "secret",
            team_id="team-1",
            model="topaz-image-upscaler",
            request_id="request-1",
            fingerprint=fingerprint,
        ),
        response_cost=0.25,
    )

    class Store:
        def __init__(self):
            self.transition_calls = []

        async def get(self, team_id, model, request_id):
            return receipt

        async def transition(self, *args, **kwargs):
            self.transition_calls.append(kwargs)
            return receipt

    class Client:
        async def apoll_image_upscale(self, provider_task_id):
            return {
                "status": 2,
                "urls": ["https://libtv.example/result.png"],
            }

    store = Store()
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: store)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints._client_for_receipt", lambda _: Client())
    monkeypatch.setenv("LIBTV_IMAGE_UPSCALE_RESUME_SECRET", "secret")

    response = await endpoints.libtv_image_upscale_poll(
        request=_action_request(
            "/v1/libtv/image-upscale/poll",
            orjson.dumps(
                {
                    "request_id": "request-1",
                    "model": "topaz-image-upscaler",
                    "resume_token": receipt.resume_token,
                }
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert response.status_code == 503
    assert store.transition_calls == []


@pytest.mark.asyncio
async def test_image_upscale_poll_without_authoritative_cost_does_not_bill_zero(monkeypatch):
    fingerprint = "f" * 64
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint=fingerprint,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token=make_resume_token(
            "dep-1",
            "task-1",
            "secret",
            team_id="team-1",
            model="topaz-image-upscaler",
            request_id="request-1",
            fingerprint=fingerprint,
        ),
    )

    class Store:
        def __init__(self):
            self.transition_calls = []

        async def get(self, team_id, model, request_id):
            return receipt

        async def transition(self, *args, **kwargs):
            self.transition_calls.append(kwargs)
            return receipt

    class Client:
        async def apoll_image_upscale(self, provider_task_id):
            return {
                "status": 2,
                "urls": ["https://libtv.example/result.png"],
            }

    store = Store()
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: store)
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints._client_for_receipt", lambda _: Client())
    monkeypatch.setenv("LIBTV_IMAGE_UPSCALE_RESUME_SECRET", "secret")

    response = await endpoints.libtv_image_upscale_poll(
        request=_action_request(
            "/v1/libtv/image-upscale/poll",
            orjson.dumps(
                {
                    "request_id": "request-1",
                    "model": "topaz-image-upscaler",
                    "resume_token": receipt.resume_token,
                }
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert response.status_code == 503
    assert store.transition_calls == []


@pytest.mark.asyncio
async def test_image_upscale_poll_maps_receipt_credential_failure_to_503(monkeypatch):
    receipt = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="submitted",
        deployment_id="dep-1",
        provider_task_id="task-1",
        resume_token="not-used",
    )

    class Store:
        async def get(self, team_id, model, request_id):
            return receipt

    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.get_receipt_store", lambda: Store())
    monkeypatch.setattr(
        "litellm.proxy.image_endpoints.endpoints._client_for_receipt",
        lambda _: (_ for _ in ()).throw(LibTVError(status_code=503, message="credentials unavailable")),
    )

    response = await endpoints.libtv_image_upscale_poll(
        request=_action_request(
            "/v1/libtv/image-upscale/poll",
            orjson.dumps(
                {
                    "request_id": "request-1",
                    "model": "topaz-image-upscaler",
                    "resume_token": "not-used",
                }
            ),
        ),
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1"),
    )

    assert response.status_code == 503
