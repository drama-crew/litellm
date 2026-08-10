import asyncio
import copy
import json
from types import SimpleNamespace
from typing import Any, Dict

import orjson
import pytest
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.image_endpoints import endpoints
from litellm.llms.libtv.handler import LibTVLLM
from litellm.llms.libtv.receipts import ReceiptClaim, StoredReceipt, request_fingerprint
from litellm.llms.libtv.image_upscale import verify_resume_token
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


class _EndpointReceiptStore:
    def __init__(self):
        self.receipt = None
        self.receipt_key = "endpoint-receipt"

    async def claim(self, team_id, model, request_id, fingerprint, deployment_id):
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
    user_api_key = UserAPIKeyAuth(team_id="team-1")
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

    user_api_key = UserAPIKeyAuth(team_id="team-1")
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
        user_api_key_dict=UserAPIKeyAuth(),
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
        user_api_key_dict=UserAPIKeyAuth(),
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
