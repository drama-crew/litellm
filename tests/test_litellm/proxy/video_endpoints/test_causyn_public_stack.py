from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import litellm
import litellm.llms.causyn.handler as causyn_module
import litellm.proxy.proxy_server as proxy_server
import litellm.proxy.video_endpoints.endpoints as video_endpoints
from litellm import Router
from litellm.llms.causyn import CausynVideoHandler
from litellm.llms.libtv.video_generate import VideoGenerateError, VideoGenerateSettings
from litellm.proxy._types import (
    LiteLLM_VerificationTokenView,
    ProxyException,
    hash_token,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider


TASK_ID = "0123456789abcdef0123456789abcdef"
CAUSYN_MODEL = "causyn-1.0"
CAUSYN_MODEL_ID = "causyn-1-0"
STAGING_URL = "https://target.example/video.mp4?signature=test"


class _State:
    def __init__(self) -> None:
        self.status: str | None = "queued"
        self.result: dict[str, object] = {
            "validation_version": "video-v1",
            "staging_key": f"staging/video-tasks/{TASK_ID}.mp4",
            "staging_url": STAGING_URL,
            "etag": "etag-1",
            "bytes": 123,
            "content_type": "video/mp4",
            "duration_seconds": 5.0,
            "width": 768,
            "height": 512,
            "sha256": "a" * 64,
        }
        self.fetch_error: Exception | None = None
        self.enqueued: list[dict[str, object]] = []
        self.fetch_calls = 0

    def body(self) -> dict[str, object]:
        if self.fetch_error is not None:
            raise self.fetch_error
        body: dict[str, object] = {"status": self.status}
        if self.status == "succeeded":
            body["result"] = self.result
        return body


class _Billing:
    def __init__(self) -> None:
        self.stored_usage: dict[str, object] | None = None
        self.mark_calls: list[tuple[str, float, float]] = []
        self.marked_keys: set[str] = set()
        self.outbox_events: list[object] = []

    async def enqueue(self, _: object, event: object) -> bool:
        self.outbox_events.append(event)
        return True

    async def store_video_task_usage(
        self,
        billing_key: str,
        duration_seconds: float,
        video_resolution: str | None,
    ) -> None:
        self.stored_usage = {
            "duration_seconds": duration_seconds,
            "video_resolution": video_resolution,
        }

    async def get_video_task_usage(self, billing_key: str) -> dict[str, object] | None:
        return self.stored_usage

    async def mark_video_billed(
        self,
        billing_key: str,
        duration_seconds: float,
        response_cost: float,
    ) -> bool:
        self.mark_calls.append((billing_key, duration_seconds, response_cost))
        if billing_key in self.marked_keys:
            return False
        self.marked_keys.add(billing_key)
        return True


@dataclass
class _ContentGet:
    calls: list[tuple[str, object, bool]] = field(default_factory=list)

    async def __call__(self, url: str, timeout: object, follow_redirects: bool) -> httpx.Response:
        self.calls.append((url, timeout, follow_redirects))
        return httpx.Response(200, content=b"video-bytes", request=httpx.Request("GET", url))


class _ProxyLogging:
    async def pre_call_hook(self, *, data: dict, **_: object) -> dict:
        return data

    async def during_call_hook(self, **_: object) -> None:
        return None

    async def post_call_success_hook(self, *, response: object, **_: object) -> object:
        return response

    async def post_call_failure_hook(self, **_: object) -> None:
        return None

    async def post_call_response_headers_hook(self, **_: object) -> dict[str, str]:
        return {}

    async def update_request_status(self, **_: object) -> None:
        return None


class _ProxyConfig:
    async def _get_hierarchical_router_settings(self, **_: object) -> None:
        return None


class _AuthStore:
    """Small DB seam for exercising the production virtual-key resolver."""

    def __init__(self, keys: dict[str, list[str]]) -> None:
        self.keys = {hash_token(key): models for key, models in keys.items()}
        self.lookups: list[str] = []

    async def get_data(self, *, token: str, table_name: str, **_: object) -> LiteLLM_VerificationTokenView | None:
        assert table_name == "combined_view"
        self.lookups.append(token)
        models = self.keys.get(token)
        if models is None:
            return None
        return LiteLLM_VerificationTokenView(token=token, models=models)


@dataclass
class _Stack:
    app: FastAPI
    client: TestClient
    router: Router
    state: _State
    billing: _Billing
    content_get: _ContentGet
    handler: CausynVideoHandler
    allowed_key: str
    other_model_key: str


@pytest.fixture
def stack(monkeypatch: pytest.MonkeyPatch) -> _Stack:
    state = _State()
    billing = _Billing()
    content_get = _ContentGet()

    async def enqueue(payload: dict[str, object], *, redis_factory: object, settings: object) -> str:
        state.enqueued.append(payload)
        return "1-0"

    async def fetch_status(task_id: str, *, redis: object) -> dict[str, object]:
        assert task_id == TASK_ID
        state.fetch_calls += 1
        return state.body()

    settings = VideoGenerateSettings(
        source_hosts=frozenset({"source.example"}),
        target_hosts=frozenset({"target.example"}),
    )
    handler = CausynVideoHandler(
        redis_factory=lambda: object(),
        settings_factory=lambda: settings,
        content_get=content_get,
        task_id_factory=lambda: TASK_ID,
        clock=lambda: 2_000_000_000.0,
        persistence_factory=lambda: billing,
        billing_enqueue=billing.enqueue,
    )
    monkeypatch.setattr(causyn_module, "enqueue_video_generate", enqueue)
    monkeypatch.setattr(causyn_module, "fetch_video_generate_status", fetch_status)
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", "true")

    old_map = list(litellm.custom_provider_map)
    old_custom_providers = list(litellm._custom_providers)
    old_provider_list = list(litellm.provider_list)
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [{"provider": "causyn", "custom_handler": handler}],
    )
    litellm.utils.custom_llm_setup()

    router = Router(
        model_list=[
            {
                "model_name": CAUSYN_MODEL,
                "litellm_params": {
                    "model": CAUSYN_MODEL,
                    "custom_llm_provider": "causyn",
                },
                "model_info": {
                    "id": CAUSYN_MODEL_ID,
                    "output_cost_per_second_768x512": 0.1,
                },
            }
        ],
        num_retries=0,
    )

    allowed_key = "sk-causyn"
    other_model_key = "sk-other-model"
    auth_store = _AuthStore(
        {
            allowed_key: [CAUSYN_MODEL],
            other_model_key: ["other-model"],
        }
    )
    auth_cache = UserApiKeyCache()

    proxy_logging = _ProxyLogging()
    proxy_config = _ProxyConfig()
    for name, value in {
        "llm_router": router,
        "llm_model_list": router.model_list,
        "proxy_logging_obj": proxy_logging,
        "general_settings": {"disable_budget_reservation": True},
        "proxy_config": proxy_config,
        "master_key": "sk-test-master",
        "prisma_client": auth_store,
        "user_api_key_cache": auth_cache,
        "user_custom_auth": None,
        "select_data_generator": None,
        "user_model": None,
        "user_temperature": None,
        "user_request_timeout": None,
        "user_max_tokens": None,
        "user_api_base": None,
        "version": "test",
    }.items():
        monkeypatch.setattr(proxy_server, name, value)

    app = FastAPI()

    @app.exception_handler(ProxyException)
    async def proxy_exception_handler(_: Request, exc: ProxyException) -> JSONResponse:
        status_code = int(exc.code) if exc.code else 500
        return JSONResponse(status_code=status_code, content={"error": {"message": exc.message}})

    app.include_router(video_endpoints.router)
    client = TestClient(app, raise_server_exceptions=False)

    yield _Stack(
        app=app,
        client=client,
        router=router,
        state=state,
        billing=billing,
        content_get=content_get,
        handler=handler,
        allowed_key=allowed_key,
        other_model_key=other_model_key,
    )

    litellm.provider_list[:] = old_provider_list
    litellm._custom_providers[:] = old_custom_providers
    litellm.custom_provider_map[:] = old_map


def _create_body() -> dict[str, object]:
    return {
        "model": CAUSYN_MODEL,
        "prompt": "animate the reference",
        "seconds": "5",
        "size": "768x512",
        "aspect_ratio": "3:2",
        "reference_images": ["https://source.example/reference.png"],
    }


def _auth_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_public_endpoint_create_decodes_routes_and_bills_idempotently(stack: _Stack) -> None:
    created = stack.client.post("/v1/videos", json=_create_body(), headers=_auth_headers(stack.allowed_key))
    assert created.status_code == 200, created.text

    video_id = created.json()["id"]
    decoded = decode_video_id_with_provider(video_id)
    assert decoded == {
        "video_id": TASK_ID,
        "custom_llm_provider": "causyn",
        "model_id": CAUSYN_MODEL_ID,
    }
    assert stack.state.enqueued[0]["model"] == CAUSYN_MODEL

    stack.state.status = "succeeded"
    first = stack.client.get(f"/v1/videos/{video_id}", headers=_auth_headers(stack.allowed_key))
    second = stack.client.get(f"/v1/videos/{video_id}", headers=_auth_headers(stack.allowed_key))

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["status"] == "completed"
    assert float(first.headers["x-litellm-response-cost"]) == pytest.approx(0.5)
    assert float(second.headers.get("x-litellm-response-cost") or 0.0) == pytest.approx(0.5)
    assert len(stack.billing.outbox_events) == 2
    assert [event.event_id for event in stack.billing.outbox_events] == [
        f"causyn-video:{TASK_ID}",
        f"causyn-video:{TASK_ID}",
    ]
    assert [event.response_cost for event in stack.billing.outbox_events] == [
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]


def test_public_endpoint_content_uses_real_router_and_handler(stack: _Stack) -> None:
    created = stack.client.post("/v1/videos", json=_create_body(), headers=_auth_headers(stack.allowed_key))
    video_id = created.json()["id"]
    stack.state.status = "succeeded"

    response = stack.client.get(f"/v1/videos/{video_id}/content", headers=_auth_headers(stack.allowed_key))

    assert response.status_code == 200, response.text
    assert response.content == b"video-bytes"
    assert stack.content_get.calls == [(STAGING_URL, 600, False)]


def test_public_endpoint_virtual_key_allowlist_cannot_be_bypassed(stack: _Stack) -> None:
    created = stack.client.post("/v1/videos", json=_create_body(), headers=_auth_headers(stack.allowed_key))
    video_id = created.json()["id"]

    denied = stack.client.get(f"/v1/videos/{video_id}", headers=_auth_headers(stack.other_model_key))

    assert denied.status_code == 403
    assert stack.state.status == "queued"
    assert stack.state.fetch_calls == 0


@pytest.mark.parametrize(
    ("operation", "status_code"),
    [
        ("status_not_found", 404),
        ("status_service_error", 503),
        ("content_not_ready", 409),
        ("content_unavailable", 502),
    ],
)
def test_public_endpoint_preserves_causyn_error_statuses(
    stack: _Stack,
    operation: str,
    status_code: int,
) -> None:
    created = stack.client.post("/v1/videos", json=_create_body(), headers=_auth_headers(stack.allowed_key))
    video_id = created.json()["id"]

    if operation == "status_not_found":
        stack.state.status = None
        response = stack.client.get(f"/v1/videos/{video_id}", headers=_auth_headers(stack.allowed_key))
    elif operation == "status_service_error":
        stack.state.fetch_error = VideoGenerateError("misconfigured", "redis unavailable")
        response = stack.client.get(f"/v1/videos/{video_id}", headers=_auth_headers(stack.allowed_key))
    elif operation == "content_not_ready":
        response = stack.client.get(f"/v1/videos/{video_id}/content", headers=_auth_headers(stack.allowed_key))
    else:
        stack.state.status = "succeeded"
        stack.state.result["staging_url"] = None
        response = stack.client.get(f"/v1/videos/{video_id}/content", headers=_auth_headers(stack.allowed_key))

    assert response.status_code == status_code, response.text


@pytest.mark.asyncio
async def test_sync_and_async_public_apis_share_causyn_retrieval_contract(stack: _Stack) -> None:
    stack.state.status = "succeeded"
    encoded_id = f"causyn_{TASK_ID}"

    sync_status = litellm.video_status(
        video_id=encoded_id,
        custom_llm_provider="causyn",
        model_info={"output_cost_per_second_768x512": 0.1},
    )
    async_status = await litellm.avideo_status(
        video_id=encoded_id,
        custom_llm_provider="causyn",
        model_info={"output_cost_per_second_768x512": 0.1},
    )
    sync_content = litellm.video_content(video_id=encoded_id, custom_llm_provider="causyn")
    async_content = await litellm.avideo_content(video_id=encoded_id, custom_llm_provider="causyn")

    assert sync_status.status == "completed"
    assert async_status.status == "completed"
    assert sync_content == async_content == b"video-bytes"


def test_legacy_causyn_id_and_other_provider_encoding_remain_decodable() -> None:
    legacy = decode_video_id_with_provider(f"causyn_{TASK_ID}")
    other_id = encode_video_id_with_provider(TASK_ID, "libtv", "libtv-model")
    other = decode_video_id_with_provider(other_id)

    assert legacy["custom_llm_provider"] == "causyn"
    assert legacy["video_id"] == f"causyn_{TASK_ID}"
    assert other["custom_llm_provider"] == "libtv"
