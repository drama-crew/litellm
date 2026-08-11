import json
import time

import httpx
import pytest

from litellm.exceptions import BadRequestError, Timeout
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, MaskedHTTPStatusError
from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.client import parse_progress
from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.handler import LibTVLLM
from litellm.llms.libtv.image_upscale import (
    ImageUpscaleSubmitter,
    ProviderRejected,
    ProviderTransportError,
    TopazImageUpscaleBuilder,
    make_resume_token,
    verify_resume_token,
)
from litellm.llms.libtv.receipts import ReceiptClaim, StoredReceipt
from litellm.llms.libtv.transfer import ValidatedDelegatedTransfer
from litellm.llms.libtv.transform import build_topaz_upscale_params

_TOPAZ_SOURCE_URL = "https://libtv-res.liblib.art/upload-images/uid/source.mp4"


def _topaz_tool_spec_payload():
    meta = {
        "modelKey": "topaz-video-upscaler",
        "modelVendor": "topazlabs",
        "properties": {
            "resolution": {"default": "1080p", "enum": ["1080p", "2K", "4K"]},
            "specifiedModel": {"default": "prob-4", "enum": ["apo-8", "prob-4"]},
            "fps": {"default": 30, "enum": [24, 30, 60, 90, 120]},
            "slowmo": {"default": "1", "enum": ["1", "2", "3", "5"]},
        },
        "config": {"settings": ["resolution", "specifiedModel", "fps", "slowmo"]},
    }
    return {"data": {"tools": [{"type": "video", "metadata": json.dumps(meta)}]}}


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSyncClient:
    def __init__(self, post_by_path=None, get_payload=None):
        self.post_by_path = post_by_path or {}
        self.get_payload = get_payload
        self.calls = []

    def _path(self, url):
        return url.split("api.liblib.tv", 1)[-1]

    def post(self, url, json=None, headers=None, timeout=None):
        path = self._path(url)
        self.calls.append((path, json))
        queue = self.post_by_path[path]
        item = queue.pop(0) if isinstance(queue, list) else queue
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append((self._path(url), None))
        return FakeResponse(self.get_payload)


class FakeAsyncClient(FakeSyncClient):
    async def post(self, url, json=None, headers=None, timeout=None):
        return FakeSyncClient.post(self, url, json, headers, timeout)

    async def post_once(self, url, json=None, headers=None, timeout=None):
        return FakeSyncClient.post(self, url, json, headers, timeout)

    async def get(self, url, headers=None, timeout=None, params=None):
        return FakeSyncClient.get(self, url, headers, timeout, params)


_CREATE_ROUTES = {
    "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
    "/api/canvas/nodes/batch": {"code": 0, "data": {}},
    "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
}


@pytest.fixture
def topaz_builder():
    return TopazImageUpscaleBuilder()


def submit_body():
    return {
        "request_id": "generation-1",
        "source_url": "https://source.example/input.png",
        "style": "Standard V2",
        "scale": 2,
        "response_cost": 0.25,
        "api_key": "key-1",
        "user_id": "user-1",
        "organization_id": "org-1",
    }


def test_topaz_image_payload_is_flat_and_has_no_mode_type(topaz_builder):
    payload = topaz_builder.build(
        source_url="https://source.example/input.png",
        style="CGI",
        scale=4,
    )
    assert payload["style"] == "CGI"
    assert payload["scale"] == 4
    assert payload["imageList"] == ["https://source.example/input.png"]
    assert "modeType" not in payload


def test_completed_image_progress_exposes_verified_result_metadata():
    result = parse_progress(
        {
            "data": {
                "progresses": [
                    {
                        "taskId": "task-1",
                        "status": 2,
                        "taskResult": json.dumps(
                            {
                                "images": [
                                    {
                                        "imageUrl": "https://provider.example/result.png",
                                        "bytes": 42,
                                        "mime": "image/png",
                                        "width": 8,
                                        "height": 8,
                                        "sha256": "a" * 64,
                                    }
                                ]
                            }
                        ),
                    }
                ]
            }
        },
        "image",
        "task-1",
    )

    assert result["result_metadata"] == {
        "bytes": 42,
        "mime": "image/png",
        "width": 8,
        "height": 8,
        "sha256": "a" * 64,
    }


@pytest.mark.parametrize("sources", [[], ["a.png", "b.png"]])
def test_topaz_image_rejects_zero_or_multiple_sources(topaz_builder, sources):
    with pytest.raises(ValueError, match="exactly one source image"):
        topaz_builder.build(source_urls=sources, style="Standard V2", scale=2)


def test_resume_token_is_signed_and_bound_to_task():
    token = make_resume_token("dep-1", "task-1", "secret")
    assert token.startswith("v2.")
    assert verify_resume_token(token, "secret", deployment_id="dep-1", provider_task_id="task-1")
    assert not verify_resume_token(token, "wrong", deployment_id="dep-1", provider_task_id="task-1")
    assert not verify_resume_token(token, "secret", deployment_id="dep-2", provider_task_id="task-1")


def test_topaz_image_rejects_non_http_source_before_opening_local_file(monkeypatch):
    llm = LibTVLLM()
    monkeypatch.setattr(
        llm,
        "_make_client",
        lambda *args, **kwargs: type(
            "Client",
            (),
            {
                "resolve_model_spec": lambda self, model: {
                    "vendor": "topazlabs",
                    "task_type": "image",
                    "model_key": "topaz-image-upscaler",
                }
            },
        )(),
    )
    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("opened local source"))
    )

    with pytest.raises(LibTVError, match="HTTP"):
        llm.submit_image_upscale(
            "topaz-image-upscaler",
            "tok",
            None,
            {"input_reference": "/tmp/source.png"},
            None,
        )


@pytest.mark.asyncio
async def test_async_http_handler_post_once_does_not_retry_connection_error():
    class Client:
        def __init__(self):
            self.calls = 0

        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, stream=False):
            self.calls += 1
            raise httpx.RemoteProtocolError("connection closed after submit")

        async def aclose(self):
            return None

    handler = object.__new__(AsyncHTTPHandler)
    handler.client = Client()
    handler.timeout = None
    handler.event_hooks = None

    with pytest.raises(httpx.RemoteProtocolError):
        await handler.post_once("https://api.liblib.tv/api/task/generation/create", json={})

    assert handler.client.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        Timeout("generation timed out", "m", "libtv"),
        MaskedHTTPStatusError(
            httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "https://api.liblib.tv/api/task/generation/create"),
                response=httpx.Response(503),
            )
        ),
    ],
)
async def test_paid_create_http_errors_return_unknown_receipt_without_retry(error):
    fake = FakeAsyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "project-1"}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": error,
        }
    )
    lt = LibTVClient(token="t", webid="w", async_client=fake, persistence=None, poll_interval=0)

    receipt = await lt.asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://libtv-res/source.png",
        "Standard V2",
        2,
        "proj-name",
        "request-1",
        "dep-1",
    )

    assert receipt.submission_state == "unknown"
    assert len([path for path, _ in fake.calls if path == "/api/task/generation/create"]) == 1


@pytest.mark.asyncio
async def test_paid_create_response_without_task_id_is_unknown_receipt():
    fake = FakeAsyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "project-1"}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {}},
        }
    )
    lt = LibTVClient(token="t", webid="w", async_client=fake, persistence=None, poll_interval=0)

    receipt = await lt.asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://libtv-res/source.png",
        "Standard V2",
        2,
        "proj-name",
        "request-1",
        "dep-1",
    )

    assert receipt.submission_state == "unknown"


@pytest.mark.asyncio
async def test_paid_create_fails_closed_without_post_once():
    class AsyncClientWithoutPostOnce:
        def __init__(self):
            self.calls = []

        async def post(self, url, json=None, headers=None, timeout=None):
            path = url.split("api.liblib.tv", 1)[-1]
            self.calls.append(path)
            if path == "/api/task/generation/create":
                raise AssertionError("paid create must not fall back to post")
            return FakeResponse({"code": 0, "data": {"projectMeta": {"uuid": "project-1"}}})

    fake = AsyncClientWithoutPostOnce()
    lt = LibTVClient(token="t", webid="w", async_client=fake, persistence=None, poll_interval=0)

    receipt = await lt.asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://libtv-res/source.png",
        "Standard V2",
        2,
        "proj-name",
        "request-1",
        "dep-1",
    )

    assert receipt.submission_state == "not_submitted"
    assert "/api/task/generation/create" not in fake.calls


@pytest.mark.asyncio
async def test_paid_pre_create_timeout_returns_not_submitted_receipt():
    fake = FakeAsyncClient(
        post_by_path={
            "/api/canvas/project/create": Timeout("project timed out", "m", "libtv"),
        }
    )
    lt = LibTVClient(token="t", webid="w", async_client=fake, persistence=None, poll_interval=0)

    receipt = await lt.asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://libtv-res/source.png",
        "Standard V2",
        2,
        "proj-name",
        "request-1",
        "dep-1",
    )

    assert receipt.submission_state == "not_submitted"


@pytest.mark.asyncio
async def test_validated_transfer_uses_dedicated_stream_and_validates_result_shape(monkeypatch):
    monkeypatch.setenv("LIBTV_PLATFORM_SOURCE_HOSTS", "source.platform.example")

    class Redis:
        def __init__(self):
            self.payload = None

        async def zcount(self, *args):
            return 1

        async def set(self, *args, **kwargs):
            return True

        async def xadd(self, stream, fields):
            self.payload = (stream, json.loads(fields["payload"]))

        async def brpop(self, *args, **kwargs):
            return (
                "result",
                json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "bytes": 3,
                            "sha256": "a" * 64,
                            "mime": "image/png",
                            "width": 1,
                            "height": 1,
                            "validation_version": 1,
                            "etags": [{"n": 1, "etag": "e1"}],
                        },
                    }
                ),
            )

    redis = Redis()
    transfer = ValidatedDelegatedTransfer(redis=redis, wait_timeout=0.1)
    result = await transfer.transfer(
        "https://source.platform.example/input.png?X-Amz-Signature=abc&X-Amz-Expires=60",
        3,
        [{"n": 1, "url": "https://bridge.example/part-1"}],
        source_sha256="a" * 64,
        hard_cap=64,
    )

    assert result == [{"n": 1, "etag": "e1"}]
    stream, payload = redis.payload
    assert stream.endswith("validated_media_transfer")
    assert payload["mode"] == "source_transfer"
    assert payload["source"] == {
        "url": "https://source.platform.example/input.png?X-Amz-Signature=abc&X-Amz-Expires=60",
        "bytes": 3,
        "sha256": "a" * 64,
    }
    assert payload["hard_cap"] == 64


@pytest.mark.asyncio
async def test_validated_transfer_rejects_untrusted_source_host(monkeypatch):
    monkeypatch.setenv("LIBTV_PLATFORM_SOURCE_HOSTS", "assets.platform.example")

    class Redis:
        async def zcount(self, *args):
            return 1

        async def set(self, *args, **kwargs):
            return True

        async def xadd(self, *args, **kwargs):
            return "stream-id"

        async def brpop(self, *args, **kwargs):
            return None

    with pytest.raises(LibTVError, match="platform-signed"):
        await ValidatedDelegatedTransfer(Redis()).transfer(
            "https://attacker.example/input.png?X-Amz-Signature=abc&X-Amz-Expires=60",
            3,
            [{"n": 1, "url": "https://bridge.example/part-1"}],
            source_sha256="a" * 64,
            hard_cap=64,
        )


@pytest.mark.asyncio
async def test_validated_transfer_accepts_platform_signed_source_url(monkeypatch):
    monkeypatch.setenv("LIBTV_PLATFORM_SOURCE_HOSTS", "assets.platform.example")

    class Redis:
        def __init__(self):
            self.payload = None

        async def zcount(self, *args):
            return 1

        async def set(self, *args, **kwargs):
            return True

        async def xadd(self, stream, fields):
            self.payload = (stream, json.loads(fields["payload"]))

        async def brpop(self, *args, **kwargs):
            return (
                "result",
                json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "bytes": 3,
                            "sha256": "a" * 64,
                            "mime": "image/png",
                            "width": 1,
                            "height": 1,
                            "validation_version": 1,
                            "etags": [{"n": 1, "etag": "e1"}],
                        },
                    }
                ),
            )

    result = await ValidatedDelegatedTransfer(Redis(), wait_timeout=0.1).transfer(
        "https://assets.platform.example/input.png?X-Amz-Signature=abc&X-Amz-Expires=60",
        3,
        [{"n": 1, "url": "https://bridge.example/part-1"}],
        source_sha256="a" * 64,
        hard_cap=64,
    )

    assert result == [{"n": 1, "etag": "e1"}]


@pytest.mark.asyncio
async def test_validated_transfer_accepts_oss2_presigned_source_url(monkeypatch):
    monkeypatch.setenv("LIBTV_PLATFORM_SOURCE_HOSTS", "assets.platform.example")

    class Redis:
        async def zcount(self, *args):
            return 1

        async def set(self, *args, **kwargs):
            return True

        async def xadd(self, *args, **kwargs):
            return "stream-id"

        async def brpop(self, *args, **kwargs):
            return (
                "result",
                json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "bytes": 3,
                            "sha256": "a" * 64,
                            "mime": "image/png",
                            "width": 1,
                            "height": 1,
                            "validation_version": 1,
                            "etags": [{"n": 1, "etag": "e1"}],
                        },
                    }
                ),
            )

    expires = int(time.time()) + 60
    result = await ValidatedDelegatedTransfer(Redis()).transfer(
        f"https://assets.platform.example/input.png?OSSAccessKeyId=test-key&Expires={expires}&Signature=real-oss2-signature",
        3,
        [{"n": 1, "url": "https://bridge.example/part-1"}],
        source_sha256="a" * 64,
        hard_cap=64,
    )

    assert result == [{"n": 1, "etag": "e1"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "OSSAccessKeyId=test-key&Expires=1&Signature=real-oss2-signature",
        "OSSAccessKeyId=test-key&Expires=not-a-timestamp&Signature=real-oss2-signature",
        "OSSAccessKeyId=test-key&Expires=9999999999",
    ],
)
async def test_validated_transfer_rejects_expired_or_unsigned_oss2_source_url(monkeypatch, query):
    monkeypatch.setenv("LIBTV_PLATFORM_SOURCE_HOSTS", "assets.platform.example")

    class Redis:
        async def zcount(self, *args):
            return 1

    with pytest.raises(LibTVError, match="platform-signed"):
        await ValidatedDelegatedTransfer(Redis()).transfer(
            f"https://assets.platform.example/input.png?{query}",
            3,
            [{"n": 1, "url": "https://bridge.example/part-1"}],
            source_sha256="a" * 64,
            hard_cap=64,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("mime", None), ("width", 0), ("height", "1"), ("validation_version", 2)])
async def test_validated_transfer_rejects_unvalidated_worker_media_metadata(monkeypatch, field, value):
    monkeypatch.setenv("LIBTV_PLATFORM_SOURCE_HOSTS", "assets.platform.example")

    class Redis:
        async def zcount(self, *args):
            return 1

        async def set(self, *args, **kwargs):
            return True

        async def xadd(self, *args, **kwargs):
            return "stream-id"

        async def brpop(self, *args, **kwargs):
            result = {
                "bytes": 3,
                "sha256": "a" * 64,
                "mime": "image/png",
                "width": 1,
                "height": 1,
                "validation_version": 1,
                "etags": [{"n": 1, "etag": "e1"}],
            }
            result[field] = value
            return "result", json.dumps({"ok": True, "result": result})

    with pytest.raises(LibTVError, match="metadata"):
        await ValidatedDelegatedTransfer(Redis()).transfer(
            "https://assets.platform.example/input.png?X-Amz-Signature=abc&X-Amz-Expires=60",
            3,
            [{"n": 1, "url": "https://bridge.example/part-1"}],
            source_sha256="a" * 64,
            hard_cap=64,
        )


@pytest.mark.asyncio
async def test_submit_returns_submitted_receipt():
    class Provider:
        async def create(self, payload):
            return {"task_id": "provider-task-1"}

    receipt = await ImageUpscaleSubmitter(("primary", Provider())).submit(submit_body())
    assert receipt.submission_state == "submitted"
    assert receipt.provider_task_id == "provider-task-1"


@pytest.mark.asyncio
async def test_post_create_transport_error_returns_unknown_without_failover():
    class Provider:
        def __init__(self):
            self.calls = 0

        async def create(self, payload):
            self.calls += 1
            raise ProviderTransportError(crossed_create_boundary=True)

    primary = Provider()
    secondary = Provider()
    receipt = await ImageUpscaleSubmitter(("primary", primary), ("secondary", secondary)).submit(submit_body())
    assert receipt.submission_state == "unknown"
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_explicit_capacity_rejection_can_try_second_deployment():
    class RejectedProvider:
        async def create(self, payload):
            raise ProviderRejected("capacity")

    class AcceptedProvider:
        def __init__(self):
            self.calls = 0

        async def create(self, payload):
            self.calls += 1
            return {"task_id": "provider-task-2"}

    secondary = AcceptedProvider()
    receipt = await ImageUpscaleSubmitter(("primary", RejectedProvider()), ("secondary", secondary)).submit(
        submit_body()
    )
    assert receipt.submission_state == "submitted"
    assert receipt.deployment_id == "secondary"
    assert secondary.calls == 1


@pytest.mark.asyncio
async def test_client_pool_resolves_each_deployment_credential_for_failover(monkeypatch):
    class Store:
        def __init__(self):
            self.receipt = None

        async def readiness(self):
            return True

        async def claim(
            self,
            team_id,
            model,
            request_id,
            fingerprint,
            deployment_id,
            *,
            response_cost=None,
        ):
            self.receipt = StoredReceipt(
                team_id=team_id,
                model=model,
                request_id=request_id,
                fingerprint=fingerprint,
                submission_state="submitting",
                deployment_id=deployment_id,
                response_cost=response_cost,
            )
            return ReceiptClaim("owner", f"receipt-{deployment_id}", self.receipt)

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
                response_cost=receipt.response_cost,
            )
            return self.receipt

    store = Store()
    monkeypatch.setattr("litellm.llms.libtv.client.get_receipt_store", lambda: store)
    monkeypatch.setenv("LIBTV_TOKEN", "primary-token")
    monkeypatch.setenv("LIBTV_WEBID", "primary-webid")
    monkeypatch.setenv("LIBTV_TOKEN_2", "secondary-token")
    monkeypatch.setenv("LIBTV_WEBID_2", "secondary-webid")
    calls = []

    async def fake_acreate(self, *args, **kwargs):
        calls.append(self.token)
        if self.token == "primary-token":
            raise ProviderRejected("capacity")
        return {"task_id": "secondary-task"}

    monkeypatch.setattr(LibTVClient, "acreate", fake_acreate)
    client = LibTVClient(token="primary-token", webid="primary-webid")

    receipt = await client.asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://source.example/input.png",
        "Standard V2",
        2,
        "project",
        "request-1",
        "primary",
        team_id="team-1",
        source_sha256="f" * 64,
        durable_receipts=True,
        response_cost=0.42,
        deployment_pool=[
            {
                "id": "primary",
                "api_key": "os.environ/LIBTV_TOKEN",
                "webid": "os.environ/LIBTV_WEBID",
            },
            {
                "id": "secondary",
                "api_key": "os.environ/LIBTV_TOKEN_2",
                "webid": "os.environ/LIBTV_WEBID_2",
            },
        ],
    )

    assert calls == ["primary-token", "secondary-token"]
    assert receipt.submission_state == "submitted"
    assert receipt.deployment_id == "secondary"
    assert verify_resume_token(
        receipt.resume_token or "",
        "secondary-token",
        deployment_id="secondary",
        provider_task_id="secondary-task",
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint=store.receipt.fingerprint,
    )
    assert not verify_resume_token(receipt.resume_token or "", "primary-token")
    assert store.receipt.response_cost == 0.42


@pytest.mark.asyncio
async def test_client_reuses_submitted_receipt_without_reuploading_expired_source(monkeypatch):
    fingerprint = "f" * 64
    stored = StoredReceipt(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint=fingerprint,
        submission_state="submitted",
        deployment_id="primary",
        provider_task_id="task-1",
        resume_token="receipt-token",
        response_cost=0.42,
        api_key="key-1",
        user_id="user-1",
        organization_id="org-1",
    )

    class Store:
        async def readiness(self):
            return True

        async def claim(self, *args, **kwargs):
            return ReceiptClaim("existing", "receipt-primary", stored)

    async def failed_upload(*args, **kwargs):
        raise AssertionError("existing submitted receipt must return before source upload")

    monkeypatch.setattr("litellm.llms.libtv.client.get_receipt_store", lambda: Store())
    monkeypatch.setattr(LibTVClient, "aensure_libtv_url", failed_upload)

    receipt = await LibTVClient(token="token", webid="webid").asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://source.example/expired-signature.png",
        "Standard V2",
        2,
        "project",
        "request-1",
        "primary",
        team_id="team-1",
        source_sha256=fingerprint,
        durable_receipts=True,
        response_cost=0.42,
        receipt_api_key="key-1",
        receipt_user_id="user-1",
        receipt_organization_id="org-1",
    )

    assert receipt.submission_state == "submitted"
    assert receipt.provider_task_id == "task-1"


def _gen_params(calls):
    return next(j["params"] for path, j in calls if path == "/api/task/generation/create")


def test_build_topaz_upscale_params_defaults():
    params = build_topaz_upscale_params("upscale me", {})
    assert params["prompt"] == "upscale me"
    assert params["resolution"] == "1080p"
    assert params["specifiedModel"] == "prob-4"
    assert params["slowmo"] == "1"
    assert "modeType" not in params
    assert "fps" not in params  # only present when specifiedModel != prob-4


def test_build_topaz_upscale_params_explicit_values():
    params = build_topaz_upscale_params(
        "x",
        {"resolution": "4K", "specifiedModel": "apo-8", "fps": 60, "slowmo": "3"},
    )
    assert params["resolution"] == "4K"
    assert params["specifiedModel"] == "apo-8"
    assert params["fps"] == 60
    assert params["slowmo"] == "3"


def test_build_topaz_upscale_params_invalid_values_fall_back_to_defaults():
    params = build_topaz_upscale_params("x", {"resolution": "8K", "specifiedModel": "bogus", "slowmo": "9"})
    assert params["resolution"] == "1080p"
    assert params["specifiedModel"] == "prob-4"
    assert params["slowmo"] == "1"


def test_build_topaz_upscale_params_fps_only_present_for_non_prob4_model():
    params = build_topaz_upscale_params("x", {"specifiedModel": "prob-4", "fps": 60})
    assert "fps" not in params

    params2 = build_topaz_upscale_params("x", {"specifiedModel": "apo-8", "fps": 24})
    assert params2["fps"] == 24


def test_build_topaz_upscale_params_invalid_fps_falls_back_to_30():
    params = build_topaz_upscale_params("x", {"specifiedModel": "apo-8", "fps": 999})
    assert params["fps"] == 30


def test_video_generation_topaz_upscale_sets_no_mode_type_and_uploads_source_video():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "video_references": _TOPAZ_SOURCE_URL},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert "modeType" not in gen_params
    assert gen_params["videoList"] == [_TOPAZ_SOURCE_URL]
    assert gen_params["resolution"] == "1080p"
    assert gen_params["specifiedModel"] == "prob-4"
    assert gen_params["slowmo"] == "1"
    assert "fps" not in gen_params


def test_video_generation_topaz_upscale_accepts_input_reference_alias():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "input_reference": _TOPAZ_SOURCE_URL},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert gen_params["videoList"] == [_TOPAZ_SOURCE_URL]


def test_video_generation_topaz_upscale_forwards_settings():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {
            "webid": "w",
            "video_references": _TOPAZ_SOURCE_URL,
            "resolution": "4K",
            "specifiedModel": "apo-8",
            "fps": 60,
            "slowmo": "5",
        },
        None,
        client=fake,
    )
    gen_params = _gen_params(fake.calls)
    assert gen_params["resolution"] == "4K"
    assert gen_params["specifiedModel"] == "apo-8"
    assert gen_params["fps"] == 60
    assert gen_params["slowmo"] == "5"


def test_video_generation_topaz_upscale_missing_source_video_raises_bad_request():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    with pytest.raises(BadRequestError):
        llm.video_generation("topaz-video-upscaler", "upscale this", "tok", None, {"webid": "w"}, None, client=fake)


def test_video_generation_topaz_upscale_omitted_resolution_bills_at_1080p_default():
    # regression: build_topaz_upscale_params defaults resolution to 1080p internally, but
    # the usage/cost path reads optional_params directly. If that default doesn't reach
    # usage.video_resolution, the tiered-only deployment (no plain output_cost_per_second)
    # bills $0. video_resolution must be the same 1080p default the payload used.
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "video_references": _TOPAZ_SOURCE_URL, "seconds": "5"},
        None,
        client=fake,
    )
    gen_params = _gen_params(fake.calls)
    assert gen_params["resolution"] == "1080p"
    assert vo.usage is not None
    assert vo.usage["video_resolution"] == "1080p"


def test_video_generation_topaz_upscale_explicit_resolution_reaches_usage():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "video_references": _TOPAZ_SOURCE_URL, "resolution": "4K", "seconds": "5"},
        None,
        client=fake,
    )
    assert vo.usage is not None
    assert vo.usage["video_resolution"] == "4K"


@pytest.mark.asyncio
async def test_avideo_generation_topaz_upscale_sets_no_mode_type_and_uploads_source_video():
    fake = FakeAsyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = await llm.avideo_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "video_references": _TOPAZ_SOURCE_URL},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert "modeType" not in gen_params
    assert gen_params["videoList"] == [_TOPAZ_SOURCE_URL]
