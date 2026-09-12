import hashlib
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.types.router import GenericLiteLLMParams
from tests.test_litellm.proxy.video_endpoints.test_moderation_execution import SECRET, ticket


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_standard_openai_completion_imports_existing_content_adapter_privately(monkeypatch, legacy):
    from starlette.requests import Request

    from litellm.llms.custom_httpx import llm_http_handler
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.video_endpoints import endpoints, openapi_log_capture
    from litellm.proxy.video_endpoints import moderation_bridge as bridge
    from litellm.videos.content_sink import VIDEO_CONTENT_SINK

    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.test")
    raw = b"synthetic-mp4-original-content"
    content_calls = []

    async def provider(request):
        content_calls.append(request.url.path)
        assert request.url.path == "/v1/videos/native/content"
        assert request.headers["authorization"] == "Bearer synthetic-provider-key"
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as provider_client:
        handler = Mock()
        handler.client = provider_client

        async def get(**kwargs):
            return await provider_client.get(**kwargs)

        handler.get = get
        monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", lambda **kwargs: handler)
        config = OpenAIVideoConfig()
        status = config.transform_video_status_retrieve_response(
            httpx.Response(
                200,
                json={"id": "native", "object": "video", "status": "completed", "model": "sora-2"},
                request=httpx.Request("GET", "https://api.synthetic.invalid/v1/videos/native"),
            ),
            Mock(),
            custom_llm_provider="openai",
        )
        assert status._hidden_params == {} and status.object_store_result is None

        async def process(self, **kwargs):
            if kwargs["route_type"] == "avideo_status":
                return status
            assert kwargs["route_type"] == "avideo_content"
            return await llm_http_handler.BaseLLMHTTPHandler().async_video_content_handler(
                video_id=status.id,
                video_content_provider_config=config,
                custom_llm_provider="openai",
                litellm_params=GenericLiteLLMParams(
                    api_base="https://api.synthetic.invalid/v1", api_key="synthetic-provider-key"
                ),
                logging_obj=Mock(),
                timeout=120,
                api_key="synthetic-provider-key",
            )

        monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "base_process_llm_request", process)
        app = FastAPI()
        app.include_router(execution.router)
        app.include_router(endpoints.router)
        auth = UserAPIKeyAuth(api_key="a" * 64, user_id="owner", team_id="owner", metadata={"openapi_key_id": "key"})
        app.dependency_overrides[user_api_key_auth] = lambda: auth
        monkeypatch.setattr(bridge, "legacy_owner", AsyncMock(return_value=True))
        monkeypatch.setattr(openapi_log_capture, "database", lambda: None)
        outputs = []
        approved = []
        uploaded = []
        digest = hashlib.sha256(raw).hexdigest()

        async def control(request):
            body = json.loads(request.content)
            if request.url.path.endswith("/lookup"):
                return httpx.Response(200, json={"bound": False})
            if request.url.path.endswith("/view"):
                return httpx.Response(
                    200,
                    json={
                        "id": "mod_video_intent",
                        "model": "sora-2",
                        "state": "completed" if approved else "output_moderation",
                        "moderation_status": "approved" if approved else "pending",
                        "policy_source": "model",
                        "created_at": 1,
                        "output": {"url": "https://private.synthetic.invalid/approved"} if approved else None,
                    },
                )
            if request.url.path.endswith("/uploads"):
                assert body["principal"]["fingerprint"] == "a" * 64
                return httpx.Response(
                    200,
                    json={
                        "url": "https://private.synthetic.invalid/upload",
                        "headers": {"x-oss-object-acl": "private"},
                        "reference": "oss://private/moderation/uploads/" + "a" * 64 + "/" + digest,
                    },
                )
            if request.url.path.endswith("/execution"):
                return httpx.Response(
                    200,
                    json={
                        "native_id": status.id,
                        "principal": {"fingerprint": "a" * 64, "user_id": "owner", "team_id": "owner"},
                        "model": "sora-2",
                        "route": "avideo_generation",
                        "billing": {"request_id": "original-consumption"},
                        "state": "submitted",
                        "upload_ticket": "bound-output-upload",
                    },
                )
            if request.url.path.endswith("/output-upload"):
                assert body["native_id"] == status.id
                assert body["digest"] == digest and body["size"] == len(raw)
                return httpx.Response(
                    200,
                    json={
                        "url": "https://private.synthetic.invalid/upload",
                        "headers": {"x-oss-object-acl": "private"},
                        "reference": "oss://private/moderation/outputs/intent/" + digest,
                    },
                )
            assert request.url.path.endswith("/output")
            outputs.append(body)
            return httpx.Response(200, json={"accepted": True})

        async def private_upload(request):
            uploaded.append(await request.aread())
            assert request.headers["x-oss-object-acl"] == "private"
            return httpx.Response(200)

        app.state.moderation_transport = httpx.MockTransport(control)
        app.state.moderation_media_transport = httpx.MockTransport(private_upload)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fork") as client:
            if legacy:
                source = await bridge.source_descriptor(
                    Request(
                        {
                            "type": "http",
                            "method": "POST",
                            "path": "/videos/edits",
                            "headers": [],
                            "query_string": b"",
                            "app": app,
                        }
                    ),
                    auth,
                    {"video": {"id": status.id}},
                )
                assert source["native_id"] == status.id
                assert source["url"] == "oss://private/moderation/uploads/" + "a" * 64 + "/" + digest
            else:
                response = await client.post(
                    "/internal/moderation/collect",
                    json={"ticket": ticket(purpose="collect", model="sora-2")},
                    headers={"Authorization": "Bearer " + SECRET},
                )
                assert response.status_code == 200, response.text
                assert outputs[0]["native_id"] == status.id
                assert outputs[0]["facts"]["private_reference"] == "oss://private/moderation/outputs/intent/" + digest
                assert outputs[0]["facts"]["digest"] == digest
                assert "original-content" not in response.text
                pending = await client.get("/v1/videos/mod_video_intent/content")
                assert pending.status_code == 409
                approved.append(True)
                published = await client.get("/v1/videos/mod_video_intent/content")
                assert published.status_code == 302
                assert published.headers["location"] == "https://private.synthetic.invalid/approved"
        assert content_calls == ["/v1/videos/native/content"]
        assert uploaded == [raw]
        assert VIDEO_CONTENT_SINK.get() is None
        unchanged = await llm_http_handler.BaseLLMHTTPHandler().async_video_content_handler(
            video_id=status.id,
            video_content_provider_config=config,
            custom_llm_provider="openai",
            litellm_params=GenericLiteLLMParams(
                api_base="https://api.synthetic.invalid/v1", api_key="synthetic-provider-key"
            ),
            logging_obj=Mock(),
            timeout=120,
            api_key="synthetic-provider-key",
        )
        assert unchanged == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["limit", "declared-limit", "timeout", "cancel"])
async def test_content_stream_bounds_close_response(failure):
    import asyncio
    from tempfile import TemporaryFile

    from litellm.videos.content_sink import VideoContentSink, stream_content_to_sink

    closed = []
    started = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            if failure in {"timeout", "cancel"}:
                await asyncio.Event().wait()
            yield b"oversized-synthetic"

        async def aclose(self):
            closed.append(True)

    async def provider(request):
        return httpx.Response(
            200, headers={"content-length": "100"} if failure == "declared-limit" else {}, stream=Stream()
        )

    with TemporaryFile() as media:
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            sink = VideoContentSink(media, max_bytes=8, timeout_seconds=0.01 if failure == "timeout" else 120)
            task = asyncio.create_task(stream_content_to_sink(client, "https://synthetic.invalid/content", {}, sink))
            if failure == "cancel":
                await started.wait()
                task.cancel()
            with pytest.raises(
                asyncio.CancelledError if failure == "cancel" else TimeoutError if failure == "timeout" else ValueError
            ):
                await task
            assert closed == [True]
            assert media.tell() == 0


@pytest.mark.asyncio
async def test_materializer_cancellation_resets_context_and_closes_file(monkeypatch):
    import asyncio
    from tempfile import TemporaryFile

    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.video_endpoints import endpoints, moderation_content
    from litellm.videos.content_sink import VIDEO_CONTENT_SINK

    files = []

    def temporary():
        media = TemporaryFile()
        files.append(media)
        return media

    async def cancelled(*args):
        assert VIDEO_CONTENT_SINK.get().file is files[-1]
        raise asyncio.CancelledError()

    monkeypatch.setattr(moderation_content, "TemporaryFile", temporary)
    monkeypatch.setattr(endpoints, "video_content", cancelled)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""})
    with pytest.raises(asyncio.CancelledError):
        await moderation_content.materialize_content(request, UserAPIKeyAuth(), "native")
    assert files[0].closed
    assert VIDEO_CONTENT_SINK.get() is None


@pytest.mark.asyncio
async def test_content_sink_isolated_between_concurrent_tasks():
    import asyncio
    from tempfile import TemporaryFile

    from litellm.videos.content_sink import VIDEO_CONTENT_SINK, VideoContentSink, stream_content_to_sink

    entered = []
    ready = asyncio.Event()

    async def run(name):
        with TemporaryFile() as media:
            sink = VideoContentSink(media)
            token = VIDEO_CONTENT_SINK.set(sink)
            try:
                entered.append(name)
                if len(entered) == 2:
                    ready.set()
                await ready.wait()
                assert VIDEO_CONTENT_SINK.get() is sink
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, content=name.encode()))
                ) as client:
                    await stream_content_to_sink(client, "https://synthetic.invalid/" + name, {}, sink)
                media.seek(0)
                return media.read()
            finally:
                VIDEO_CONTENT_SINK.reset(token)

    assert await asyncio.gather(run("first"), run("second")) == [b"first", b"second"]
    assert VIDEO_CONTENT_SINK.get() is None


@pytest.mark.asyncio
async def test_content_retry_discards_partial_previous_attempt():
    from tempfile import TemporaryFile

    from litellm.videos.content_sink import VideoContentSink, stream_content_to_sink

    class Partial(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 65536
            raise httpx.ReadError("synthetic interrupted read")

    attempts = []

    async def provider(request):
        attempts.append(True)
        return httpx.Response(200, stream=Partial()) if len(attempts) == 1 else httpx.Response(200, content=b"complete")

    with TemporaryFile() as media:
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            sink = VideoContentSink(media)
            with pytest.raises(httpx.ReadError):
                await stream_content_to_sink(client, "https://synthetic.invalid/content", {}, sink)
            assert media.tell() == 65536
            await stream_content_to_sink(client, "https://synthetic.invalid/content", {}, sink)
            media.seek(0)
            assert media.read() == b"complete"
