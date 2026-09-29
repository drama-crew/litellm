from __future__ import annotations

import json

import httpx
import pytest

from litellm.llms.causyn import context_ir_client as client
from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError
from litellm.proxy.video_endpoints.minimax_h3_models import AudioItem, ImageItem, MediaURL, TextItem, VideoItem


def _spec(n_refs: int = 1) -> ContextIRRequest:
    return ContextIRRequest(
        model="causyn-1.1",
        duration=5,
        ratio="16:9",
        content=(
            TextItem(type="text", text="a woman in a field"),
            *(
                ImageItem(
                    type="image_url",
                    image_url=MediaURL(url=f"https://source.example/r{i}.png"),
                    role="reference_image",
                )
                for i in range(n_refs)
            ),
        ),
    )


def _spec_with_video_reference() -> ContextIRRequest:
    return ContextIRRequest(
        model="causyn-1.1",
        duration=5,
        ratio="16:9",
        content=(
            TextItem(type="text", text="a woman in a field"),
            VideoItem(type="video_url", video_url=MediaURL(url="https://source.example/r1.mp4")),
        ),
    )


def _spec_with_audio_reference() -> ContextIRRequest:
    return ContextIRRequest(
        model="causyn-1.1",
        duration=5,
        ratio="16:9",
        content=(
            TextItem(type="text", text="a woman in a field"),
            VideoItem(type="video_url", video_url=MediaURL(url="https://source.example/r1.mp4")),
            AudioItem(type="audio_url", audio_url=MediaURL(url="https://source.example/r1.mp3")),
        ),
    )


def _transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)


class TestRouting:
    def test_ref2va_is_routed_to_the_service(self):
        assert client.should_use_service(_spec(1)) is True

    @pytest.mark.parametrize("mode_spec", ["t2va", "i2va", "fl2va"])
    def test_other_modes_stay_on_the_single_shot_path(self, mode_spec):
        """服务只实现了 Ref2VA 一种；其余四种模式的能力必须原样保留。"""
        content = [TextItem(type="text", text="p")]
        if mode_spec in {"i2va", "fl2va"}:
            content.append(
                ImageItem(type="image_url", image_url=MediaURL(url="https://s.example/a.png"), role="first_frame")
            )
        if mode_spec == "fl2va":
            content.append(
                ImageItem(type="image_url", image_url=MediaURL(url="https://s.example/b.png"), role="last_frame")
            )
        spec = ContextIRRequest(
            model="causyn-1.1", duration=5, ratio="16:9" if mode_spec == "t2va" else "adaptive", content=tuple(content)
        )
        assert client.should_use_service(spec) is False

    def test_ref2va_with_a_video_reference_stays_on_the_single_shot_path(self):
        """服务的提交形状(ContextIRSubmission)只有 images 字段；带视频/音频参考的
        ref2va 请求如果被当作 ref2va 路由过去，_payload 会在 item.image_url 上
        抛 AttributeError,而不是把任务体面地降级。"""
        assert client.should_use_service(_spec_with_video_reference()) is False

    def test_ref2va_with_an_audio_reference_stays_on_the_single_shot_path(self):
        assert client.should_use_service(_spec_with_audio_reference()) is False


class TestSubmission:
    @pytest.mark.asyncio
    async def test_prompt_and_geometry_reach_the_service(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"id": "cir-1", "status": "queued"})

        async with _transport(handler) as http:
            await client.submit(
                _spec(2), base_url="http://ctx:8030", api_key=None, idempotency_key="video-42", http=http
            )
        assert seen["prompt"] == "a woman in a field"
        assert seen["duration_s"] == 5
        assert seen["ratio"] == "16:9"
        assert len(seen["images"]) == 2
        assert seen["idempotency_key"] == "video-42"

    @pytest.mark.asyncio
    async def test_bearer_token_is_sent_when_configured(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"id": "cir-1", "status": "queued"})

        async with _transport(handler) as http:
            await client.submit(_spec(), base_url="http://ctx:8030", api_key="k", idempotency_key="v", http=http)
        assert seen["auth"] == "Bearer k"

    @pytest.mark.asyncio
    async def test_service_rejection_surfaces_as_a_rewrite_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"error": "ratio must be one of ..."})

        async with _transport(handler) as http:
            with pytest.raises(RewriteError) as caught:
                await client.submit(_spec(), base_url="http://ctx:8030", api_key=None, idempotency_key="v", http=http)
        assert caught.value.status_code == 422

    @pytest.mark.asyncio
    async def test_transport_failure_is_retryable(self):
        """服务短暂不可达不该让一条已收费的视频任务永久失败。"""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        async with _transport(handler) as http:
            with pytest.raises(RewriteError) as caught:
                await client.submit(_spec(), base_url="http://ctx:8030", api_key=None, idempotency_key="v", http=http)
        assert caught.value.retryable is True


class TestPolling:
    @pytest.mark.asyncio
    async def test_succeeded_task_returns_its_caption(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"id": "cir-1", "status": "succeeded", "caption": "subject_definitions: <Subject 1> ..."}
            )

        async with _transport(handler) as http:
            got = await client.poll("cir-1", base_url="http://ctx:8030", api_key=None, http=http)
        assert got.status == "succeeded"
        assert got.caption is not None and got.caption.startswith("subject_definitions:")

    @pytest.mark.asyncio
    async def test_failed_task_reports_the_step_that_died(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"id": "cir-1", "status": "failed", "failed_step": "assets", "error": "upstream 503"}
            )

        async with _transport(handler) as http:
            got = await client.poll("cir-1", base_url="http://ctx:8030", api_key=None, http=http)
        assert got.failed_step == "assets"


class TestConfiguration:
    def test_service_is_off_unless_a_base_url_is_configured(self, monkeypatch):
        """没配地址就不该悄悄走新路径——默认必须是现有的单次改写。"""
        monkeypatch.delenv("DRAMA_CONTEXT_IR_BASE_URL", raising=False)
        assert client.service_base_url() is None

    def test_configured_base_url_is_normalised(self, monkeypatch):
        monkeypatch.setenv("DRAMA_CONTEXT_IR_BASE_URL", "http://drama-context-ir:8030/")
        assert client.service_base_url() == "http://drama-context-ir:8030"


class TestRewriteAdapter:
    """把服务的 submit/poll 适配成 ContextIRService 要的单次 rewrite。

    调用点有 155s 单次超时，而九张参考图的改写可能跑 3-5 分钟。所以适配器不等到底：
    在预算内轮询，没完成就抛 retryable，让既有重试机制充当轮询循环。幂等键保证
    重入时命中同一个任务，而不是把 81 次模型调用重跑一遍。
    """

    @pytest.mark.asyncio
    async def test_completed_task_returns_the_caption_as_a_rewrite_result(self):
        caption = "subject_definitions: <Subject 1> a woman.\n\nsummary: [reference generation] x"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"id": "cir-9", "status": "queued"})
            return httpx.Response(200, json={"id": "cir-9", "status": "succeeded", "caption": caption})

        async with _transport(handler) as http:
            got = await client.rewrite_via_service(
                _spec(),
                base_url="http://ctx:8030",
                api_key=None,
                idempotency_key="v-1",
                http=http,
                poll_interval_s=0,
                budget_s=5,
            )
        assert got.prompt == caption
        assert got.model == client.SERVICE_MODEL

    @pytest.mark.asyncio
    async def test_unfinished_task_raises_retryable_so_the_caller_polls_again(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"id": "cir-9", "status": "queued"})
            return httpx.Response(200, json={"id": "cir-9", "status": "queued", "completed_steps": ["observations"]})

        async with _transport(handler) as http:
            with pytest.raises(RewriteError) as caught:
                await client.rewrite_via_service(
                    _spec(),
                    base_url="http://ctx:8030",
                    api_key=None,
                    idempotency_key="v-2",
                    http=http,
                    poll_interval_s=0,
                    budget_s=0.05,
                )
        assert caught.value.retryable is True

    @pytest.mark.asyncio
    async def test_failed_task_is_not_retried_forever(self):
        """服务已判定失败就是终态；继续重试只会把同一个失败重放三遍。"""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"id": "cir-9", "status": "queued"})
            return httpx.Response(
                200, json={"id": "cir-9", "status": "failed", "failed_step": "plan", "error": "no caption"}
            )

        async with _transport(handler) as http:
            with pytest.raises(RewriteError) as caught:
                await client.rewrite_via_service(
                    _spec(),
                    base_url="http://ctx:8030",
                    api_key=None,
                    idempotency_key="v-3",
                    http=http,
                    poll_interval_s=0,
                    budget_s=5,
                )
        assert caught.value.retryable is False
        assert "plan" in str(caught.value)

    @pytest.mark.asyncio
    async def test_resubmission_of_the_same_video_reuses_the_task(self):
        """重入时不能开新任务——否则每次重试都重跑整条链。"""
        posts = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                import json

                posts.append(json.loads(request.content)["idempotency_key"])
                return httpx.Response(200, json={"id": "cir-9", "status": "queued"})
            return httpx.Response(200, json={"id": "cir-9", "status": "succeeded", "caption": "subject_definitions: x"})

        async with _transport(handler) as http:
            for _ in range(2):
                await client.rewrite_via_service(
                    _spec(),
                    base_url="http://ctx:8030",
                    api_key=None,
                    idempotency_key="video-same",
                    http=http,
                    poll_interval_s=0,
                    budget_s=5,
                )
        assert posts == ["video-same", "video-same"]


class TestRouting_EndToEnd:
    """rewrite_prompt 的分流：只有 ref2va 且服务已配置时才走新路径。"""

    @pytest.mark.asyncio
    async def test_ref2va_without_configuration_stays_on_the_single_shot_path(self, monkeypatch):
        """没配服务地址就必须保持原行为——部署服务本身不该改变任何请求的走向。"""
        monkeypatch.delenv("DRAMA_CONTEXT_IR_BASE_URL", raising=False)
        called = {}

        async def fake_single_shot(spec):
            called["single_shot"] = True
            from litellm.llms.causyn.h3_prompt import RewriteResult, RewriteUsage

            return RewriteResult(prompt="legacy", usage=RewriteUsage(), system_sha256="a" * 64)

        from litellm.llms.causyn import h3_prompt

        monkeypatch.setattr(h3_prompt, "_rewrite_single_shot", fake_single_shot)
        got = await h3_prompt.rewrite_prompt(_spec())
        assert called == {"single_shot": True}
        assert got.prompt == "legacy"

    @pytest.mark.asyncio
    async def test_non_ref2va_never_uses_the_service_even_when_configured(self, monkeypatch):
        """服务只实现 Ref2VA；其余四种模式配置了也不能被误导过去。"""
        monkeypatch.setenv("DRAMA_CONTEXT_IR_BASE_URL", "http://ctx:8030")
        called = {}

        async def fake_single_shot(spec):
            called["single_shot"] = True
            from litellm.llms.causyn.h3_prompt import RewriteResult, RewriteUsage

            return RewriteResult(prompt="legacy", usage=RewriteUsage(), system_sha256="a" * 64)

        from litellm.llms.causyn import h3_prompt

        monkeypatch.setattr(h3_prompt, "_rewrite_single_shot", fake_single_shot)
        t2va = ContextIRRequest(
            model="causyn-1.1", duration=5, ratio="16:9", content=(TextItem(type="text", text="a kite"),)
        )
        await h3_prompt.rewrite_prompt(t2va)
        assert called == {"single_shot": True}

    def test_idempotency_key_is_stable_for_the_same_request(self):
        """同一个视频任务重试时必须命中同一个改写任务，否则整条链重跑。"""
        assert client.idempotency_key_for(_spec(2)) == client.idempotency_key_for(_spec(2))

    def test_idempotency_key_differs_for_different_requests(self):
        assert client.idempotency_key_for(_spec(1)) != client.idempotency_key_for(_spec(2))

    @pytest.mark.asyncio
    async def test_ref2va_with_a_video_reference_never_uses_the_service_even_when_configured(self, monkeypatch):
        monkeypatch.setenv("DRAMA_CONTEXT_IR_BASE_URL", "http://ctx:8030")
        called = {}

        async def fake_single_shot(spec):
            called["single_shot"] = True
            from litellm.llms.causyn.h3_prompt import RewriteResult, RewriteUsage

            return RewriteResult(prompt="legacy", usage=RewriteUsage(), system_sha256="a" * 64)

        from litellm.llms.causyn import h3_prompt

        monkeypatch.setattr(h3_prompt, "_rewrite_single_shot", fake_single_shot)
        await h3_prompt.rewrite_prompt(_spec_with_video_reference())
        assert called == {"single_shot": True}


class TestPayloadRobustnessAgainstNonImageReferences:
    """_payload 和 idempotency_key_for 曾经无条件对每个媒体条目取 .image_url.url;
    一旦 ordered_media 里混进 VideoItem/AudioItem(例如被直接调用,或是路由逻辑
    以后又出现新的疏漏),就会在这两个字段访问上抛 AttributeError 而不是一个
    可控的错误。"""

    def test_payload_does_not_crash_building_a_video_reference_submission(self):
        submission = client._payload(_spec_with_video_reference(), "key-1")
        assert len(submission.images) == 1
        assert submission.images[0].url == "https://source.example/r1.mp4"

    def test_payload_does_not_crash_building_a_mixed_video_and_audio_submission(self):
        submission = client._payload(_spec_with_audio_reference(), "key-1")
        urls = {image.url for image in submission.images}
        assert urls == {"https://source.example/r1.mp4", "https://source.example/r1.mp3"}

    def test_idempotency_key_does_not_crash_on_a_video_reference(self):
        assert client.idempotency_key_for(_spec_with_video_reference())

    def test_idempotency_key_does_not_crash_on_an_audio_reference(self):
        assert client.idempotency_key_for(_spec_with_audio_reference())

    def test_idempotency_key_differs_between_a_video_and_an_audio_reference_request(self):
        assert client.idempotency_key_for(_spec_with_video_reference()) != client.idempotency_key_for(
            _spec_with_audio_reference()
        )


class TestTransientFailureResubmission:
    @staticmethod
    def _service(failure: dict, *, fail_first: int = 1):
        posts: list[str] = []
        failed_keys: set[str] = set()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                key = json.loads(request.content)["idempotency_key"]
                posts.append(key)
                if key in failed_keys:
                    return httpx.Response(200, json={"id": f"cir-{key}", "status": "failed", **failure})
                return httpx.Response(200, json={"id": f"cir-{key}", "status": "queued"})
            task_id = str(request.url).rsplit("/", 1)[-1]
            key = task_id.removeprefix("cir-")
            if len(failed_keys) < fail_first:
                failed_keys.add(key)
            if key in failed_keys:
                return httpx.Response(200, json={"id": task_id, "status": "failed", **failure})
            return httpx.Response(200, json={"id": task_id, "status": "succeeded", "caption": "ok"})

        return handler, posts

    async def _run(self, handler):
        async with _transport(handler) as http:
            return await client.rewrite_via_service(
                _spec(),
                base_url="http://ctx:8030",
                api_key=None,
                idempotency_key="v",
                http=http,
                poll_interval_s=0,
                budget_s=5,
            )

    @pytest.mark.asyncio
    async def test_transient_failure_raises_retryable_then_reentry_uses_a_fresh_key(self):
        handler, posts = self._service(
            {"failed_step": "observations", "retryable": True, "error_kind": "upstream_transient"}
        )
        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.retryable is True
        got = await self._run(handler)  # outer retry re-enters with the same base key
        assert got.prompt == "ok"
        assert posts == ["v", "v", "v:retry-1"]

    @pytest.mark.asyncio
    async def test_error_kind_alone_marks_the_failure_transient(self):
        handler, _ = self._service({"failed_step": "x", "error_kind": "upstream_transient"})
        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.retryable is True

    @pytest.mark.asyncio
    async def test_legacy_failure_message_with_429_is_treated_as_transient(self):
        handler, _ = self._service({"failed_step": "observations", "error": "OpenRouterError: HTTP 429: rate-limited"})
        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.retryable is True
        assert "429" in caught.value.describe()

    @pytest.mark.asyncio
    async def test_legacy_text_without_an_http_status_stays_terminal(self):
        handler, _ = self._service({"failed_step": "observations", "error": "the model was temporarily rate limited"})
        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    async def test_upstream_status_from_the_service_is_recorded(self):
        handler, _ = self._service(
            {
                "failed_step": "observations",
                "retryable": True,
                "error_kind": "upstream_transient",
                "upstream_status": 429,
            }
        )
        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.upstream_status == 429
        assert "upstream 429" in caught.value.describe()

    @pytest.mark.asyncio
    async def test_explicit_non_retryable_stays_terminal(self):
        handler, _ = self._service({"failed_step": "plan", "retryable": False, "error": "HTTP 429 but service says no"})
        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    async def test_opaque_legacy_failure_stays_terminal(self):
        handler, _ = self._service({"failed_step": "observations"})
        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    async def test_resubmission_chain_is_bounded(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "cir-x", "status": "failed", "failed_step": "s", "retryable": True})

        with pytest.raises(RewriteError) as caught:
            await self._run(handler)
        assert caught.value.retryable is False
