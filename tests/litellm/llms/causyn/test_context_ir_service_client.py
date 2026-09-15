from __future__ import annotations

import httpx
import pytest

from litellm.llms.causyn import context_ir_client as client
from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError
from litellm.proxy.video_endpoints.minimax_h3_models import ImageItem, MediaURL, TextItem


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
