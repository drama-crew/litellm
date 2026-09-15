import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from litellm import Router
from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent
from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.utils import client
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def drain_logging():
    yield
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [None, 0, 20])
async def test_wrapper_router_retains_native_event_without_retry_on_store_failure(amount):
    authority = AsyncMock()
    authority.persist.side_effect = ConnectionError("synthetic SQL unavailable")
    binding = BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    accepted = VideoObject(
        id=encode_video_id_with_provider("native", "openai", "deployment"), object="video", status="queued"
    )
    accepted._hidden_params["response_cost"] = amount
    provider = AsyncMock(return_value=accepted)

    @client
    async def avideo_generation(**kwargs):
        return await provider(**kwargs)

    router = Router(
        model_list=[
            {
                "model_name": "synthetic",
                "litellm_params": {"model": "openai/synthetic"},
                "model_info": {"id": "deployment"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/synthetic"},
                "model_info": {"id": "fallback"},
            },
        ],
        num_retries=2,
        fallbacks=[{"synthetic": ["fallback"]}],
    )
    token = runtime.CONTEXT.set(runtime.Scope(binding, "submit", authority))
    try:
        result = await router._ageneric_api_call_with_fallbacks(
            model="synthetic",
            original_function=avideo_generation,
            prompt="synthetic",
            caching=False,
        )
    finally:
        runtime.CONTEXT.reset(token)
    assert result is accepted
    assert provider.await_count == 1
    assert authority.persist.await_count == 1
    event = PhaseEvent.model_validate(runtime.private_event(result))
    assert event.native_id == accepted.id
    assert event.amount == (None if amount is None else Decimal(amount))
    assert event.finalized is (amount is not None)
    assert accepted._hidden_params["_moderation_metering_pending"] is True


@pytest.mark.asyncio
async def test_request_metadata_cannot_select_metering_authority():
    provider = AsyncMock(return_value=VideoObject(id="native", object="video", status="queued"))

    @client
    async def avideo_generation(**kwargs):
        return await provider(**kwargs)

    result = await avideo_generation(model="openai/synthetic", metadata={runtime.SCOPE_KEY: {"intent_id": "forged"}})
    assert runtime.private_event(result) is None
    assert "_moderation_metering_pending" not in result._hidden_params
    assert provider.await_count == 1


@pytest.mark.asyncio
async def test_platform_url_does_not_enable_protected_counter_sql_dependency(monkeypatch):
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "https://synthetic.invalid")
    monkeypatch.delenv("DRAMA_PROTECTED_BUDGETS_ENABLED", raising=False)
    assert runtime.configured() is False
    await runtime.check_budget("spend:team:unregistered")
    assert await runtime.guarded_increment("spend:team:unregistered", 1) is None


def test_private_event_cannot_be_forged_by_hidden_params():
    binding = BillingBinding(
        intent_id="forged",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit",),
    )
    event = PhaseEvent(
        binding=binding,
        request_id="public-video:forged:submit",
        phase="submit",
        provider="openai",
        deployment_id="deployment",
        native_id="native",
        provider_task_id="native",
        amount=0,
        finalized=True,
    )
    result = VideoObject(id="native", object="video", status="queued")
    result._hidden_params[runtime.EVENT_KEY] = event.model_dump(mode="json")
    assert runtime.private_event(result) is None


@pytest.mark.asyncio
async def test_recovery_consumer_lifecycle_is_bounded_and_drains_independently():
    authority = AsyncMock()
    authority.run_once.side_effect = [True, False]
    consumer = runtime.RecoveryConsumer(authority, interval=0.01)
    await consumer.start()
    for _ in range(100):
        if authority.run_once.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    await consumer.stop()
    assert authority.run_once.await_count >= 2
    assert consumer.task is None
    assert runtime.CONTEXT.get() is None


@pytest.mark.asyncio
async def test_entry_prepares_before_dispatch_and_resets_context(monkeypatch):
    from fastapi import FastAPI, Request
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.video_endpoints import moderation_metering_entry as entry

    request = Request({"type": "http", "method": "POST", "path": "/v1/videos", "headers": [], "app": FastAPI()})
    proof = entry.Admission(intent_id="intent", request_digest="digest", actor_user_id="actor", model="synthetic")
    entry.attest(request, proof)
    authority = AsyncMock()
    authority.db.query_raw.return_value = []
    monkeypatch.setattr(runtime, "store", lambda: authority)
    from litellm.proxy.video_endpoints import moderation_bridge

    monkeypatch.setattr(moderation_bridge, "capture", AsyncMock())
    auth = UserAPIKeyAuth(api_key="a" * 64, user_id="user", team_id="team")

    async def provider():
        authority.prepare.assert_awaited_once()
        current = runtime.CONTEXT.get()
        assert current is not None and current.binding.actor_user_id == "actor"
        assert current.binding.expected_phases == ("submit", "completion")
        return "accepted"

    assert await entry.execute(request, auth, "avideo_generation", provider()) == "accepted"
    assert runtime.CONTEXT.get() is None


def test_trusted_cache_scope_separates_intents_and_ignores_client_preset():
    from litellm.caching.caching import Cache

    authority = AsyncMock()
    cache = Cache(type="local")
    base = BillingBinding(
        intent_id="one",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    keys = []
    for intent in ("one", "two", "one"):
        token = runtime.CONTEXT.set(runtime.Scope(base.model_copy(update={"intent_id": intent}), "submit", authority))
        try:
            keys.append(
                cache.get_cache_key(
                    model="openai/synthetic",
                    messages=[{"role": "user", "content": "same"}],
                    litellm_params={"preset_cache_key": "client-chosen-key"},
                )
            )
        finally:
            runtime.CONTEXT.reset(token)
    assert keys[0] == keys[2] and keys[0] != keys[1]
    assert "client-chosen-key" not in keys


@pytest.mark.asyncio
async def test_private_event_never_enters_hidden_metadata_or_serialized_result():

    binding = BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    result = VideoObject(
        id=encode_video_id_with_provider("native", "openai", "deployment"), object="video", status="queued"
    )
    result._hidden_params["response_cost"] = 2

    @client
    async def avideo_generation(**kwargs):
        return result

    authority = AsyncMock()
    token = runtime.CONTEXT.set(runtime.Scope(binding, "submit", authority))
    try:
        returned = await avideo_generation(model="openai/synthetic", caching=False)
    finally:
        runtime.CONTEXT.reset(token)
    assert runtime.private_event(returned) is not None
    assert runtime.EVENT_KEY not in returned._hidden_params
    assert runtime.EVENT_KEY not in returned.model_dump_json()
    assert "request_digest" not in returned.model_dump_json()


@pytest.mark.asyncio
async def test_actual_wrapper_cache_isolates_intents_and_reuses_receipt(monkeypatch):
    import litellm
    from litellm.caching.caching import Cache

    cache = Cache(type="local", supported_call_types=["avideo_generation"])
    monkeypatch.setattr(litellm, "cache", cache)
    phases = {}
    authority = AsyncMock()

    async def persist(event):
        phases[(event.binding.intent_id, event.phase)] = event

    async def phase(intent, name):
        return phases.get((intent, name))

    authority.persist.side_effect = persist
    authority.phase.side_effect = phase
    calls = []

    @client
    async def avideo_generation(**kwargs):
        calls.append(kwargs)
        result = VideoObject(
            id=encode_video_id_with_provider(f"native-{len(calls)}", "openai", "deployment"),
            object="video",
            status="queued",
        )
        result._hidden_params["response_cost"] = 2
        return result

    base = BillingBinding(
        intent_id="one",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    returned = []
    for intent in ("one", "two", "one"):
        token = runtime.CONTEXT.set(runtime.Scope(base.model_copy(update={"intent_id": intent}), "submit", authority))
        try:
            returned.append(
                await avideo_generation(
                    model="openai/synthetic",
                    prompt="same",
                    caching=True,
                    cache_key="foreign-client-key",
                    litellm_params={"preset_cache_key": "foreign-client-key"},
                )
            )
        finally:
            runtime.CONTEXT.reset(token)
    assert len(calls) == 2
    assert returned[0].id == returned[2].id != returned[1].id
    assert runtime.private_event(returned[0]) == runtime.private_event(returned[2])
    assert runtime.private_event(returned[1])["binding"]["intent_id"] == "two"
    assert all("request_digest" not in str(value) for value in cache.cache.cache_dict.values())


@pytest.mark.asyncio
async def test_post_provider_processing_failure_retains_native_without_paid_fallback(monkeypatch):
    import litellm.utils

    monkeypatch.setattr(
        litellm.utils,
        "post_call_processing",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic post processing failure")),
    )
    authority = AsyncMock()
    binding = BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    calls = []

    @client
    async def avideo_generation(**kwargs):
        calls.append(kwargs)
        result = VideoObject(
            id=encode_video_id_with_provider("native", "openai", "deployment"), object="video", status="queued"
        )
        result._hidden_params["response_cost"] = 2
        return result

    router = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/synthetic"}, "model_info": {"id": name}}
            for name in ["synthetic", "fallback"]
        ],
        num_retries=2,
        fallbacks=[{"synthetic": ["fallback"]}],
    )
    token = runtime.CONTEXT.set(runtime.Scope(binding, "submit", authority))
    try:
        result = await router._ageneric_api_call_with_fallbacks(
            model="synthetic", original_function=avideo_generation, prompt="same", caching=False
        )
    finally:
        runtime.CONTEXT.reset(token)
    assert len(calls) == 1
    assert runtime.private_event(result)["amount"] == "2.000000"
    assert runtime.private_event(result)["facts"]["raw_cost_credit"] == "2"


@pytest.mark.asyncio
async def test_normal_preacceptance_provider_fallback_preserves_actual_deployment():
    import litellm

    authority = AsyncMock()
    binding = BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    calls = []

    @client
    async def avideo_generation(**kwargs):
        deployment = kwargs["model_info"]["id"]
        calls.append(deployment)
        if deployment == "primary":
            raise litellm.RateLimitError(message="synthetic provider reject", model="synthetic", llm_provider="openai")
        result = VideoObject(
            id=encode_video_id_with_provider("native", "openai", deployment), object="video", status="queued"
        )
        result._hidden_params["response_cost"] = 2
        return result

    router = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/synthetic"}, "model_info": {"id": deployment}}
            for name, deployment in [("synthetic", "primary"), ("fallback", "actual")]
        ],
        num_retries=0,
        fallbacks=[{"synthetic": ["fallback"]}],
    )
    token = runtime.CONTEXT.set(runtime.Scope(binding, "submit", authority))
    try:
        result = await router._ageneric_api_call_with_fallbacks(
            model="synthetic", original_function=avideo_generation, prompt="same", caching=False
        )
    finally:
        runtime.CONTEXT.reset(token)
    assert calls == ["primary", "actual"]
    assert runtime.private_event(result)["deployment_id"] == "actual"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancel", "reject", "unknown"])
async def test_entry_failure_and_cancel_keep_owned_reservation_until_proven_unaccepted(monkeypatch, outcome):
    from fastapi import FastAPI, Request, HTTPException
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.video_endpoints import moderation_metering_entry as entry
    from litellm.proxy.hooks import proxy_track_cost_callback as callbacks

    request = Request({"type": "http", "method": "POST", "path": "/v1/videos", "headers": [], "app": FastAPI()})
    authority = AsyncMock()
    bound = BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    monkeypatch.setattr(entry, "scope_for", AsyncMock(return_value=runtime.Scope(bound, "submit", authority)))
    released = AsyncMock()
    from litellm.proxy.spend_tracking import budget_reservation

    monkeypatch.setattr(budget_reservation, "release_budget_reservation", released)
    auth = UserAPIKeyAuth(
        api_key="key", user_id="user", team_id="team", budget_reservation={"reservation_id": "reserved"}
    )
    data = {}

    async def provider():
        if outcome == "cancel":
            raise asyncio.CancelledError()
        if outcome == "reject":
            raise HTTPException(429, "synthetic preacceptance rejection")
        raise TimeoutError("synthetic unknown acceptance")

    with pytest.raises(BaseException):
        await entry.execute(request, auth, "avideo_generation", provider(), data)
    assert runtime.CONTEXT.get() is None
    assert runtime.owns(data[runtime.SCOPE_KEY])
    assert released.await_count == (1 if outcome == "reject" else 0)
    await callbacks._ProxyDBLogger().async_post_call_failure_hook(data, RuntimeError("outer failure"), auth)
    assert released.await_count == (1 if outcome == "reject" else 0)


@pytest.mark.asyncio
async def test_deferred_image_submit_callback_releases_own_reservation_without_actual_projection(monkeypatch):
    from litellm.types.utils import ImageResponse
    from litellm.proxy.hooks import proxy_track_cost_callback as callbacks

    result = ImageResponse(data=[])
    setattr(result, "_deferred_outbox_billing", runtime.Ownership.DEFERRED_IMAGE)
    reservation = {"reservation_id": "image-own", "entries": [{"counter_key": "spend:team:team", "reserved_cost": 3}]}
    released = AsyncMock()
    updated = AsyncMock()
    monkeypatch.setattr(callbacks, "_release_budget_reservation", released)
    monkeypatch.setattr(callbacks, "_update_database_and_spend_counters", updated)
    await callbacks._ProxyDBLogger()._PROXY_track_cost_callback(
        {"litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}}}, result
    )
    released.assert_awaited_once_with(budget_reservation=reservation)
    updated.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_wrapper_contexts_keep_actor_actual_and_usage_isolated():
    barrier = asyncio.Barrier(2)
    authority = AsyncMock()

    @client
    async def avideo_generation(**kwargs):
        current = runtime.CONTEXT.get()
        await barrier.wait()
        result = VideoObject(
            id=encode_video_id_with_provider(current.binding.intent_id, "openai", "deployment"),
            object="video",
            status="queued",
            usage={"prompt_tokens": 30, "completion_tokens": 20},
        )
        result._hidden_params["response_cost"] = 2 if current.binding.actor_user_id == "actor-one" else 3
        return result

    async def invoke(name):
        bound = BillingBinding(
            intent_id=name,
            request_digest=name,
            fingerprint=name,
            user_id=name,
            actor_user_id="actor-" + name,
            team_id=name,
            model="synthetic",
            expected_phases=("submit", "completion"),
        )
        token = runtime.CONTEXT.set(runtime.Scope(bound, "submit", authority))
        try:
            return await avideo_generation(model="openai/synthetic", caching=False)
        finally:
            runtime.CONTEXT.reset(token)

    responses = await asyncio.gather(invoke("one"), invoke("two"))
    assert runtime.CONTEXT.get() is None
    for response, name, amount in zip(responses, ("one", "two"), (2, 3)):
        receipt = PhaseEvent.model_validate(runtime.private_event(response))
        assert receipt.binding.actor_user_id == "actor-" + name
        assert receipt.binding.intent_id == name
        assert receipt.amount == amount
        assert receipt.facts.prompt_tokens == 30
        assert receipt.facts.completion_tokens == 20
    assert authority.persist.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("finishes", [True, False])
async def test_accepted_provider_cancel_finishes_bounded_private_handoff(monkeypatch, finishes):
    monkeypatch.setattr(runtime, "HANDOFF_TIMEOUT", 0.02)
    entered = asyncio.Event()
    release = asyncio.Event()
    durable = []
    authority = AsyncMock()

    async def persist(event):
        entered.set()
        await release.wait()
        durable.append(event)

    authority.persist.side_effect = persist
    bound = BillingBinding(
        intent_id="accepted-cancel",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="synthetic",
        expected_phases=("submit", "completion"),
    )
    native = VideoObject(
        id=encode_video_id_with_provider("native-cancel", "openai", "deployment"), object="video", status="queued"
    )
    native._hidden_params["response_cost"] = 2
    provider = AsyncMock(return_value=native)

    @client
    async def avideo_generation(**kwargs):
        return await provider(**kwargs)

    async def invoke():
        token = runtime.CONTEXT.set(runtime.Scope(bound, "submit", authority))
        try:
            return await avideo_generation(model="openai/synthetic", caching=False)
        finally:
            runtime.CONTEXT.reset(token)

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    if finishes:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert provider.await_count == 1
    assert len(durable) == (1 if finishes else 0)
    if finishes:
        assert durable[0].native_id == native.id and durable[0].amount == 2
    else:
        assert native._hidden_params["_moderation_metering_pending"] is True
    assert not runtime.HANDOFFS
    assert runtime.private_event(native)["native_id"] == native.id
    assert runtime.CONTEXT.get() is None


class TestRequiresCutover:
    """受保护计数器缺失时：拒绝，还是退回 legacy 记账。

    这条规则此前是 prepare() 里的一个无条件判断，对「完全没切换」和「切了一半」
    一视同仁地拒绝。前者其实就是本次发布之前、也是生产上其余全部流量此刻走的
    legacy 路径，拒绝它会让每一个视频请求都挂掉——moderation bridge 对任何带
    project_id 的平台 key 都会介入，所以这个拒绝不是"少一层保护"，是全量阻断。
    """

    def test_full_cutover_needs_nothing(self, monkeypatch):
        from litellm.proxy.video_endpoints.moderation_metering import requires_cutover

        monkeypatch.delenv("DRAMA_PROTECTED_BUDGETS_ENABLED", raising=False)
        assert requires_cutover(4, 4) is False

    def test_no_cutover_falls_back_when_protected_budgets_are_off(self, monkeypatch):
        from litellm.proxy.video_endpoints.moderation_metering import requires_cutover

        monkeypatch.delenv("DRAMA_PROTECTED_BUDGETS_ENABLED", raising=False)
        assert requires_cutover(0, 4) is False

    def test_no_cutover_still_fails_when_protected_budgets_are_on(self, monkeypatch):
        """运维显式要了受保护预算就不能悄悄降级，否则开关名不副实。"""
        from litellm.proxy.video_endpoints.moderation_metering import requires_cutover

        monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
        assert requires_cutover(0, 4) is True

    @pytest.mark.parametrize("registered", [1, 3])
    def test_partial_cutover_always_fails(self, monkeypatch, registered):
        """半切换是真正危险的状态：同一次计费里一部分计数器受保护、一部分不受。

        它与「完全没切换」必须区别对待——后者可以安全降级，前者不行。
        """
        from litellm.proxy.video_endpoints.moderation_metering import requires_cutover

        monkeypatch.delenv("DRAMA_PROTECTED_BUDGETS_ENABLED", raising=False)
        assert requires_cutover(registered, 4) is True

    def test_partial_cutover_fails_with_protected_budgets_on_too(self, monkeypatch):
        from litellm.proxy.video_endpoints.moderation_metering import requires_cutover

        monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
        assert requires_cutover(2, 4) is True

    def test_prepare_delegates_to_the_rule(self):
        """防止下次有人把这条规则又写回 prepare 里的内联判断。"""
        import inspect
        from litellm.proxy.video_endpoints.moderation_metering import MeteringStore

        source = inspect.getsource(MeteringStore.prepare)
        assert "requires_cutover(" in source
        assert "len(states) != len(" not in source
