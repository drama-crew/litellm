from __future__ import annotations

import asyncio
import json
from collections import Counter

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI, Request

from litellm.llms.causyn.context_ir import ContextIRService
from litellm.llms.causyn.context_ir_budget import REFUND_SCRIPT
from litellm.llms.causyn.context_ir_store import BillingIdentity, ContextIRStore, task_key
from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError, RewriteResult, RewriteUsage, validate_prompt
from litellm.llms.libtv.billing_outbox import CAUSYN_BILLING_STREAM_KEY, CausynBillingEvent, enqueue_causyn_billing
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.spend_tracking.budget_reservation import estimate_request_max_cost
from litellm.proxy.video_endpoints import context_ir_endpoints as ir
from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3

PROMPT = (
    "integrated_multimodal_description: [Shot 1] A cat walks.\noverall_soundscape: Quiet.\nnon_diegetic_music: None."
)
RESULT = RewriteResult(
    prompt=PROMPT,
    usage=RewriteUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50, cost=0.001),
    system_sha256="a" * 64,
)


def spec(**kwargs):
    return ContextIRRequest.model_validate(
        {
            "model": "MiniMax-H3",
            "content": [{"type": "text", "text": "A cat walks."}],
            "duration": 5,
            "ratio": "16:9",
            **kwargs,
        }
    )


async def rewrite(request):
    return RESULT


async def no_settle(task):
    return None


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_result_reused_after_settlement_failure_and_duplicate_delivery(redis):
    calls = Counter()

    async def counted_rewrite(request):
        calls["rewrite"] += 1
        return RESULT

    async def settle(task):
        calls["settle"] += 1
        await enqueue_causyn_billing(
            redis,
            CausynBillingEvent(
                provider_task_id=task.id, response_cost=4, model="causyn-h3-context-ir", task_type="h3_context_ir"
            ),
        )
        if calls["settle"] == 1:
            raise ConnectionError("lost settlement acknowledgement")

    service = ContextIRService(ContextIRStore(redis), rewrite=counted_rewrite, settle=settle)
    task = await service.create(spec(), owner="owner", billing=BillingIdentity())
    await service.process(task.id)
    persisted = await service.store.get(task.id)
    assert persisted.result == RESULT
    assert persisted.status == "running"
    restarted = ContextIRService(ContextIRStore(redis), rewrite=counted_rewrite, settle=settle)
    await asyncio.gather(restarted.process(task.id), restarted.process(task.id))
    ready = await restarted.store.get(task.id)
    assert ready.status == "succeeded"
    assert calls["rewrite"] == 1
    assert await redis.xlen(CAUSYN_BILLING_STREAM_KEY) == 1
    payload = json.loads((await redis.xrange(CAUSYN_BILLING_STREAM_KEY))[0][1][b"payload"])
    assert payload["response_cost"] == 4
    assert payload["task_type"] == "h3_context_ir"
    assert "cost" not in ready.public()["usage"]


@pytest.mark.asyncio
async def test_invalid_output_failed_and_not_billable(redis):
    charged = []

    async def invalid(request):
        validate_prompt("raw prose", request)
        return RESULT

    async def settle(task):
        charged.append(task.result is not None and task.status not in {"failed", "cancelled"})

    service = ContextIRService(ContextIRStore(redis), rewrite=invalid, settle=settle)
    task = await service.create(spec(), owner="owner", billing=BillingIdentity())
    await service.process(task.id)
    failed = await service.store.get(task.id)
    assert failed.status == "failed"
    assert failed.settled
    assert charged == [False]
    assert "content" not in failed.public()


@pytest.mark.asyncio
async def test_cancel_owner_running_and_terminal_rules(redis):
    calls = Counter()

    async def counted(request):
        calls["rewrite"] += 1
        return RESULT

    service = ContextIRService(ContextIRStore(redis), rewrite=counted, settle=no_settle)
    task = await service.create(spec(), owner="owner", billing=BillingIdentity())
    with pytest.raises(RewriteError, match="not found"):
        await service.cancel_or_delete(task.id, "other")
    assert await service.cancel_or_delete(task.id, "owner") == "cancelled"
    await service.process(task.id)
    assert calls["rewrite"] == 0
    with pytest.raises(RewriteError, match="cannot be cancelled"):
        await service.cancel_or_delete(task.id, "owner")
    second = await service.create(spec(), owner="owner", billing=BillingIdentity())
    await service.process(second.id)
    assert await service.cancel_or_delete(second.id, "owner") == "deleted"
    assert await service.store.get(second.id) is None


@pytest.mark.asyncio
async def test_callback_statuses_are_ordered_and_terminal_retry_does_not_rewrite(redis):
    statuses = []
    calls = Counter()

    async def notify(url, body):
        status = body["task"]["status"]
        statuses.append(status)
        calls[status] += 1
        return status != "succeeded" or calls[status] > 1

    async def counted(request):
        calls["rewrite"] += 1
        return RESULT

    service = ContextIRService(ContextIRStore(redis), rewrite=counted, settle=no_settle, notify=notify)
    task = await service.create(spec(callback_url="https://callback.example"), owner="owner", billing=BillingIdentity())
    await service.process(task.id)
    assert (await service.store.get(task.id)).status == "succeeded"
    await service.process(task.id)
    assert statuses == ["queued", "running", "succeeded", "succeeded"]
    assert calls["rewrite"] == 1
    assert not (await service.store.get(task.id)).notifications


@pytest.mark.asyncio
async def test_rate_limit_retry_bound(redis):
    calls = Counter()

    async def limited(request):
        calls["rewrite"] += 1
        raise RewriteError("rate limited", 429)

    service = ContextIRService(ContextIRStore(redis), rewrite=limited, settle=no_settle)
    task = await service.create(spec(), owner="owner", billing=BillingIdentity())
    for _ in range(4):
        await service.process(task.id)
    assert calls["rewrite"] == 3
    assert (await service.store.get(task.id)).status == "failed"


@pytest.mark.asyncio
async def test_expired_lease_recovers_and_fences_old_worker(redis):
    store = ContextIRStore(redis)
    task = await ContextIRService(store, settle=no_settle).create(spec(), owner="owner", billing=BillingIdentity())
    assert await store.claim(task.id, "old")
    await redis.delete(task_key(task.id) + ":lease")
    assert await store.claim(task.id, "new")
    with pytest.raises(RewriteError, match="lease was lost"):
        await store.save(task.model_copy(update={"status": "succeeded"}), "old", done=True)
    assert (await store.get(task.id)).status == "queued"


@pytest.mark.asyncio
async def test_budget_refund_is_atomic_and_idempotent(redis):
    await redis.set("counter", "12")
    await asyncio.gather(*(redis.eval(REFUND_SCRIPT, 2, "counter", "refund", -4) for _ in range(10)))
    assert float(await redis.get("counter")) == 8
    await redis.set("missing-marker-counter", "2")
    await redis.eval(REFUND_SCRIPT, 2, "missing-marker-counter", "refund2", -4)
    assert float(await redis.get("missing-marker-counter")) == 2


def test_fixed_price_and_poll_has_no_reservation():
    assert estimate_request_max_cost({"model": "causyn-h3-context-ir", "price": 0}, "/v2/h3_context_ir", None) == 4
    assert estimate_request_max_cost({"model": "causyn-h3-context-ir"}, "/v2/query/video_generation/x", None) is None


@pytest.mark.asyncio
async def test_automatic_rewrite_is_included_while_standalone_costs_four(redis, monkeypatch):
    import time
    import litellm.llms.causyn.context_ir as context_ir
    from litellm.llms.causyn.video_prompt import VideoSubmission, submit_video_prompt

    service = ContextIRService(ContextIRStore(redis), rewrite=rewrite, deliver=no_settle)
    monkeypatch.setattr(context_ir, "get_context_ir_service", lambda: service)
    monkeypatch.setattr(context_ir, "get_transfer_redis", lambda url: redis)
    video_id = "0123456789abcdef0123456789abcdef"
    await submit_video_prompt(VideoSubmission(
        task_id=video_id, model="causyn-1.1", deadline_ts=time.time() + 300,
        request={"prompt": "A cat walks.", "duration_seconds": 5, "ratio": "16:9"},
        task_metadata={},
    ), BillingIdentity())
    await service.process("h3_ir_" + video_id)
    automatic = await service.store.get("h3_ir_" + video_id)
    assert automatic.status == "succeeded"
    assert automatic.price == 0
    assert automatic.settled
    assert await redis.xlen(CAUSYN_BILLING_STREAM_KEY) == 0

    standalone = await service.create(spec(), owner="owner", billing=BillingIdentity())
    await service.process(standalone.id)
    assert (await service.store.get(standalone.id)).price == 4
    assert await redis.xlen(CAUSYN_BILLING_STREAM_KEY) == 1


def test_mixed_media_indexes_and_audio_rejection():
    request = spec(
        content=[
            {"type": "text", "text": "Follow the actor from the picture and movement in the video."},
            {"type": "image_url", "image_url": {"url": "https://media.example/a.png"}, "role": "reference_image"},
            {"type": "video_url", "video_url": {"url": "https://media.example/a.mp4"}},
        ]
    )
    assert request.mode == "ref2va"
    assert request.user_content()[0]["text"].startswith("<Picture 1>")
    assert request.user_content()[2]["text"].startswith("<Video 1>")
    audio = spec(
        content=[
            {"type": "text", "text": "Hello"},
            {"type": "audio_url", "audio_url": {"url": "https://media.example/a.mp3"}},
        ]
    )
    with pytest.raises(RewriteError) as failure:
        audio.require_supported()
    assert failure.value.status_code == 422


@pytest.mark.asyncio
async def test_public_protocol_normalization_query_list_delete_and_audio(redis):
    service = ContextIRService(ContextIRStore(redis), rewrite=rewrite, settle=no_settle)
    normalized = []

    async def auth(request: Request):
        normalized.append(await _read_request_body(request))
        return UserAPIKeyAuth(api_key=request.headers.get("authorization", "owner"))

    async def dependency():
        return service

    app = FastAPI()
    app.dependency_overrides[h3.user_api_key_auth] = auth
    app.dependency_overrides[ir.context_ir_service] = dependency
    app.dependency_overrides[h3.context_ir_service_for_request] = dependency
    app.include_router(h3.router)
    app.include_router(ir.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/v2/h3_context_ir", json=spec().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        task_id = created.json()["task_id"]
        assert normalized[0] == {"model": "causyn-1.1"}
        await service.process(task_id)
        query = await client.get("/v2/query/video_generation/" + task_id)
        assert query.status_code == 200, query.text
        assert query.json()["task"]["content"]["prompt"] == PROMPT
        other = await client.get("/v2/query/video_generation/" + task_id, headers={"Authorization": "another-owner"})
        assert other.status_code == 404
        listing = await client.get(
            "/v2/query/video_generation", params={"filter.task_type": "h3_context_ir", "filter.status": "succeeded"}
        )
        assert listing.json()["total"] == 1
        deleted = await client.delete("/v2/video_generation/" + task_id)
        assert deleted.json() == {"task_id": task_id, "action": "deleted", "status": "deleted"}
        audio = spec(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "audio_url", "audio_url": {"url": "https://media.example/a.mp3"}},
            ]
        )
        count = len(normalized)
        rejected = await client.post("/v2/h3_context_ir", json=audio.model_dump(mode="json"))
        assert rejected.status_code == 422
        assert len(normalized) == count


@pytest.mark.asyncio
async def test_actual_auth_enforces_causyn_permission_for_all_ir_routes(redis, monkeypatch):
    import litellm
    import litellm.proxy.proxy_server as proxy
    from litellm import Router
    from litellm.llms.causyn.handler import CausynVideoHandler
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from tests.test_litellm.proxy.video_endpoints.test_causyn_public_stack import (
        _AuthStore,
        _ProxyConfig,
        _ProxyLogging,
    )

    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "causyn", "custom_handler": CausynVideoHandler()}]
    )
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    litellm.utils.custom_llm_setup()
    router = Router(
        model_list=[
            {
                "model_name": "causyn-1.1",
                "litellm_params": {"model": "causyn/causyn-1.1"},
                "model_info": {"id": "causyn-1-1"},
            }
        ],
        num_retries=0,
    )
    for key, value in {
        "llm_router": router,
        "llm_model_list": router.model_list,
        "proxy_logging_obj": _ProxyLogging(),
        "proxy_config": _ProxyConfig(),
        "general_settings": {"disable_budget_reservation": True},
        "master_key": "sk-master",
        "prisma_client": _AuthStore({"sk-causyn": ["causyn-1.1"], "sk-other": ["other-model"]}),
        "user_api_key_cache": UserApiKeyCache(),
        "user_custom_auth": None,
    }.items():
        monkeypatch.setattr(proxy, key, value)
    service = ContextIRService(ContextIRStore(redis), rewrite=rewrite, settle=no_settle)

    async def dependency():
        return service

    app = FastAPI()
    app.dependency_overrides[ir.context_ir_service] = dependency
    app.dependency_overrides[h3.context_ir_service_for_request] = dependency
    app.include_router(h3.router)
    app.include_router(ir.router)
    allowed = {"Authorization": "Bearer sk-causyn"}
    denied = {"Authorization": "Bearer sk-other"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        bad = await client.post("/v2/h3_context_ir", json=spec().model_dump(mode="json"), headers=denied)
        assert bad.status_code in (401, 403), bad.text
        created = await client.post("/v2/h3_context_ir", json=spec().model_dump(mode="json"), headers=allowed)
        assert created.status_code == 200, created.text
        task_id = created.json()["task_id"]
        for method, url in [
            ("GET", "/v2/query/video_generation/" + task_id),
            ("GET", "/v2/query/video_generation"),
            ("DELETE", "/v2/video_generation/" + task_id),
        ]:
            rejected = await client.request(method, url, headers=denied)
            assert rejected.status_code in (401, 403), rejected.text
        cancelled = await client.delete("/v2/video_generation/" + task_id, headers=allowed)
        assert cancelled.status_code == 200, cancelled.text


@pytest.mark.asyncio
async def test_video_bridge_survives_process_loss_without_rewrite_or_duplicate_enqueue(redis, monkeypatch):
    import time
    import litellm.llms.causyn.video_prompt as bridge
    from litellm.llms.causyn.handler import CausynVideoHandler
    from litellm.llms.libtv.video_generate import alive_zset_key, stream_key, TASK_TYPE_VIDEO_GENERATE
    from litellm.llms.libtv.transfer import status_key

    monkeypatch.setenv("DRAMA_CAUSYN_1_1_ENABLED", "true")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_SOURCE_HOSTS", "source.example")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_TARGET_HOSTS", "target.example")
    monkeypatch.setattr(bridge, "get_transfer_redis", lambda url: redis)
    calls = Counter()

    async def counted_rewrite(request):
        calls["rewrite"] += 1
        return RESULT

    async def settle(task):
        calls["settle"] += 1
        await enqueue_causyn_billing(redis, CausynBillingEvent(
            provider_task_id=task.id, response_cost=4, model="causyn-h3-context-ir", task_type="h3_context_ir",
        ))
        if calls["settle"] == 1:
            raise ConnectionError("lost billing acknowledgement after GPU enqueue")

    service = ContextIRService(ContextIRStore(redis), rewrite=counted_rewrite, settle=settle)
    submitted = []

    async def submit(payload, billing):
        submitted.append(await service.create(
            bridge.VideoPromptInput.model_validate(payload.request).context_ir(),
            owner="video:" + payload.task_id, billing=billing, task_id="h3_ir_" + payload.task_id,
            listed=False, video_payload=payload.model_dump(mode="json"),
        ))

    handler = CausynVideoHandler(redis_factory=lambda: redis, prompt_submit=submit)
    video = await handler.avideo_generation(
        model="causyn-1.1", prompt="cat", api_key=None, api_base=None, logging_obj=None,
        optional_params={"seconds": "5", "size": "768p", "aspect_ratio": "16:9",
                         "model_info": {"id": "causyn-1-1", "output_cost_per_second_768p": 5.0}},
    )
    assert calls["rewrite"] == 0
    assert (await handler.avideo_status(video.id, None, None, {}, None)).status == "queued"
    task = submitted[0]
    await service.process(task.id)
    pending = await service.store.get(task.id)
    assert pending.result == RESULT
    assert pending.status == "running"
    assert calls["settle"] == 0
    assert await redis.xlen(stream_key(TASK_TYPE_VIDEO_GENERATE)) == 0

    await redis.zadd(alive_zset_key(TASK_TYPE_VIDEO_GENERATE), {"worker": time.time()})
    restarted = ContextIRService(ContextIRStore(redis), rewrite=counted_rewrite, settle=settle)
    await restarted.process(task.id)
    assert calls["settle"] == 1
    assert (await restarted.store.get(task.id)).status == "running"
    await redis.delete(alive_zset_key(TASK_TYPE_VIDEO_GENERATE))
    await asyncio.gather(restarted.process(task.id), restarted.process(task.id))
    assert (await restarted.store.get(task.id)).status == "succeeded"
    assert calls["rewrite"] == 1
    assert await redis.xlen(stream_key(TASK_TYPE_VIDEO_GENERATE)) == 1
    assert await redis.xlen(CAUSYN_BILLING_STREAM_KEY) == 1
    gpu_payload = json.loads((await redis.xrange(stream_key(TASK_TYPE_VIDEO_GENERATE)))[0][1][b"payload"])
    assert gpu_payload["request"]["prompt"] == PROMPT
    assert await redis.get(status_key(gpu_payload["task_id"])) == b"queued"


@pytest.mark.asyncio
async def test_failed_video_rewrite_never_reaches_gpu_and_is_visible_in_video_status(redis):
    from litellm.llms.causyn.handler import CausynVideoHandler
    from litellm.types.videos.utils import encode_video_id_with_provider
    from litellm.llms.libtv.video_generate import stream_key, TASK_TYPE_VIDEO_GENERATE

    async def bad_rewrite(request):
        raise RewriteError("H3 prompt rewrite failed")

    service = ContextIRService(ContextIRStore(redis), rewrite=bad_rewrite, settle=no_settle)
    video_id = "0123456789abcdef0123456789abcdef"
    task = await service.create(spec(), owner="video:" + video_id, billing=BillingIdentity(),
                                task_id="h3_ir_" + video_id, listed=False, video_payload={"task_id": video_id})
    await service.process(task.id)
    handler = CausynVideoHandler(redis_factory=lambda: redis)
    response = await handler.avideo_status(
        encode_video_id_with_provider(video_id, "causyn", "causyn-1-1"), None, None, {}, None,
    )
    assert response.status == "failed"
    assert await redis.xlen(stream_key(TASK_TYPE_VIDEO_GENERATE)) == 0
    assert (await service.store.get(task.id)).settled


@pytest.mark.asyncio
async def test_ir_admission_refunds_only_if_task_was_not_persisted(redis, monkeypatch):
    service = ContextIRService(ContextIRStore(redis), rewrite=rewrite, settle=no_settle)
    refunded = []
    original_create = service.create

    async def refund(reservation):
        refunded.append(reservation)

    monkeypatch.setattr(ir, "release_unaccepted_reservation", refund)

    async def lost_ack(*args, **kwargs):
        await original_create(*args, **kwargs)
        raise ConnectionError("lost create acknowledgement")

    monkeypatch.setattr(service, "create", lost_ack)
    reservation = {"reserved_cost": 4, "entries": []}
    task = await ir.accept_context_ir(service, spec(), "owner", reservation, BillingIdentity())
    assert task.status == "queued"
    assert refunded == []

    async def unavailable(*args, **kwargs):
        raise ConnectionError("not accepted")

    monkeypatch.setattr(service, "create", unavailable)
    with pytest.raises(ConnectionError):
        await ir.accept_context_ir(service, spec(), "owner", reservation, BillingIdentity())
    assert refunded == [reservation]
