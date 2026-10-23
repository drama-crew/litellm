# ruff: noqa: F811  # pytest fixtures imported from test_h3_ref2va_rewrite are requested by name
"""A rewriter answer that fails the structural validators is a bad sample, not a permanent failure.

Production (causyn.cn): ``The rewritten H3 prompt has invalid shot numbering`` and ``... does not mirror the source
video's shot cuts`` ended a task at attempt=1 (one answer + the one repair call), because validator errors were
non-retryable. These tests drive the real ContextIRService -> H3PromptRewriter -> validators path with a stubbed
OpenRouter transport.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import fakeredis.aioredis
import httpx
import pytest

from litellm.llms.causyn import h3_prompt
from litellm.llms.causyn.context_ir import ContextIRService, rewrite_policy
from litellm.llms.causyn.context_ir_store import BillingIdentity, ContextIRStore, ContextIRTask
from litellm.llms.causyn.h3_prompt import H3PromptRewriter, RewriteError

from .test_h3_ref2va_rewrite import GOLD, Provider, case, clean_state, mutate, patched  # noqa: F401

SHOTS_MSG = "The rewritten H3 prompt has invalid shot numbering"
PREFIX = "prompt rewrite could not produce a valid structured prompt"


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.aclose()


async def no_settle(task):
    return None


def bad_shots() -> str:
    return mutate("R1", "[Shot 2] At", "[Shot 3] At")


async def drive(service: ContextIRService, redis, task_id: str, rounds: int = 8) -> ContextIRTask:
    """Run the worker loop by hand: process, drop the lease, until the task is terminal."""
    for _ in range(rounds):
        await service.process(task_id)
        await redis.delete(f"causyn:context-ir:task:{task_id}:lease")
        current = await service.store.get(task_id)
        if current.status in {"succeeded", "failed"}:
            return current
    return await service.store.get(task_id)


async def run_task(redis, provider, spec, rounds=8, **create):
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        service = ContextIRService(
            ContextIRStore(redis), rewrite=H3PromptRewriter(client, "k").rewrite, settle=no_settle
        )
        task = await service.create(spec, owner="owner", billing=BillingIdentity(), **create)
        return service, await drive(service, redis, task.id, rounds)


@pytest.fixture(autouse=True)
def fast_retry(monkeypatch):
    # The tests drive attempts by hand: never sleep, never jitter.
    monkeypatch.setattr(h3_prompt.random, "uniform", lambda a, b: 0.0)
    monkeypatch.delenv("CAUSYN_REWRITE_VALIDATION_MAX_ATTEMPTS", raising=False)


@pytest.mark.asyncio
async def test_validation_failure_is_retried_with_a_fresh_rewrite_and_then_succeeds(redis, patched):
    spec, facts = case("R1")
    patched(spec, facts)
    # attempt 1: first answer + the one repair call both violate; attempt 2: a fresh sample is valid.
    provider = Provider(bad_shots(), bad_shots(), GOLD["R1"])
    service, done = await run_task(redis, provider, spec)
    assert done.status == "succeeded", done.error
    assert done.result.prompt == GOLD["R1"].strip()
    assert len(provider.bodies) == 3
    assert done.attempts == 2 and done.validation_failures == 1
    # The retry is a *fresh* rewrite: it does not replay the violating answer.
    assert provider.bodies[2]["messages"] == provider.bodies[0]["messages"]
    # Only the winning attempt's usage is recorded (failed attempts are not billed to the customer).
    assert done.result.usage.prompt_tokens == 100 and done.result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_all_attempts_failing_is_bounded_and_the_reason_names_the_violation(redis, patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(*[bad_shots()] * 20)
    service, done = await run_task(redis, provider, spec)
    assert done.status == "failed"
    # 3 attempts x (first answer + repair call); never more.
    assert len(provider.bodies) == 6
    assert done.attempts == 3 and done.validation_failures == 3
    assert done.error == f"{PREFIX}: {SHOTS_MSG}"
    assert SHOTS_MSG in (done.error_detail or "")
    assert "Bearer" not in json.dumps(done.model_dump(mode="json"))
    assert done.settled and done.financial_actual == 0  # a failed rewrite is never billable


@pytest.mark.asyncio
async def test_max_validation_attempts_is_configurable(redis, patched, monkeypatch):
    monkeypatch.setenv("CAUSYN_REWRITE_VALIDATION_MAX_ATTEMPTS", "1")
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(*[bad_shots()] * 20)
    service, done = await run_task(redis, provider, spec)
    assert done.status == "failed" and len(provider.bodies) == 2
    assert done.error == f"{PREFIX}: {SHOTS_MSG}"


@pytest.mark.asyncio
async def test_repair_within_one_attempt_is_unchanged(redis, patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(bad_shots(), GOLD["R1"])
    service, done = await run_task(redis, provider, spec)
    assert done.status == "succeeded"
    assert len(provider.bodies) == 2
    assert done.attempts == 1 and done.validation_failures == 0


@pytest.mark.asyncio
async def test_non_validation_provider_error_still_fails_at_once_with_the_old_message(redis, patched):
    spec, facts = case("R1")
    patched(spec, facts)
    calls = Counter()

    def provider(request):
        calls["n"] += 1
        return httpx.Response(400)

    service, done = await run_task(redis, provider, spec)
    assert done.status == "failed" and calls["n"] == 1
    assert done.error == "H3 prompt rewrite provider unavailable"
    assert done.validation_failures == 0


@pytest.mark.asyncio
async def test_transient_errors_do_not_consume_the_validation_budget(redis, patched):
    spec, facts = case("R1")
    patched(spec, facts)
    answers = iter([bad_shots(), bad_shots(), GOLD["R1"]])
    calls = Counter()

    def provider(request):
        calls["n"] += 1
        if calls["n"] in {1, 4}:  # first attempt's first call, and the second attempt's first call
            return httpx.Response(429)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": next(answers)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    service, done = await run_task(redis, provider, spec, rounds=12)
    assert done.status == "succeeded", done.error
    assert done.validation_failures == 1


def _task(model="causyn-1.1", **update):
    now = int(time.time())
    base = {
        "id": "t1",
        "owner": "o",
        "request": case("R1")[0] if model == "causyn-1.1" else case("R1")[0].model_copy(update={"model": model}),
        "created_at": now,
        "updated_at": now,
        "billing": BillingIdentity(),
        "price": 1.0,
    }
    return ContextIRTask(**{**base, **update})


def _invalid(**kwargs):
    return RewriteError(SHOTS_MSG, validation=True, **kwargs)


def test_validation_retry_respects_the_retry_window_and_render_reserve():
    from litellm.llms.causyn.context_ir import REWRITE_RETRY_WINDOW_S

    task = _task(validation_failures=1, attempts=1)
    task = task.model_copy(update={"rewrite_started_at": float(task.created_at)})
    policy = rewrite_policy(task)
    assert policy.next_delay(task, _invalid(), now=task.created_at + 5) is not None
    assert policy.next_delay(task, _invalid(), now=task.created_at + REWRITE_RETRY_WINDOW_S) is None
    tight = _task(
        validation_failures=1, attempts=1, video_payload={"deadline_ts": _task().created_at + 1700}
    )
    assert rewrite_policy(tight).next_delay(tight, _invalid(), now=tight.created_at + 10) is None


def test_validation_retry_is_capped_by_the_validation_budget():
    task = _task(validation_failures=3, attempts=3)
    assert rewrite_policy(task).next_delay(task, _invalid(), now=task.created_at) is None
    fewer = _task(validation_failures=2, attempts=2)
    assert rewrite_policy(fewer).next_delay(fewer, _invalid(), now=fewer.created_at) is not None


def test_non_validation_non_retryable_errors_are_still_terminal():
    task = _task()
    bad = RewriteError("nope", 502, retryable=False, upstream_status=400)
    assert rewrite_policy(task).next_delay(task, bad, now=task.created_at) is None


def test_validation_retry_keeps_the_legacy_three_attempt_cap_for_minimax_h3():
    two = _task("MiniMax-H3", validation_failures=2, attempts=2)
    three = _task("MiniMax-H3", validation_failures=3, attempts=3)
    assert rewrite_policy(two).next_delay(two, _invalid(), now=two.created_at) is not None
    assert rewrite_policy(three).next_delay(three, _invalid(), now=three.created_at) is None


def test_validator_failures_are_marked_and_provider_failures_are_not(patched):
    spec, facts = case("R1")
    with pytest.raises(RewriteError) as caught:
        h3_prompt.validate_prompt(bad_shots(), spec, facts)
    assert caught.value.validation is True
    assert RewriteError("x", 502).validation is False
    assert RewriteError("x", 429).validation is False


@pytest.mark.asyncio
async def test_final_video_failure_exposes_the_specific_rewrite_reason(redis, patched):
    from litellm.llms.causyn.handler import CausynVideoHandler
    from litellm.types.videos.utils import encode_video_id_with_provider

    spec, facts = case("R1")
    patched(spec, facts)
    video_id = "0123456789abcdef0123456789abcdef"
    provider = Provider(*[bad_shots()] * 20)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        service = ContextIRService(
            ContextIRStore(redis), rewrite=H3PromptRewriter(client, "k").rewrite, settle=no_settle
        )
        task = await service.create(
            spec,
            owner="video:" + video_id,
            billing=BillingIdentity(),
            task_id="h3_ir_" + video_id,
            listed=False,
            video_payload={"task_id": video_id},
        )
        done = await drive(service, redis, task.id)
    assert done.status == "failed"
    response = await CausynVideoHandler(redis_factory=lambda: redis).avideo_status(
        encode_video_id_with_provider(video_id, "causyn", "causyn-1-1"), None, None, {}, None
    )
    assert response.status == "failed"
    assert response.error["code"] == "prompt_rewrite_failed"
    assert response.error["message"] == f"{PREFIX}: {SHOTS_MSG}"


# --- rollback safety: blobs stay readable by the previous schema (extra="forbid") ---------------


def test_validation_failures_is_omitted_from_the_blob_until_a_validation_retry_happened():
    fresh = _task()
    assert "validation_failures" not in json.loads(fresh.model_dump_json())
    assert "validation_failures" not in fresh.model_dump(mode="json")
    retried = _task(validation_failures=2)
    assert json.loads(retried.model_dump_json())["validation_failures"] == 2
    assert ContextIRTask.model_validate_json(retried.model_dump_json()).validation_failures == 2


def test_an_old_blob_without_validation_failures_still_loads():
    blob = json.loads(_task().model_dump_json())
    assert "validation_failures" not in blob
    assert ContextIRTask.model_validate_json(json.dumps(blob)).validation_failures == 0


@pytest.mark.asyncio
async def test_stored_blob_of_an_untouched_task_has_no_validation_failures_key(redis):
    store = ContextIRStore(redis)
    service = ContextIRService(store, rewrite=lambda r: None, settle=no_settle)
    task = await service.create(case("R1")[0], owner="owner", billing=BillingIdentity())
    raw = await redis.get(f"causyn:context-ir:task:{task.id}")
    assert "validation_failures" not in json.loads(raw)


# --- handler: only our own validation reason is exposed ---------------------------------------------


@pytest.mark.parametrize(
    "code,message",
    [
        ("worker_failed", f"{PREFIX}: {SHOTS_MSG}"),  # wrong code
        ("prompt_rewrite_failed", PREFIX),  # prefix without ": "
        ("prompt_rewrite_failed", f"{PREFIX}:{SHOTS_MSG}"),  # no space after the colon
        ("prompt_rewrite_failed", f"{PREFIX}: " + "x" * 300),  # over 300 characters
        ("prompt_rewrite_failed", "redis://user:secret@internal:6379 upstream failed"),  # opaque worker text
        ("prompt_rewrite_failed", None),
    ],
)
def test_public_worker_error_keeps_everything_else_opaque(code, message):
    from litellm.llms.causyn.handler import _public_worker_error, _WorkerError

    public = _public_worker_error(_WorkerError(code=code, message=message))
    assert public["message"] == "video generation failed"


def test_public_worker_error_exposes_the_validation_reason():
    from litellm.llms.causyn.handler import _public_worker_error, _WorkerError

    reason = f"{PREFIX}: {SHOTS_MSG}"
    public = _public_worker_error(_WorkerError(code="prompt_rewrite_failed", message=reason))
    assert public["message"] == reason and public["code"] == "prompt_rewrite_failed"


# --- a fresh attempt never replays the violating first answer -----------------------------------------


@pytest.mark.asyncio
async def test_first_answer_cache_is_not_reused_by_a_fresh_attempt_after_a_validation_failure(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(bad_shots(), bad_shots(), GOLD["R1"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        rewriter = H3PromptRewriter(client, "k")
        with pytest.raises(RewriteError) as caught:
            await rewriter.rewrite(spec)
        assert caught.value.validation is True
        assert len(provider.bodies) == 2 and not h3_prompt._FIRST_ANSWERS  # nothing left to replay
        result = await rewriter.rewrite(spec)
    assert len(provider.bodies) == 3  # a new provider call, not the cached violating answer
    assert provider.bodies[2]["messages"] == provider.bodies[0]["messages"]  # started over, no repair turn
    assert result.prompt == GOLD["R1"].strip()
