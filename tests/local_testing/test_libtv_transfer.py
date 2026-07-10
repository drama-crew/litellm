import asyncio
import json
import os
import time

import pytest

fakeredis = pytest.importorskip("fakeredis")
from fakeredis import aioredis as fakeredis_aioredis

from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.transfer import (
    DelegatedTransfer,
    DirectTransfer,
    WORKERS_ZSET,
    build_transfer_strategy,
    cas_status,
    status_key,
    result_key,
)


def _make_redis():
    return fakeredis_aioredis.FakeRedis(decode_responses=True)


async def _mark_worker_alive(redis, worker_id="w1"):
    await redis.zadd(WORKERS_ZSET, {worker_id: time.time()})


# ---------------- DirectTransfer ----------------


@pytest.mark.asyncio
async def test_direct_transfer_slices_and_puts_parts():
    fetched = {}
    put_calls = []

    async def fetch(url):
        fetched["url"] = url
        return b"AAAAABBBBBC"

    async def put(url, data):
        put_calls.append((url, bytes(data)))
        return 200, f"etag-{url[-1]}"

    strategy = DirectTransfer(fetch=fetch, put=put, part_size=5)
    parts = [{"n": 1, "url": "https://oss/1"}, {"n": 2, "url": "https://oss/2"}, {"n": 3, "url": "https://oss/3"}]
    etags = await strategy.transfer("https://source/x", 11, parts)

    assert fetched["url"] == "https://source/x"
    assert put_calls == [
        ("https://oss/1", b"AAAAA"),
        ("https://oss/2", b"BBBBB"),
        ("https://oss/3", b"C"),
    ]
    assert etags == [
        {"n": 1, "etag": "etag-1"},
        {"n": 2, "etag": "etag-2"},
        {"n": 3, "etag": "etag-3"},
    ]


@pytest.mark.asyncio
async def test_direct_transfer_raises_on_put_failure():
    async def fetch(url):
        return b"AAAAA"

    async def put(url, data):
        return 403, ""

    strategy = DirectTransfer(fetch=fetch, put=put, part_size=5)
    with pytest.raises(LibTVError):
        await strategy.transfer("https://source/x", 5, [{"n": 1, "url": "https://oss/1"}])


# ---------------- DelegatedTransfer ----------------


@pytest.mark.asyncio
async def test_delegated_transfer_completes_normally():
    redis = _make_redis()
    await _mark_worker_alive(redis)

    fallback_calls = []

    async def fallback_fetch(url):
        fallback_calls.append(url)
        return b""

    fallback = DirectTransfer(fetch=fallback_fetch, put=await _async_put(200, ""))
    strategy = DelegatedTransfer(redis=redis, wait_timeout=5, fallback=fallback)

    async def fake_worker():
        # Wait for the task to land on the stream, then act like a worker completing it.
        for _ in range(50):
            entries = await redis.xrange("worker:tasks:media_transfer")
            if entries:
                break
            await asyncio.sleep(0.05)
        _, fields = entries[0]
        payload = json.loads(fields["payload"])
        assert payload["type"] == "media_transfer"
        task_id = payload["task_id"]
        await redis.set(status_key(task_id), "done")
        await redis.lpush(
            result_key(task_id),
            json.dumps({"ok": True, "result": {"etags": [{"n": 1, "etag": "worker-etag"}], "bytes": payload["source"]["size"]}}),
        )

    worker_task = asyncio.create_task(fake_worker())
    etags = await strategy.transfer("https://source/x", 100, [{"n": 1, "url": "https://oss/1"}])
    await worker_task

    assert etags == [{"n": 1, "etag": "worker-etag"}]
    assert fallback_calls == []  # never fell back to direct


@pytest.mark.asyncio
async def test_delegated_transfer_falls_back_to_direct_on_timeout():
    redis = _make_redis()
    await _mark_worker_alive(redis)

    fallback_calls = []

    async def fallback_fetch(url):
        fallback_calls.append(url)
        return b"DIRECTBYTES"

    fallback = DirectTransfer(fetch=fallback_fetch, put=await _async_put(200, "direct-etag"), part_size=100)
    strategy = DelegatedTransfer(redis=redis, wait_timeout=0.2, fallback=fallback)

    etags = await strategy.transfer("https://source/x", 100, [{"n": 1, "url": "https://oss/1"}])

    assert fallback_calls == ["https://source/x"]
    assert etags == [{"n": 1, "etag": "direct-etag"}]


@pytest.mark.asyncio
async def test_delegated_transfer_late_grace_picks_up_worker_result_after_done_cas():
    redis = _make_redis()
    await _mark_worker_alive(redis)

    fallback_calls = []

    async def fallback_fetch(url):
        fallback_calls.append(url)
        return b"DIRECT"

    fallback = DirectTransfer(fetch=fallback_fetch, put=await _async_put(200, "direct-etag"), part_size=100)
    strategy = DelegatedTransfer(redis=redis, wait_timeout=0.2, late_grace_timeout=2.0, fallback=fallback)

    async def slow_worker():
        # Claim + finish the CAS before the requester's first BRPOP expires, but only
        # push the result afterwards: the requester's cancel CAS must fail (status is
        # already done) and the late-grace second BRPOP must pick the result up.
        for _ in range(50):
            entries = await redis.xrange("worker:tasks:media_transfer")
            if entries:
                break
            await asyncio.sleep(0.01)
        _, fields = entries[0]
        payload = json.loads(fields["payload"])
        task_id = payload["task_id"]
        assert await cas_status(redis, task_id, {"queued"}, "claimed")
        assert await cas_status(redis, task_id, {"claimed"}, "done")
        await asyncio.sleep(0.4)  # past the requester's first BRPOP window
        await redis.lpush(
            result_key(task_id),
            json.dumps({"ok": True, "result": {"etags": [{"n": 1, "etag": "late-but-committed"}]}}),
        )

    worker_task = asyncio.create_task(slow_worker())
    etags = await strategy.transfer("https://source/x", 100, [{"n": 1, "url": "https://oss/1"}])
    await worker_task

    assert etags == [{"n": 1, "etag": "late-but-committed"}]
    assert fallback_calls == []  # the committed worker result was used, not a direct re-upload


@pytest.mark.asyncio
async def test_delegated_transfer_no_active_worker_goes_direct_immediately():
    redis = _make_redis()  # no heartbeat registered

    fallback_calls = []

    async def fallback_fetch(url):
        fallback_calls.append(url)
        return b"X"

    fallback = DirectTransfer(fetch=fallback_fetch, put=await _async_put(200, ""), part_size=100)
    strategy = DelegatedTransfer(redis=redis, wait_timeout=5, fallback=fallback)

    start = asyncio.get_event_loop().time()
    await strategy.transfer("https://source/x", 1, [{"n": 1, "url": "https://oss/1"}])
    elapsed = asyncio.get_event_loop().time() - start

    assert fallback_calls == ["https://source/x"]
    assert elapsed < 1  # no XADD/BRPOP wait incurred
    entries = await redis.xrange("worker:tasks:media_transfer")
    assert entries == []  # never enqueued a task nobody would claim


@pytest.mark.asyncio
async def test_delegated_transfer_ignores_late_worker_result_after_cancel():
    redis = _make_redis()
    await _mark_worker_alive(redis)

    fallback = DirectTransfer(fetch=lambda u: _async_bytes(b"DIRECT"), put=await _async_put(200, "direct-etag"), part_size=100)
    strategy = DelegatedTransfer(redis=redis, wait_timeout=0.2, late_grace_timeout=0.05, fallback=fallback)

    etags = await strategy.transfer("https://source/x", 6, [{"n": 1, "url": "https://oss/1"}])
    assert etags == [{"n": 1, "etag": "direct-etag"}]

    # The task_id is not known to the caller directly, but we can recover it by
    # inspecting the status keys fakeredis now holds; simulate the worker waking up
    # after the deadline and trying to CAS claimed/queued -> done, which must fail
    # because the requester already cancelled it.
    keys = [k async for k in redis.scan_iter("worker:task:status:*")]
    assert len(keys) == 1
    task_id = keys[0].split(":")[-1]
    assert await redis.get(status_key(task_id)) == "cancelled"

    claimed_to_done = await cas_status(redis, task_id, {"queued", "claimed"}, "done")
    assert claimed_to_done is False  # late worker CAS is rejected; must not overwrite cancelled

    # Even if the late worker ignores the CAS result and pushes anyway, nothing reads it.
    await redis.lpush(result_key(task_id), json.dumps({"ok": True, "result": {"etags": [{"n": 1, "etag": "late"}]}}))
    assert etags == [{"n": 1, "etag": "direct-etag"}]  # already returned before the late push


async def _async_bytes(data: bytes) -> bytes:
    return data


async def _async_put(status: int, etag: str = ""):
    async def _put(url, data):
        return status, etag

    return _put


@pytest.mark.asyncio
async def test_delegated_transfer_two_workers_race_only_one_cas_wins():
    redis = _make_redis()
    await _mark_worker_alive(redis, "w1")
    await _mark_worker_alive(redis, "w2")

    fallback = DirectTransfer(fetch=lambda u: _async_bytes(b""), put=await _async_put(200, ""))
    strategy = DelegatedTransfer(redis=redis, wait_timeout=5, fallback=fallback)

    task_id = "race-task"
    key = status_key(task_id)
    assert await redis.set(key, "queued", nx=True)

    results = await asyncio.gather(
        cas_status(redis, task_id, {"queued"}, "claimed"),
        cas_status(redis, task_id, {"queued"}, "claimed"),
    )
    assert sorted(results) == [False, True]
    assert await redis.get(key) == "claimed"


@pytest.mark.asyncio
async def test_delegated_transfer_worker_reported_failure_falls_back_to_direct():
    redis = _make_redis()
    await _mark_worker_alive(redis)

    fallback_calls = []

    async def fallback_fetch(url):
        fallback_calls.append(url)
        return b"DIRECTBYTES"

    fallback = DirectTransfer(fetch=fallback_fetch, put=await _async_put(200, "direct-etag"), part_size=100)
    strategy = DelegatedTransfer(redis=redis, wait_timeout=5, fallback=fallback)

    async def failing_worker():
        for _ in range(50):
            entries = await redis.xrange("worker:tasks:media_transfer")
            if entries:
                break
            await asyncio.sleep(0.02)
        _, fields = entries[0]
        payload = json.loads(fields["payload"])
        task_id = payload["task_id"]
        await redis.lpush(result_key(task_id), json.dumps({"ok": False, "error": "part 2 put failed"}))

    worker_task = asyncio.create_task(failing_worker())
    etags = await strategy.transfer("https://source/x", 100, [{"n": 1, "url": "https://oss/1"}])
    await worker_task

    assert fallback_calls == ["https://source/x"]
    assert etags == [{"n": 1, "etag": "direct-etag"}]


# ---------------- factory ----------------


def test_build_transfer_strategy_defaults_to_direct(monkeypatch):
    monkeypatch.delenv("MEDIA_TRANSFER_MODE", raising=False)
    strategy = build_transfer_strategy(size=10 * 1024 * 1024, redis_client=_make_redis())
    assert isinstance(strategy, DirectTransfer)


def test_build_transfer_strategy_delegated_needs_redis(monkeypatch):
    monkeypatch.setenv("MEDIA_TRANSFER_MODE", "delegated")
    monkeypatch.delenv("MEDIA_TRANSFER_REDIS_URL", raising=False)
    strategy = build_transfer_strategy(size=10 * 1024 * 1024, redis_client=None)
    assert isinstance(strategy, DirectTransfer)  # no redis client injected -> can't delegate


def test_build_transfer_strategy_delegated_with_redis(monkeypatch):
    monkeypatch.setenv("MEDIA_TRANSFER_MODE", "delegated")
    strategy = build_transfer_strategy(size=10 * 1024 * 1024, redis_client=_make_redis())
    assert isinstance(strategy, DelegatedTransfer)


def test_build_transfer_strategy_below_min_bytes_stays_direct(monkeypatch):
    monkeypatch.setenv("MEDIA_TRANSFER_MODE", "delegated")
    monkeypatch.setenv("MEDIA_TRANSFER_MIN_BYTES", str(2 * 1024 * 1024))
    strategy = build_transfer_strategy(size=1024, redis_client=_make_redis())
    assert isinstance(strategy, DirectTransfer)


# ---------------- size-aware wait timeout ----------------


def test_wait_timeout_small_file_floors_at_60(monkeypatch):
    from litellm.llms.libtv.transfer import resolve_wait_timeout

    monkeypatch.delenv("MEDIA_TRANSFER_WAIT_TIMEOUT", raising=False)
    assert resolve_wait_timeout(2 * 1024 * 1024) == 60.0  # 2MiB: formula gives 2s, floor wins


def test_wait_timeout_large_file_scales_with_size(monkeypatch):
    from litellm.llms.libtv.transfer import resolve_wait_timeout

    monkeypatch.delenv("MEDIA_TRANSFER_WAIT_TIMEOUT", raising=False)
    # 200MiB at the assumed 0.5MiB/s direct rate, halved: 200MiB / 0.5MiB * 0.5 = 200s
    assert resolve_wait_timeout(200 * 1024 * 1024) == 200.0


def test_wait_timeout_capped_at_300(monkeypatch):
    from litellm.llms.libtv.transfer import resolve_wait_timeout

    monkeypatch.delenv("MEDIA_TRANSFER_WAIT_TIMEOUT", raising=False)
    assert resolve_wait_timeout(1024 * 1024 * 1024) == 300.0  # 1GiB: formula gives 1024s, cap wins


def test_wait_timeout_env_override_wins(monkeypatch):
    from litellm.llms.libtv.transfer import resolve_wait_timeout

    monkeypatch.setenv("MEDIA_TRANSFER_WAIT_TIMEOUT", "42")
    assert resolve_wait_timeout(1024 * 1024 * 1024) == 42.0


def test_build_transfer_strategy_wait_timeout_is_size_aware(monkeypatch):
    monkeypatch.setenv("MEDIA_TRANSFER_MODE", "delegated")
    monkeypatch.delenv("MEDIA_TRANSFER_WAIT_TIMEOUT", raising=False)
    strategy = build_transfer_strategy(size=200 * 1024 * 1024, redis_client=_make_redis())
    assert isinstance(strategy, DelegatedTransfer)
    assert strategy.wait_timeout == 200.0


@pytest.mark.asyncio
async def test_cas_status_matches_claimed_with_owner_suffix():
    # enterprise workers may store the claim as "claimed:{worker_id}"; the cancel
    # CAS must still treat it as a claimed state it can revoke.
    redis = _make_redis()
    task_id = "owner-suffix-task"
    await redis.set(status_key(task_id), "claimed:worker-7")
    assert await cas_status(redis, task_id, {"queued", "claimed"}, "cancelled") is True
    assert await redis.get(status_key(task_id)) == "cancelled"


# ---------------- production redis wiring ----------------


@pytest.fixture
def reset_transfer_redis(monkeypatch):
    import litellm.llms.libtv.transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "_redis_clients", {})
    return transfer_mod


def test_get_transfer_redis_none_without_url(monkeypatch, reset_transfer_redis):
    from litellm.llms.libtv.transfer import get_transfer_redis

    monkeypatch.delenv("MEDIA_TRANSFER_REDIS_URL", raising=False)
    assert get_transfer_redis() is None


def test_get_transfer_redis_builds_and_reuses_client(monkeypatch, reset_transfer_redis):
    from litellm.llms.libtv.transfer import get_transfer_redis

    monkeypatch.setenv("MEDIA_TRANSFER_REDIS_URL", "redis://127.0.0.1:6399/0")
    client = get_transfer_redis()
    assert client is not None
    assert get_transfer_redis() is client  # reused within the same event-loop context


def test_get_transfer_redis_separate_client_per_event_loop(monkeypatch, reset_transfer_redis):
    from litellm.llms.libtv.transfer import get_transfer_redis

    monkeypatch.setenv("MEDIA_TRANSFER_REDIS_URL", "redis://127.0.0.1:6399/0")

    async def grab():
        return get_transfer_redis()

    loop1 = asyncio.new_event_loop()
    loop2 = asyncio.new_event_loop()
    try:
        c1 = loop1.run_until_complete(grab())
        c1_again = loop1.run_until_complete(grab())
        c2 = loop2.run_until_complete(grab())
    finally:
        loop1.close()
        loop2.close()

    assert c1 is c1_again  # cached per loop
    assert c1 is not c2  # a redis.asyncio client must never cross event loops


@pytest.mark.asyncio
async def test_delegated_transfer_runtime_error_falls_back_without_raising():
    class WrongLoopRedis:
        async def zcount(self, *a, **k):
            raise RuntimeError("Task got Future attached to a different loop")

    fallback_calls = []

    async def fallback_fetch(url):
        fallback_calls.append(url)
        return b"X"

    fallback = DirectTransfer(fetch=fallback_fetch, put=await _async_put(200, "direct-etag"), part_size=100)
    strategy = DelegatedTransfer(redis=WrongLoopRedis(), wait_timeout=5, fallback=fallback)

    etags = await strategy.transfer("https://source/x", 100, [{"n": 1, "url": "https://oss/1"}])
    assert fallback_calls == ["https://source/x"]
    assert etags == [{"n": 1, "etag": "direct-etag"}]


def test_build_transfer_strategy_uses_env_redis_when_not_injected(monkeypatch, reset_transfer_redis):
    monkeypatch.setenv("MEDIA_TRANSFER_MODE", "delegated")
    monkeypatch.setenv("MEDIA_TRANSFER_REDIS_URL", "redis://127.0.0.1:6399/0")
    strategy = build_transfer_strategy(size=10 * 1024 * 1024)
    assert isinstance(strategy, DelegatedTransfer)


@pytest.mark.asyncio
async def test_delegated_transfer_redis_unreachable_falls_back_without_raising():
    import redis.exceptions as redis_exceptions

    class DeadRedis:
        async def zcount(self, *a, **k):
            raise redis_exceptions.ConnectionError("connection refused")

    fallback_calls = []

    async def fallback_fetch(url):
        fallback_calls.append(url)
        return b"X"

    fallback = DirectTransfer(fetch=fallback_fetch, put=await _async_put(200, "direct-etag"), part_size=100)
    strategy = DelegatedTransfer(redis=DeadRedis(), wait_timeout=5, fallback=fallback)

    etags = await strategy.transfer("https://source/x", 100, [{"n": 1, "url": "https://oss/1"}])
    assert fallback_calls == ["https://source/x"]
    assert etags == [{"n": 1, "etag": "direct-etag"}]
