"""Test worker pool: enqueue picks the *_test streams, admission is per pool, production is untouched."""

from __future__ import annotations

import json
import time

import fakeredis.aioredis
import pytest

import litellm.llms.causyn  # noqa: F401  # resolves the existing causyn <-> video_generate import cycle first
from litellm.llms.causyn.video_admission import ACTIVE, MAX_ADMITTED, active_key, max_admitted
from litellm.llms.libtv.transfer import status_key
from litellm.llms.libtv.video_generate import (
    VideoGenerateError,
    VideoGenerateSettings,
    alive_zset_key,
    enqueue_video_generate,
    stream_key,
)

SETTINGS = VideoGenerateSettings(source_hosts=frozenset({"source.example"}), target_hosts=frozenset({"target.example"}))
ALL_TYPES = (
    "video_generate",
    "video_generate_ref2va",
    "video_generate_test",
    "video_generate_ref2va_test",
)


@pytest.fixture
async def redis():
    async with fakeredis.aioredis.FakeRedis() as client:
        yield client


async def alive(redis, *task_types):
    for task_type in task_types:
        await redis.zadd(alive_zset_key(task_type), {"worker": time.time()})


def references(kind):
    url = "https://source.example/a.jpg"
    return {
        "t2va": [],
        "fl2va": [
            {"role": "first_frame", "media_type": "image", "url": url},
            {"role": "last_frame", "media_type": "image", "url": url},
        ],
        "ref2va": [{"role": "reference", "media_type": "image", "url": url}],
    }[kind]


async def submit(redis, task_id, kind="t2va", pool=None):
    payload = {
        "task_id": task_id,
        "model": "causyn-1.1",
        "deadline_ts": time.time() + 1800,
        "request": {
            "prompt": "cat",
            "duration_seconds": 5,
            "resolution": "768p",
            "ratio": "16:9",
            "references": references(kind),
        },
        "task_metadata": {"provider_task_id": task_id},
    }
    if pool is not None:
        payload["worker_pool"] = pool
    return await enqueue_video_generate(payload, redis_factory=lambda: redis, settings=SETTINGS)


async def lengths(redis):
    return {task_type: await redis.xlen(stream_key(task_type)) for task_type in ALL_TYPES}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,test_stream,prod_stream",
    [
        ("t2va", "video_generate_test", "video_generate"),
        ("fl2va", "video_generate_test", "video_generate"),
        ("ref2va", "video_generate_ref2va_test", "video_generate_ref2va"),
    ],
)
async def test_pool_selects_exactly_one_stream_and_the_matching_envelope_type(redis, kind, test_stream, prod_stream):
    await alive(redis, *ALL_TYPES)
    await submit(redis, "prod", kind)
    await submit(redis, "test", kind, "test")
    counts = await lengths(redis)
    assert counts[prod_stream] == 1 and counts[test_stream] == 1 and sum(counts.values()) == 2
    prod = json.loads((await redis.xrange(stream_key(prod_stream)))[0][1][b"payload"])
    test = json.loads((await redis.xrange(stream_key(test_stream)))[0][1][b"payload"])
    assert (prod["type"], prod["task_id"]) == (prod_stream, "prod")
    assert (test["type"], test["task_id"]) == (test_stream, "test")
    assert "worker_pool" not in test and "worker_pool" not in prod  # the worker never sees the routing flag


@pytest.mark.asyncio
async def test_liveness_follows_the_pool(redis):
    await alive(redis, "video_generate")  # only the production worker is up
    with pytest.raises(VideoGenerateError) as error:
        await submit(redis, "t", "t2va", "test")
    assert error.value.code == "no_worker_available" and "video_generate_test" in str(error.value)
    await redis.delete(alive_zset_key("video_generate"))
    await alive(redis, "video_generate_test")  # only a test worker is up: production must still be refused
    with pytest.raises(VideoGenerateError) as error:
        await submit(redis, "p", "t2va")
    assert error.value.code == "no_worker_available" and "video_generate" in str(error.value)
    await submit(redis, "t2", "t2va", "test")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["prod", "TEST", "", 1, True, ["test"]])
async def test_worker_pool_values_other_than_test_are_rejected(redis, bad):
    await alive(redis, *ALL_TYPES)
    with pytest.raises(VideoGenerateError) as error:
        await submit(redis, "t", "t2va", bad)
    assert error.value.code == "invalid_params"
    assert sum((await lengths(redis)).values()) == 0


@pytest.mark.asyncio
async def test_explicit_none_pool_is_the_production_path(redis):
    await alive(redis, *ALL_TYPES)
    payload = {
        "task_id": "t",
        "model": "causyn-1.1",
        "deadline_ts": time.time() + 1800,
        "request": {"prompt": "cat", "duration_seconds": 5, "resolution": "768p", "ratio": "16:9"},
        "task_metadata": {},
        "worker_pool": None,
    }
    await enqueue_video_generate(payload, redis_factory=lambda: redis, settings=SETTINGS)
    assert (await lengths(redis))["video_generate"] == 1


def test_admission_keys_and_caps_per_pool(monkeypatch):
    monkeypatch.delenv("CAUSYN_TEST_POOL_MAX_ADMITTED", raising=False)
    assert active_key(None) == ACTIVE == "causyn:video:admitted"
    assert active_key("test") == "causyn:video:admitted:test"
    assert max_admitted(None) == MAX_ADMITTED == 8
    assert max_admitted("test") == 2
    monkeypatch.setenv("CAUSYN_TEST_POOL_MAX_ADMITTED", "5")
    assert max_admitted("test") == 5 and max_admitted(None) == 8
    for junk in ("0", "-1", "abc", ""):
        monkeypatch.setenv("CAUSYN_TEST_POOL_MAX_ADMITTED", junk)
        assert max_admitted("test") == 2


@pytest.mark.asyncio
async def test_test_pool_cap_is_independent_and_releases_terminal_tasks(redis, monkeypatch):
    monkeypatch.delenv("CAUSYN_TEST_POOL_MAX_ADMITTED", raising=False)
    await alive(redis, *ALL_TYPES)
    await submit(redis, "t1", "t2va", "test")
    await submit(redis, "t2", "ref2va", "test")
    with pytest.raises(VideoGenerateError) as error:
        await submit(redis, "t3", "t2va", "test")
    assert error.value.code == "no_capacity_available"
    assert await redis.zcard("causyn:video:admitted:test") == 2
    assert await redis.zcard(ACTIVE) == 0  # test traffic holds no production slot
    for index in range(MAX_ADMITTED):  # a full test pool does not shrink production capacity
        await submit(redis, f"p{index}", "t2va")
    assert await redis.zcard(ACTIVE) == MAX_ADMITTED
    with pytest.raises(VideoGenerateError):
        await submit(redis, "p-over", "t2va")
    await redis.set(status_key("t1"), "done")  # completing a test task frees the TEST set
    await submit(redis, "t4", "t2va", "test")
    assert await redis.zcard("causyn:video:admitted:test") == 2
    assert await redis.zcard(ACTIVE) == MAX_ADMITTED
    assert set(await redis.zrange("causyn:video:admitted:test", 0, -1)) == {b"t2", b"t4"}
