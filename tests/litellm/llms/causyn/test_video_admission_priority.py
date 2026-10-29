import asyncio
import json
import time

import pytest
from fakeredis import aioredis as fakeredis_async

from litellm.llms.causyn.video_admission import active_key, admit_video
from litellm.llms.causyn.video_priority import PRIORITY_ENV, priority_route
from litellm.llms.libtv.transfer import status_key

STREAM = "worker:tasks:video_generate_ref2va"
REQUEST = {"duration_seconds": 15, "references": [{"media_type": "video"}]}


async def submit(redis, identifier, *, request=REQUEST, pool=None, stream=STREAM):
    return await admit_video(
        redis,
        task_id=identifier,
        metadata=json.dumps({"identity": identifier}),
        envelope=json.dumps({"task_id": identifier, "request": request}),
        deadline=time.time() + 1200,
        stream=stream,
        pool=pool,
        request=request,
    )


@pytest.mark.asyncio
async def test_concurrent_duplicate_admission_keeps_one_stream_entry_and_catalog_member(monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "1")
    async with fakeredis_async.FakeRedis() as redis:
        route = priority_route(STREAM, model="causyn-1.1", request=REQUEST)
        assert all(await asyncio.gather(*(submit(redis, "same") for _ in range(12))))
        assert await redis.xlen(route.stream) == 1
        assert await redis.zrange(route.catalog_key, 0, -1, withscores=True) == [
            (route.stream.encode(), float(route.priority))
        ]
        assert await redis.get(status_key("same")) == b"queued"
        assert await redis.zcard(active_key()) == 1
        assert not await redis.exists(STREAM)
        changed = {"duration_seconds": 5}
        assert await submit(redis, "same", request=changed)
        assert not await redis.exists(priority_route(STREAM, model="causyn-1.1", request=changed).stream)


@pytest.mark.asyncio
async def test_cap_is_shared_by_all_priority_scores_and_terminal_jobs_release_admission(monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "true")
    monkeypatch.setenv("CAUSYN_H3_MAX_ADMITTED", "2")
    async with fakeredis_async.FakeRedis() as redis:
        assert await submit(redis, "long")
        assert await submit(redis, "short", request={"duration_seconds": 5})
        assert not await submit(redis, "refused", request={"duration_seconds": 10})
        assert not await redis.exists(status_key("refused"))
        assert not await redis.exists("worker:task:metadata:refused")
        await redis.set(status_key("long"), "done")
        assert await submit(redis, "next", request={"duration_seconds": 10})
        assert await redis.zcard(active_key()) == 2


@pytest.mark.asyncio
async def test_disabled_writer_and_test_pool_keep_their_legacy_streams(monkeypatch):
    monkeypatch.delenv(PRIORITY_ENV, raising=False)
    async with fakeredis_async.FakeRedis() as redis:
        assert await submit(redis, "legacy")
        assert await redis.xlen(STREAM) == 1
        monkeypatch.setenv(PRIORITY_ENV, "true")
        test_stream = "worker:tasks:video_generate_ref2va_test"
        assert await submit(redis, "test", stream=test_stream, pool="test")
        assert await redis.xlen(test_stream) == 1
        assert await redis.zcard(active_key("test")) == 1
        assert not await redis.exists("worker:priority-streams:video_generate_ref2va_test")
        assert not await redis.exists("worker:priority-streams:video_generate_ref2va")


@pytest.mark.asyncio
async def test_catalog_wrong_type_is_refused_before_any_new_durable_write(monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "true")
    async with fakeredis_async.FakeRedis() as redis:
        route = priority_route(STREAM, model="causyn-1.1", request=REQUEST)
        await redis.set(route.catalog_key, "bad-type")
        assert not await submit(redis, "refused")
        assert not await redis.exists(status_key("refused"))
        assert not await redis.exists(route.stream)
        assert not await redis.exists(active_key())
        await redis.set(status_key("already-admitted"), "claimed")
        assert await submit(redis, "already-admitted")


@pytest.mark.asyncio
async def test_existing_wrong_type_priority_stream_cannot_leave_an_orphan_status(monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "true")
    async with fakeredis_async.FakeRedis() as redis:
        route = priority_route(STREAM, model="causyn-1.1", request=REQUEST)
        await redis.set(route.stream, "bad-type")
        assert not await submit(redis, "refused")
        assert not await redis.exists(status_key("refused"))
        assert not await redis.exists("worker:task:metadata:refused")
        assert not await redis.exists(route.catalog_key)
        assert not await redis.exists(active_key())
