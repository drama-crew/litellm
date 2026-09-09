"""third_asset registration: intra-request dedupe, cross-request cache, rate limit.

Regression suite for the 2026-09-01 causyn.cn incident: 10 video shots carrying
97 reference images (30 distinct) were submitted inside 11.5s, firing ~97
`third_asset/create` calls against libtv's documented cap of 15/minute. Every
one of the 10 submits came back `code=10026 当前素材检测提交较频繁`.
"""

import asyncio
import time

import pytest

from litellm.llms.libtv.asset_registry import (
    ThirdAssetRateLimiter,
    acquire_timeout_seconds,
    asset_cache_ttl_seconds,
    create_rate_per_minute,
    dedupe_preserving_order,
)
from litellm.llms.libtv.common import LibTVError


# ---------- dedupe ----------


def test_dedupe_preserving_order_returns_unique_values_and_index_map():
    values = ["a", "b", "a", "c", "b"]
    unique, index_for = dedupe_preserving_order(values)
    assert unique == ["a", "b", "c"]
    assert [index_for[v] for v in values] == [0, 1, 0, 2, 1]


def test_dedupe_preserving_order_handles_empty():
    unique, index_for = dedupe_preserving_order([])
    assert unique == []
    assert index_for == {}


# ---------- env knobs ----------


def test_create_rate_defaults_below_the_upstream_cap(monkeypatch):
    monkeypatch.delenv("LIBTV_THIRD_ASSET_CREATE_RPM", raising=False)
    # Upstream tells us 15/min; the default must leave headroom, not sit on it.
    assert create_rate_per_minute() == 12
    assert create_rate_per_minute() < 15


@pytest.mark.parametrize("raw", ["0", "-3", "not-a-number", ""])
def test_create_rate_rejects_unusable_overrides(monkeypatch, raw):
    monkeypatch.setenv("LIBTV_THIRD_ASSET_CREATE_RPM", raw)
    assert create_rate_per_minute() == 12


def test_create_rate_honours_a_valid_override(monkeypatch):
    monkeypatch.setenv("LIBTV_THIRD_ASSET_CREATE_RPM", "5")
    assert create_rate_per_minute() == 5


def test_acquire_timeout_stays_under_the_platform_submit_read_timeout(monkeypatch):
    monkeypatch.delenv("LIBTV_THIRD_ASSET_ACQUIRE_TIMEOUT_SECONDS", raising=False)
    # openhands video_provider.DEFAULT_VIDEO_SUBMIT_READ_TIMEOUT_SECONDS is 600s;
    # waiting past that turns a survivable queue into a client-side ReadTimeout.
    assert acquire_timeout_seconds() == 420.0
    assert acquire_timeout_seconds() < 600.0


def test_asset_cache_ttl_default_is_one_day(monkeypatch):
    monkeypatch.delenv("LIBTV_ASSET_CACHE_TTL_SECONDS", raising=False)
    assert asset_cache_ttl_seconds() == 86400.0


def test_asset_cache_ttl_zero_disables_the_cache(monkeypatch):
    monkeypatch.setenv("LIBTV_ASSET_CACHE_TTL_SECONDS", "0")
    assert asset_cache_ttl_seconds() == 0.0


# ---------- in-process limiter ----------


class _Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.mark.asyncio
async def test_limiter_lets_the_first_n_through_without_sleeping():
    clock = _Clock()
    limiter = ThirdAssetRateLimiter(
        redis_client=None, rate_per_minute=3, monotonic=clock.monotonic, sleep=clock.sleep
    )
    for _ in range(3):
        await limiter.acquire("acct")
    assert clock.slept == []


@pytest.mark.asyncio
async def test_limiter_waits_for_the_window_to_roll_before_the_extra_call():
    clock = _Clock()
    limiter = ThirdAssetRateLimiter(
        redis_client=None, rate_per_minute=3, monotonic=clock.monotonic, sleep=clock.sleep
    )
    for _ in range(3):
        await limiter.acquire("acct")
    await limiter.acquire("acct")
    # The 4th call must wait until the oldest of the 3 leaves the 60s window.
    assert clock.slept and sum(clock.slept) >= 60.0


@pytest.mark.asyncio
async def test_limiter_windows_are_per_account():
    clock = _Clock()
    limiter = ThirdAssetRateLimiter(
        redis_client=None, rate_per_minute=2, monotonic=clock.monotonic, sleep=clock.sleep
    )
    await limiter.acquire("acct-a")
    await limiter.acquire("acct-a")
    await limiter.acquire("acct-b")
    await limiter.acquire("acct-b")
    assert clock.slept == []


@pytest.mark.asyncio
async def test_limiter_raises_a_rate_limit_error_once_the_wait_budget_is_gone():
    clock = _Clock()
    limiter = ThirdAssetRateLimiter(
        redis_client=None,
        rate_per_minute=1,
        acquire_timeout=30.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    await limiter.acquire("acct")
    with pytest.raises(LibTVError) as excinfo:
        await limiter.acquire("acct")
    # 429 so the caller can tell "queued out" apart from "this input is invalid".
    assert excinfo.value.status_code == 429


@pytest.mark.asyncio
async def test_limiter_is_concurrency_safe_within_one_process():
    clock = _Clock()
    limiter = ThirdAssetRateLimiter(
        redis_client=None, rate_per_minute=5, monotonic=clock.monotonic, sleep=clock.sleep
    )
    await asyncio.gather(*(limiter.acquire("acct") for _ in range(5)))
    assert clock.slept == []
    await limiter.acquire("acct")
    assert clock.slept, "the 6th concurrent acquire must have been throttled"


# ---------- redis-backed limiter ----------


class _FakeRedis:
    """Just enough of redis.asyncio for the sliding-window Lua script."""

    def __init__(self, *, fail=False):
        self.fail = fail
        self.zsets: dict[str, list[float]] = {}
        self.eval_calls = 0

    async def eval(self, script, numkeys, key, now, window, limit, member):  # noqa: ARG002
        self.eval_calls += 1
        if self.fail:
            raise RuntimeError("redis down")
        now = float(now)
        window = float(window)
        limit = int(limit)
        bucket = [t for t in self.zsets.setdefault(key, []) if t > now - window]
        self.zsets[key] = bucket
        if len(bucket) < limit:
            bucket.append(now)
            return "0"
        return str(bucket[0] + window - now)


@pytest.mark.asyncio
async def test_redis_limiter_shares_the_window_across_processes():
    clock = _Clock()
    redis = _FakeRedis()
    # Two limiters = two uvicorn workers sharing one libtv account.
    kwargs = dict(rate_per_minute=3, monotonic=clock.monotonic, sleep=clock.sleep)
    a = ThirdAssetRateLimiter(redis_client=redis, **kwargs)
    b = ThirdAssetRateLimiter(redis_client=redis, **kwargs)
    await a.acquire("acct")
    await b.acquire("acct")
    await a.acquire("acct")
    assert clock.slept == []
    await b.acquire("acct")
    assert clock.slept, "the shared window must throttle the 4th call"


@pytest.mark.asyncio
async def test_redis_limiter_falls_back_in_process_when_redis_is_down():
    clock = _Clock()
    redis = _FakeRedis(fail=True)
    limiter = ThirdAssetRateLimiter(
        redis_client=redis, rate_per_minute=2, monotonic=clock.monotonic, sleep=clock.sleep
    )
    # A broken limiter must never take generation down with it: it degrades to
    # the in-process window rather than raising.
    await limiter.acquire("acct")
    await limiter.acquire("acct")
    assert clock.slept == []
    await limiter.acquire("acct")
    assert clock.slept, "in-process fallback must still throttle"


@pytest.mark.asyncio
async def test_redis_limiter_stops_calling_redis_after_it_fails():
    clock = _Clock()
    redis = _FakeRedis(fail=True)
    limiter = ThirdAssetRateLimiter(
        redis_client=redis, rate_per_minute=10, monotonic=clock.monotonic, sleep=clock.sleep
    )
    for _ in range(5):
        await limiter.acquire("acct")
    assert redis.eval_calls == 1, "a dead redis must be tried once, not on every asset"


# ---------- client wiring: dedupe + cache + limiter on the real resolve path ----------

import json  # noqa: E402

from litellm.llms.libtv import asset_registry as _asset_registry  # noqa: E402
from litellm.llms.libtv import client as _libtv_client  # noqa: E402
from litellm.llms.libtv.client import LibTVClient  # noqa: E402

_REF_A = "https://libtv-res.liblib.art/upload-images/uid/a.png"
_REF_B = "https://libtv-res.liblib.art/upload-images/uid/b.png"
_REF_C = "https://libtv-res.liblib.art/upload-images/uid/c.png"


@pytest.fixture(autouse=True)
def _reset_limiters(monkeypatch):
    # The client-wiring tests exercise the real limiter object; without a wide
    # rate they would sleep out the genuine 60s window. Throttling behaviour has
    # its own tests above, driven by an injected clock.
    monkeypatch.setenv("LIBTV_THIRD_ASSET_CREATE_RPM", "100000")
    _asset_registry.reset_third_asset_limiters()
    yield
    _asset_registry.reset_third_asset_limiters()


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.headers = {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _AssetFakeAsyncClient:
    """Answers image/verify + third_asset/{create,check} for arbitrary urls."""

    def __init__(self, *, terminal=True, assets=None):
        self.terminal = terminal
        self.assets = assets or {}
        self.calls = []
        self._seq = 0

    async def post(self, url, json=None, headers=None, timeout=None):
        path = url.split("api.liblib.tv", 1)[-1]
        self.calls.append((path, json))
        if path == "/api/community/image/verify":
            risk = __import__("json").dumps({"passed": True, "needsReview": False, "riskDescription": "正常"})
            return _Resp(
                {"code": 0, "data": {"list": [{"url": u, "riskLabels": risk} for u in json["urlList"]]}}
            )
        if path == "/api/third_asset/create":
            self._seq += 1
            uid = f"u{self._seq}"
            self.assets[uid] = json["assetUrl"]
            return _Resp({"code": 0, "data": {"uuid": uid}})
        if path == "/api/third_asset/check":
            uid = json["uuids"][0]
            if not self.terminal:
                # status 0 == still pending: never a terminal answer.
                return _Resp({"code": 0, "data": {"list": [{"uuid": uid, "found": True, "status": 0}]}})
            return _Resp(
                {"code": 0, "data": {"list": [{"uuid": uid, "assetId": f"asset-{uid}", "found": True, "status": 1}]}}
            )
        raise AssertionError(f"unexpected path {path}")

    async def post_once(self, url, json=None, headers=None, timeout=None):
        return await self.post(url, json, headers, timeout)

    def creates(self):
        return [body["assetUrl"] for path, body in self.calls if path == "/api/third_asset/create"]


class _FakePersistence:
    def __init__(self, prestored=None):
        self.store = dict(prestored or {})
        self.stored_calls = []

    async def cached_asset(self, account_key, cdn_url, asset_type, *, ttl_seconds):
        if ttl_seconds <= 0:
            return None
        return self.store.get((account_key, cdn_url, asset_type))

    async def store_asset(self, account_key, cdn_url, asset_type, asset_id):
        self.stored_calls.append((account_key, cdn_url, asset_type, asset_id))
        self.store[(account_key, cdn_url, asset_type)] = {"url": cdn_url, "assetId": asset_id}


def _client(fake, persistence=None):
    return LibTVClient(
        token="tok",
        webid="wid",
        async_client=fake,
        poll_interval=0,
        persistence=persistence,
    )


def _payloads(urls):
    return [("url", u, None) for u in urls]


@pytest.mark.asyncio
async def test_async_image_refs_register_each_distinct_url_once():
    fake = _AssetFakeAsyncClient()
    lt = _client(fake, _FakePersistence())
    refs = await lt.aresolve_compliant_image_refs(_payloads([_REF_A, _REF_B, _REF_A]))
    assert fake.creates() == [_REF_A, _REF_B]
    # The duplicate still gets its own entry, at its original position.
    assert [r["url"] for r in refs] == [_REF_A, _REF_B, _REF_A]
    assert refs[0]["assetId"] == refs[2]["assetId"]


@pytest.mark.asyncio
async def test_async_image_refs_reuse_a_cached_registration_across_requests():
    persistence = _FakePersistence()
    first = _AssetFakeAsyncClient()
    await _client(first, persistence).aresolve_compliant_image_refs(_payloads([_REF_A]))
    assert first.creates() == [_REF_A]

    # A second generation naming the same character sheet must not re-register it.
    second = _AssetFakeAsyncClient()
    refs = await _client(second, persistence).aresolve_compliant_image_refs(_payloads([_REF_A]))
    assert second.creates() == []
    assert refs[0]["assetId"] == "asset-u1"


@pytest.mark.asyncio
async def test_async_image_refs_cache_an_exempt_registration():
    persistence = _FakePersistence()
    fake = _AssetFakeAsyncClient()
    fake.terminal = True

    class _ExemptClient(_AssetFakeAsyncClient):
        async def post(self, url, json=None, headers=None, timeout=None):
            path = url.split("api.liblib.tv", 1)[-1]
            if path == "/api/third_asset/check":
                self.calls.append((path, json))
                return _Resp({"code": 0, "data": {"list": [{"uuid": json["uuids"][0], "found": True, "status": 1}]}})
            return await super().post(url, json, headers, timeout)

    exempt = _ExemptClient()
    refs = await _client(exempt, persistence).aresolve_compliant_image_refs(_payloads([_REF_A]))
    assert refs[0]["assetId"] is None
    # "exempt" is a terminal answer and must be remembered, or exempt scenery is
    # re-submitted on every single shot.
    assert persistence.stored_calls == [(account_key_of("tok"), _REF_A, "ready-v1:image", None)]


def account_key_of(token):
    from litellm.llms.libtv.persistence import account_key

    return account_key(token)


@pytest.mark.asyncio
async def test_async_image_refs_do_not_cache_a_poll_that_never_reached_a_terminal_state():
    persistence = _FakePersistence()
    fake = _AssetFakeAsyncClient(terminal=False)
    with pytest.raises(LibTVError, match="still processing") as error:
        await _client(fake, persistence).aresolve_compliant_image_refs(_payloads([_REF_A]))
    assert error.value.status_code == 504
    assert persistence.stored_calls == []


@pytest.mark.asyncio
async def test_async_image_refs_take_one_rate_limit_slot_per_create(monkeypatch):
    acquired = []

    class _SpyLimiter:
        async def acquire(self, account_key):
            acquired.append(account_key)

    # Patch where it is looked up: client.py imports the name at module load.
    monkeypatch.setattr(_libtv_client, "get_third_asset_limiter", lambda *a, **k: _SpyLimiter())
    persistence = _FakePersistence()
    fake = _AssetFakeAsyncClient()
    await _client(fake, persistence).aresolve_compliant_image_refs(_payloads([_REF_A, _REF_B, _REF_A]))
    # Two distinct urls -> two creates -> two slots. The duplicate costs nothing.
    assert len(acquired) == 2
    assert set(acquired) == {account_key_of("tok")}


@pytest.mark.asyncio
async def test_async_video_refs_dedupe_and_cache_too():
    persistence = _FakePersistence()
    fake = _AssetFakeAsyncClient()
    await _client(fake, persistence).aresolve_compliant_video_refs(_payloads([_REF_C, _REF_C]))
    assert fake.creates() == [_REF_C]
    assert persistence.stored_calls == [(account_key_of("tok"), _REF_C, "ready-v1:video", "asset-u1")]


@pytest.mark.asyncio
async def test_incident_replay_ten_shots_ninety_seven_refs_costs_thirty_creates():
    """2026-09-01 causyn.cn: 10 shots, 97 reference images, 30 distinct.

    Before: ~97 `third_asset/create` in 12s against a 15/min cap -> every shot
    failed with code=10026. After: one create per distinct asset, once.
    """
    distinct = [f"https://libtv-res.liblib.art/upload-images/uid/{i:03d}.png" for i in range(30)]
    shots = [
        [distinct[(i * 7 + j) % 30] for j in range(sizes)]
        for i, sizes in enumerate([8, 8, 12, 8, 8, 10, 10, 8, 12, 13])
    ]
    assert sum(len(s) for s in shots) == 97

    persistence = _FakePersistence()
    total_creates = []
    for shot in shots:
        fake = _AssetFakeAsyncClient()
        await _client(fake, persistence).aresolve_compliant_image_refs(_payloads(shot))
        total_creates.extend(fake.creates())

    assert len(total_creates) == 30
    assert sorted(set(total_creates)) == sorted(set(distinct))


def test_sync_image_refs_register_each_distinct_url_once():
    class _SyncFake:
        def __init__(self):
            self.calls = []
            self._seq = 0

        def post(self, url, json=None, headers=None, timeout=None):
            path = url.split("api.liblib.tv", 1)[-1]
            self.calls.append((path, json))
            if path == "/api/community/image/verify":
                risk = __import__("json").dumps({"passed": True, "needsReview": False, "riskDescription": "正常"})
                return _Resp({"code": 0, "data": {"list": [{"url": u, "riskLabels": risk} for u in json["urlList"]]}})
            if path == "/api/third_asset/create":
                self._seq += 1
                return _Resp({"code": 0, "data": {"uuid": f"u{self._seq}"}})
            return _Resp(
                {
                    "code": 0,
                    "data": {"list": [{"uuid": json["uuids"][0], "assetId": "asset-X", "found": True, "status": 1}]},
                }
            )

        def creates(self):
            return [b["assetUrl"] for p, b in self.calls if p == "/api/third_asset/create"]

    fake = _SyncFake()
    lt = LibTVClient(token="tok", webid="wid", sync_client=fake, poll_interval=0)
    refs = lt.resolve_compliant_image_refs(_payloads([_REF_A, _REF_B, _REF_A]))
    assert fake.creates() == [_REF_A, _REF_B]
    assert [r["url"] for r in refs] == [_REF_A, _REF_B, _REF_A]


@pytest.mark.asyncio
async def test_async_image_refs_ignore_legacy_cache_without_readiness_confirmation():
    persistence = _FakePersistence(
        {(account_key_of("tok"), _REF_A, "image"): {"url": _REF_A, "assetId": "asset-pending"}}
    )
    fake = _AssetFakeAsyncClient()
    refs = await _client(fake, persistence).aresolve_compliant_image_refs(_payloads([_REF_A]))
    assert fake.creates() == [_REF_A]
    assert refs[0]["assetId"] == "asset-u1"
    assert persistence.stored_calls == [(account_key_of("tok"), _REF_A, "ready-v1:image", "asset-u1")]
