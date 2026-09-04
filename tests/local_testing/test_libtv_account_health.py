"""libtv account liveness probe.

2026-09-01 causyn.cn: seedance-2.5's two-account libtv pool had silently
degraded to one. account-2's session token was dead -- every request logged
`getUserInfo code=401 No login` -- and nothing anywhere noticed, because the
causyn health monitor only watches video_generate worker heartbeats and the
samples those workers push. A token can die while the pool still "works", so
the failure stays invisible until the surviving account hits a limit.

litellm already holds every libtv credential, so it probes and publishes a
health sample; the monitor only reads. Copying tokens into the monitor's own
Secret would put them in two places and guarantee a missed rotation.
"""

import json
import time

import pytest

from litellm.llms.libtv.account_health import (
    ACCOUNT_HEALTH_SEEN_KEY,
    LibTVAccount,
    LibTVAccountHealthProber,
    account_sample_key,
    discover_libtv_accounts,
    probe_interval_seconds,
)
from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.persistence import account_key


# ---------- discovery ----------


class _Router:
    def __init__(self, model_list):
        self._model_list = model_list

    def get_model_list(self, model_name=None):
        return self._model_list


def _deployment(dep_id, model, api_key, webid):
    return {
        "model_name": model.split("/", 1)[-1],
        "model_info": {"id": dep_id},
        "litellm_params": {"model": model, "api_key": api_key, "webid": webid},
    }


def test_discover_returns_one_account_per_distinct_credential():
    router = _Router(
        [
            _deployment("libtv-seedance-2-5-account-1", "libtv/star-video2.5", "tok-1", "web-1"),
            _deployment("libtv-seedance-2-5-account-2", "libtv/star-video2.5", "tok-2", "web-2"),
            # Same account reused by another model group: one probe, not two.
            _deployment("libtv-seedance-2-mini-account-1", "libtv/star-video2-fast", "tok-1", "web-1"),
        ]
    )
    accounts = discover_libtv_accounts(router)
    assert [a.account_key for a in accounts] == [account_key("tok-1"), account_key("tok-2")]
    assert accounts[0].label == "libtv-seedance-2-5-account-1"


def test_discover_resolves_os_environ_indirection(monkeypatch):
    monkeypatch.setenv("LIBTV_TOKEN_X", "real-token")
    monkeypatch.setenv("LIBTV_WEBID_X", "real-webid")
    router = _Router(
        [_deployment("d1", "libtv/star-video2.5", "os.environ/LIBTV_TOKEN_X", "os.environ/LIBTV_WEBID_X")]
    )
    accounts = discover_libtv_accounts(router)
    assert accounts[0].token == "real-token"
    assert accounts[0].webid == "real-webid"


def test_discover_skips_non_libtv_and_unresolvable_deployments(monkeypatch):
    monkeypatch.delenv("LIBTV_MISSING", raising=False)
    router = _Router(
        [
            _deployment("d1", "wavespeed/gpt-image-2", "tok-1", "web-1"),
            _deployment("d2", "libtv/star-video2.5", "os.environ/LIBTV_MISSING", "web-2"),
            _deployment("d3", "libtv/star-video2.5", "tok-3", ""),
            _deployment("d4", "libtv/star-video2.5", "tok-4", "web-4"),
        ]
    )
    assert [a.label for a in discover_libtv_accounts(router)] == ["d4"]


def test_discover_tolerates_a_missing_router():
    assert discover_libtv_accounts(None) == []
    assert discover_libtv_accounts(object()) == []


# ---------- probing ----------


class _Resp:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _ProbeClient:
    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc
        self.calls = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, headers))
        if self.exc is not None:
            raise self.exc
        return _Resp(self.payload)


_ACCOUNT = LibTVAccount(label="acct-1", token="tok-1", webid="web-1", account_key=account_key("tok-1"))


@pytest.mark.asyncio
async def test_probe_reports_healthy_when_getuserinfo_returns_a_uuid():
    client = _ProbeClient(payload={"code": 0, "data": {"uuid": "user-uuid"}})
    prober = LibTVAccountHealthProber(redis_client=None, http_client_factory=lambda: client)
    sample = await prober.probe(_ACCOUNT)
    assert sample["status"] == "healthy"
    assert sample["reason"] is None
    assert sample["label"] == "acct-1"
    assert sample["probe"] == "getUserInfo"
    url, headers = client.calls[0]
    assert url.endswith("/api/www/user/getUserInfo")
    assert headers["token"] == "tok-1"
    assert headers["webid"] == "web-1"


@pytest.mark.asyncio
async def test_probe_reports_unhealthy_on_the_dead_token_signature():
    # The exact 2026-09-01 signature.
    client = _ProbeClient(payload={"code": 401, "msg": "No login"})
    prober = LibTVAccountHealthProber(redis_client=None, http_client_factory=lambda: client)
    sample = await prober.probe(_ACCOUNT)
    assert sample["status"] == "unhealthy"
    assert "401" in sample["reason"]
    assert "No login" in sample["reason"]


@pytest.mark.asyncio
async def test_probe_reports_unhealthy_when_the_payload_carries_no_uuid():
    client = _ProbeClient(payload={"code": 0, "data": {}})
    prober = LibTVAccountHealthProber(redis_client=None, http_client_factory=lambda: client)
    assert (await prober.probe(_ACCOUNT))["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_probe_reports_unknown_when_the_probe_itself_could_not_run():
    # A transport fault says nothing about the account; claiming "unhealthy"
    # would page someone about our own network.
    client = _ProbeClient(exc=OSError("connection reset"))
    prober = LibTVAccountHealthProber(redis_client=None, http_client_factory=lambda: client)
    sample = await prober.probe(_ACCOUNT)
    assert sample["status"] == "unknown"
    assert "connection reset" in sample["reason"]


@pytest.mark.asyncio
async def test_probe_never_leaks_the_token_into_the_sample():
    client = _ProbeClient(exc=LibTVError(status_code=401, message="libtv getUserInfo failed for tok-1"))
    prober = LibTVAccountHealthProber(redis_client=None, http_client_factory=lambda: client)
    sample = await prober.probe(_ACCOUNT)
    assert "tok-1" not in json.dumps(sample)


# ---------- publishing ----------


class _FakeRedis:
    def __init__(self):
        self.strings = {}
        self.ttls = {}
        self.zsets = {}
        self.claims = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.claims:
            return None
        if nx:
            self.claims[key] = value
            return True
        self.strings[key] = value
        self.ttls[key] = ex
        return True

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    async def zrange(self, key, start, stop, withscores=False):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        return items if withscores else [k for k, _ in items]

    async def zrem(self, key, *members):
        bucket = self.zsets.get(key, {})
        removed = 0
        for member in members:
            removed += bucket.pop(member, None) is not None
        return removed


@pytest.mark.asyncio
async def test_publish_writes_a_sample_and_registers_the_account():
    redis = _FakeRedis()
    client = _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}})
    prober = LibTVAccountHealthProber(redis_client=redis, http_client_factory=lambda: client)
    await prober.probe_and_publish(_ACCOUNT)
    stored = json.loads(redis.strings[account_sample_key(_ACCOUNT.account_key)])
    assert stored["status"] == "healthy"
    assert stored["received_at"] > 0
    # The monitor discovers accounts from this registry instead of being told.
    assert _ACCOUNT.account_key in redis.zsets[ACCOUNT_HEALTH_SEEN_KEY]


@pytest.mark.asyncio
async def test_sample_expires_so_a_stopped_prober_reads_as_missing_not_healthy():
    redis = _FakeRedis()
    client = _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}})
    prober = LibTVAccountHealthProber(
        redis_client=redis, http_client_factory=lambda: client, interval_seconds=60
    )
    await prober.probe_and_publish(_ACCOUNT)
    ttl = redis.ttls[account_sample_key(_ACCOUNT.account_key)]
    assert ttl is not None and ttl > 60


@pytest.mark.asyncio
async def test_only_one_worker_probes_an_account_per_interval():
    redis = _FakeRedis()
    calls = []

    def factory():
        client = _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}})
        calls.append(client)
        return client

    kwargs = dict(redis_client=redis, http_client_factory=factory, interval_seconds=60)
    assert await LibTVAccountHealthProber(**kwargs).probe_and_publish(_ACCOUNT) is True
    # A second uvicorn worker in the same window must not double-probe.
    assert await LibTVAccountHealthProber(**kwargs).probe_and_publish(_ACCOUNT) is False
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_publish_without_redis_still_probes_and_does_not_raise():
    client = _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}})
    prober = LibTVAccountHealthProber(redis_client=None, http_client_factory=lambda: client)
    assert await prober.probe_and_publish(_ACCOUNT) is True


@pytest.mark.asyncio
async def test_a_redis_fault_does_not_escape_the_prober():
    class _BrokenRedis(_FakeRedis):
        async def set(self, *a, **k):
            raise RuntimeError("redis down")

    client = _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}})
    prober = LibTVAccountHealthProber(redis_client=_BrokenRedis(), http_client_factory=lambda: client)
    await prober.probe_and_publish(_ACCOUNT)  # must not raise


@pytest.mark.asyncio
async def test_run_once_probes_every_discovered_account():
    redis = _FakeRedis()
    router = _Router(
        [
            _deployment("acct-1", "libtv/star-video2.5", "tok-1", "web-1"),
            _deployment("acct-2", "libtv/star-video2.5", "tok-2", "web-2"),
        ]
    )
    prober = LibTVAccountHealthProber(
        redis_client=redis,
        http_client_factory=lambda: _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}}),
        router=router,
    )
    await prober.run_once()
    assert set(redis.zsets[ACCOUNT_HEALTH_SEEN_KEY]) == {account_key("tok-1"), account_key("tok-2")}


@pytest.mark.asyncio
async def test_run_once_keeps_going_after_one_account_blows_up():
    redis = _FakeRedis()
    router = _Router(
        [
            _deployment("acct-1", "libtv/star-video2.5", "tok-1", "web-1"),
            _deployment("acct-2", "libtv/star-video2.5", "tok-2", "web-2"),
        ]
    )

    def factory():
        return _ProbeClient(exc=RuntimeError("boom"))

    prober = LibTVAccountHealthProber(redis_client=redis, http_client_factory=factory, router=router)
    await prober.run_once()
    assert len(redis.zsets[ACCOUNT_HEALTH_SEEN_KEY]) == 2


# ---------- config ----------


def test_probe_interval_default_and_override(monkeypatch):
    monkeypatch.delenv("LIBTV_ACCOUNT_HEALTH_INTERVAL_SECONDS", raising=False)
    assert probe_interval_seconds() == 60.0
    monkeypatch.setenv("LIBTV_ACCOUNT_HEALTH_INTERVAL_SECONDS", "30")
    assert probe_interval_seconds() == 30.0
    monkeypatch.setenv("LIBTV_ACCOUNT_HEALTH_INTERVAL_SECONDS", "nonsense")
    assert probe_interval_seconds() == 60.0


def test_sample_key_is_namespaced_per_account():
    assert account_sample_key("abc") == "libtv:account-health:sample:abc"


# ---------- startup hook ----------


@pytest.mark.asyncio
async def test_startup_hook_is_a_noop_when_disabled(monkeypatch):
    from litellm.llms.libtv import account_health

    monkeypatch.setenv("LIBTV_ACCOUNT_HEALTH_ENABLED", "false")
    router = _Router([_deployment("acct-1", "libtv/star-video2.5", "tok-1", "web-1")])
    assert await account_health.start_libtv_account_health_prober(router) is None


@pytest.mark.asyncio
async def test_startup_hook_is_a_noop_without_any_libtv_account(monkeypatch):
    from litellm.llms.libtv import account_health

    monkeypatch.delenv("LIBTV_ACCOUNT_HEALTH_ENABLED", raising=False)
    router = _Router([_deployment("d1", "wavespeed/gpt-image-2", "k", "w")])
    assert await account_health.start_libtv_account_health_prober(router) is None


@pytest.mark.asyncio
async def test_startup_hook_declines_to_run_without_redis(monkeypatch):
    from litellm.llms.libtv import account_health

    monkeypatch.delenv("LIBTV_ACCOUNT_HEALTH_ENABLED", raising=False)
    monkeypatch.setattr(
        "litellm.llms.libtv.transfer.get_transfer_redis", lambda *a, **k: None
    )
    router = _Router([_deployment("acct-1", "libtv/star-video2.5", "tok-1", "web-1")])
    # Samples with nowhere to go would make the monitor read every account as
    # missing -- a monitor that alarms on our own misconfiguration is worse than
    # one that plainly is not running.
    assert await account_health.start_libtv_account_health_prober(router) is None


@pytest.mark.asyncio
async def test_startup_hook_starts_a_running_prober(monkeypatch):
    from litellm.llms.libtv import account_health

    monkeypatch.delenv("LIBTV_ACCOUNT_HEALTH_ENABLED", raising=False)
    redis = _FakeRedis()
    monkeypatch.setattr("litellm.llms.libtv.transfer.get_transfer_redis", lambda *a, **k: redis)
    # No real dial to passport.liblib.art from a unit test.
    monkeypatch.setattr(
        account_health,
        "_default_http_client_factory",
        lambda: _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}}),
    )
    router = _Router([_deployment("acct-1", "libtv/star-video2.5", "tok-1", "web-1")])
    prober = await account_health.start_libtv_account_health_prober(router)
    assert prober is not None
    try:
        for _ in range(50):
            if redis.zsets.get(ACCOUNT_HEALTH_SEEN_KEY):
                break
            await __import__("asyncio").sleep(0.01)
        assert account_key("tok-1") in redis.zsets.get(ACCOUNT_HEALTH_SEEN_KEY, {})
    finally:
        await prober.stop()


@pytest.mark.asyncio
async def test_prober_enabled_respects_explicit_off_values(monkeypatch):
    from litellm.llms.libtv.account_health import prober_enabled

    for value in ("0", "false", "FALSE", "no", "off"):
        monkeypatch.setenv("LIBTV_ACCOUNT_HEALTH_ENABLED", value)
        assert prober_enabled() is False
    for value in ("1", "true", "yes", "on", ""):
        monkeypatch.setenv("LIBTV_ACCOUNT_HEALTH_ENABLED", value)
        assert prober_enabled() is True


# ---------- registry pruning ----------


@pytest.mark.asyncio
async def test_run_once_prunes_accounts_that_no_longer_exist():
    """Rotating a token changes its account_key.

    Without pruning, the retired key stays in the registry forever, its sample
    expires, and the monitor reports a permanently missing account -- so the
    very act of FIXING the 2026-09-01 dead token would have created a
    permanent false alarm.
    """
    redis = _FakeRedis()
    redis.zsets[ACCOUNT_HEALTH_SEEN_KEY] = {"retired-key": 1.0}
    router = _Router([_deployment("acct-1", "libtv/star-video2.5", "tok-1", "web-1")])
    prober = LibTVAccountHealthProber(
        redis_client=redis,
        http_client_factory=lambda: _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}}),
        router=router,
    )
    await prober.run_once()
    assert set(redis.zsets[ACCOUNT_HEALTH_SEEN_KEY]) == {account_key("tok-1")}


@pytest.mark.asyncio
async def test_run_once_does_not_prune_when_discovery_comes_back_empty():
    # An empty discovery is far more likely to be a router that has not loaded
    # than a pool that genuinely vanished. Wiping the registry there would hide
    # a real outage precisely when visibility matters most.
    redis = _FakeRedis()
    redis.zsets[ACCOUNT_HEALTH_SEEN_KEY] = {"k1": 1.0}
    prober = LibTVAccountHealthProber(
        redis_client=redis,
        http_client_factory=lambda: _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}}),
        router=_Router([]),
    )
    await prober.run_once()
    assert set(redis.zsets[ACCOUNT_HEALTH_SEEN_KEY]) == {"k1"}


@pytest.mark.asyncio
async def test_pruning_failure_does_not_break_the_cycle():
    class _NoZrem(_FakeRedis):
        async def zrem(self, *a, **k):
            raise RuntimeError("redis down")

    redis = _NoZrem()
    redis.zsets[ACCOUNT_HEALTH_SEEN_KEY] = {"retired-key": 1.0}
    router = _Router([_deployment("acct-1", "libtv/star-video2.5", "tok-1", "web-1")])
    prober = LibTVAccountHealthProber(
        redis_client=redis,
        http_client_factory=lambda: _ProbeClient(payload={"code": 0, "data": {"uuid": "u"}}),
        router=router,
    )
    await prober.run_once()  # must not raise
    assert account_key("tok-1") in redis.zsets[ACCOUNT_HEALTH_SEEN_KEY]
