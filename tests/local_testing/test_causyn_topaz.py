from __future__ import annotations

import asyncio
import json
import logging

import pytest
from pydantic import TypeAdapter

from litellm.llms.causyn.topaz import (
    MAX_ATTEMPTS,
    MAX_FAILURE_MESSAGE_CHARS,
    TopazAccount,
    TopazAdvance,
    TopazAccountPool,
    TopazClient,
    TopazIndeterminateError,
    TopazRedis,
    TopazState,
    TopazVideoAdapter,
)
from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.image_upscale import ProviderTransportError
from litellm.llms.libtv.image_upscale import ProviderRejected
from litellm.llms.libtv.persistence import LibTVPersistence


def _state(raw: str) -> dict[str, object]:
    return TypeAdapter(dict[str, object]).validate_json(raw)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.cas_fail = False
        self.cas_barrier: asyncio.Event | None = None
        self.cas_waiters = 0

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int, nx: bool = False) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        self.ttls[key] = ex
        return True

    async def eval(self, script: str, numkeys: int, *args: object) -> object:
        if not script or numkeys != 1:
            return 0
        if self.cas_fail or len(args) != 7 or not isinstance(args[0], str):
            return 0
        key = args[0]
        if self.cas_barrier is not None and self.cas_waiters < 2:
            self.cas_waiters += 1
            if self.cas_waiters == 2:
                self.cas_barrier.set()
            await self.cas_barrier.wait()
        current = self.values.get(key)
        value = args[5]
        if current is None or not isinstance(value, str):
            return 0
        try:
            current_state = _state(current)
            expected_phase = args[1]
            expected_attempt = int(str(args[2]))
            expected_account = args[3]
            expected_updated = float(str(args[4]))
            ttl = int(str(args[6]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0
        if (
            current_state.get("phase") != expected_phase
            or current_state.get("attempt_index") != expected_attempt
            or current_state.get("account_ref") != expected_account
            or float(str(current_state.get("updated_at"))) != expected_updated
        ):
            return 0
        self.values[key] = value
        self.ttls[key] = ttl
        return 1


class FakeClient:
    def __init__(
        self,
        outcomes: list[object],
        *,
        import_outcomes: list[object] | None = None,
        poll_outcomes: list[dict[str, object]] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.import_outcomes = import_outcomes or []
        self.poll_outcomes = poll_outcomes or []
        self.create_calls: list[dict[str, object]] = []
        self.import_calls: list[str] = []

    async def aensure_libtv_url(
        self,
        kind: str,
        url: str,
        data: bytes | None,
        default_name: str,
        *,
        allow_cache: bool = True,
    ) -> str:
        assert kind == "url"
        assert data is None
        assert allow_cache is False
        self.import_calls.append(url)
        if self.import_outcomes:
            outcome = self.import_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        return f"https://libtv.example/{default_name}"

    async def acreate(
        self,
        model_key: str,
        vendor: str,
        task_type: str,
        params: dict[str, object],
        project_name: str,
        *,
        allow_cached_project_retry: bool,
        paid_submission: bool,
    ) -> dict[str, object]:
        assert (model_key, vendor, task_type, project_name) == (
            "topaz-video-upscaler",
            "topazlabs",
            "video",
            "causyn-2k",
        )
        assert allow_cached_project_retry is False
        assert paid_submission is True
        self.create_calls.append(params)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"task_id": str(outcome)}

    async def apoll_once(self, task_id: str, task_type: str) -> dict[str, object]:
        assert task_type == "video"
        if self.poll_outcomes:
            return self.poll_outcomes.pop(0)
        return {"status": 2, "urls": [f"https://libtv.example/{task_id}.mp4"]}

    async def afetch_content(self, url: str) -> bytes:
        return url.encode()


def _as_topaz_client(client: FakeClient) -> TopazClient:
    return client


def _as_topaz_redis(redis: FakeRedis) -> TopazRedis:
    return redis


@pytest.mark.asyncio
async def test_topaz_adapter_uses_fixed_paid_video_create_and_records_account_ref() -> None:
    redis = FakeRedis()
    client = FakeClient(["topaz-1"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: _as_topaz_client(client),
        redis_factory=lambda: _as_topaz_redis(redis),
        now=lambda: 100.0,
        state_ttl=3600,
    )
    result = await adapter.advance("base-1", lambda _task_id: _source_url(), validate_source=lambda _: None)

    assert result.status == "completed"
    assert client.import_calls == ["https://platform.example/source.mp4"]
    assert client.create_calls == [
        {
            "prompt": "",
            "count": 1,
            "textList": [],
            "imageList": [],
            "videoList": ["https://libtv.example/source.mp4"],
            "audioList": [],
            "resolution": "2K",
            "specifiedModel": "prob-4",
            "slowmo": "1",
        }
    ]
    state = _state(redis.values["causyn:topaz:base-1"])
    assert state["account_ref"] == "account-1"
    assert state["confirmed_submission_count"] == 1
    assert state["lease_until"] == 0
    serialized = redis.values["causyn:topaz:base-1"]
    assert all(secret not in serialized for secret in ("token", "webid", "source.mp4", "result.mp4"))
    assert "result_url" not in state
    assert redis.ttls["causyn:topaz:base-1"] == 3600


@pytest.mark.asyncio
async def test_topaz_adapter_does_not_retry_after_paid_boundary_transport_error() -> None:
    redis = FakeRedis()
    client = FakeClient([ProviderTransportError("lost", crossed_create_boundary=True)])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    with pytest.raises(TopazIndeterminateError):
        await adapter.advance("base-2", lambda _task_id: _source_url(), validate_source=lambda _: None)
    with pytest.raises(TopazIndeterminateError):
        await adapter.advance("base-2", lambda _task_id: _source_url(), validate_source=lambda _: None)

    state = _state(redis.values["causyn:topaz:base-2"])
    assert state["phase"] == "indeterminate"
    assert len(client.create_calls) == 1


@pytest.mark.asyncio
async def test_topaz_adapter_caps_safe_failures_at_four_attempts_and_rotates_accounts() -> None:
    redis = FakeRedis()
    clients = {
        "account-1": FakeClient([ProviderTransportError("429", crossed_create_boundary=False)] * 2),
        "account-2": FakeClient([ProviderTransportError("429", crossed_create_boundary=False)] * 2),
    }
    accounts = TopazAccountPool(
        (
            TopazAccount("account-1", "token-1", "webid-1"),
            TopazAccount("account-2", "token-2", "webid-2"),
        )
    )
    adapter = TopazVideoAdapter(
        accounts,
        client_factory=lambda account: clients[account.account_ref],
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    for index in range(MAX_ATTEMPTS):
        result = await adapter.advance("base-3", lambda _task_id: _source_url(), validate_source=lambda _: None)
        assert result.status == ("failed" if index == MAX_ATTEMPTS - 1 else "in_progress")

    assert [len(clients[ref].create_calls) for ref in ("account-1", "account-2")] == [2, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["import", "rejected", "terminal"])
async def test_safe_failure_rotates_account_and_never_exceeds_four_attempts(failure_kind: str) -> None:
    redis = FakeRedis()
    account_one = FakeClient(
        [
            ProviderRejected("rejected") if failure_kind == "rejected" else "topaz-1",
        ],
        import_outcomes=[RuntimeError("import failed")] if failure_kind == "import" else None,
        poll_outcomes=[{"status": 3, "failed_reason": "provider failed"}] if failure_kind == "terminal" else None,
    )
    account_two = FakeClient(["topaz-2"] * MAX_ATTEMPTS)
    clients = {"account-1": account_one, "account-2": account_two}
    adapter = TopazVideoAdapter(
        TopazAccountPool(
            (
                TopazAccount("account-1", "token-1", "webid-1"),
                TopazAccount("account-2", "token-2", "webid-2"),
            )
        ),
        client_factory=lambda account: clients[account.account_ref],
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    first = await adapter.advance("base-safe-failure", lambda _task_id: _source_url(), validate_source=lambda _: None)
    second = await adapter.advance("base-safe-failure", lambda _task_id: _source_url(), validate_source=lambda _: None)

    assert first.status == "in_progress"
    assert second.status == "completed"
    state = _state(redis.values["causyn:topaz:base-safe-failure"])
    assert state["attempt_index"] == 2
    assert state["confirmed_submission_count"] == (1 if failure_kind != "terminal" else 2)
    assert state["account_ref"] == "account-2"

    for _ in range(MAX_ATTEMPTS - 2):
        await adapter.advance("base-safe-failure", lambda _task_id: _source_url(), validate_source=lambda _: None)
    total_creates = len(account_one.create_calls) + len(account_two.create_calls)
    assert total_creates <= MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_paid_create_generic_exception_is_indeterminate_and_never_recreated() -> None:
    redis = FakeRedis()
    client = FakeClient([RuntimeError("unknown result")])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    with pytest.raises(TopazIndeterminateError):
        await adapter.advance("base-generic", lambda _task_id: _source_url(), validate_source=lambda _: None)
    with pytest.raises(TopazIndeterminateError):
        await adapter.advance("base-generic", lambda _task_id: _source_url(), validate_source=lambda _: None)

    assert len(client.create_calls) == 1
    assert _state(redis.values["causyn:topaz:base-generic"])["phase"] == "indeterminate"


@pytest.mark.asyncio
async def test_indeterminate_status_polls_warn_each_time_without_resubmission(caplog: pytest.LogCaptureFixture) -> None:
    redis = FakeRedis()
    client = FakeClient([ProviderTransportError("lost", crossed_create_boundary=True)])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    with caplog.at_level("WARNING"):
        for _ in range(3):
            with pytest.raises(TopazIndeterminateError):
                await adapter.advance(
                    "base-indeterminate", lambda _task_id: _source_url(), validate_source=lambda _: None
                )

    assert len(client.create_calls) == 1
    assert sum("state is indeterminate" in record.message for record in caplog.records) == 3


@pytest.mark.asyncio
async def test_stale_preparing_uses_persisted_account_ref_and_fails_closed_on_config_change() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-stale"] = TopazState(
        target_resolution="2k",
        attempt_index=1,
        confirmed_submission_count=0,
        provider_task_id=None,
        account_ref="account-2",
        phase="preparing",
        updated_at=1.0,
        lease_until=2.0,
    ).to_json()
    source_calls: list[str] = []
    client = FakeClient(["topaz-stale"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    async def source_url(task_id: str) -> str:
        source_calls.append(task_id)
        return await _source_url()

    with pytest.raises(TopazIndeterminateError):
        await adapter.advance("base-stale", source_url, validate_source=lambda _: None)

    assert source_calls == []
    assert client.create_calls == []
    assert _state(redis.values["causyn:topaz:base-stale"])["phase"] == "indeterminate"


@pytest.mark.asyncio
async def test_unexpired_preparing_lease_is_not_taken_over() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-active"] = TopazState(
        target_resolution="2k",
        attempt_index=1,
        confirmed_submission_count=0,
        provider_task_id=None,
        account_ref="account-1",
        phase="preparing",
        updated_at=99.0,
        lease_until=101.0,
    ).to_json()
    client = FakeClient(["topaz-active"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    result = await adapter.advance("base-active", lambda _task_id: _source_url(), validate_source=lambda _: None)

    assert result.status == "in_progress"
    assert client.create_calls == []


@pytest.mark.asyncio
async def test_stale_creating_lease_is_indeterminate_without_a_second_paid_create() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-creating"] = TopazState(
        target_resolution="2k",
        attempt_index=1,
        confirmed_submission_count=0,
        provider_task_id=None,
        account_ref="account-1",
        phase="creating",
        updated_at=1.0,
        lease_until=2.0,
    ).to_json()
    client = FakeClient(["topaz-creating"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    with pytest.raises(TopazIndeterminateError):
        await adapter.advance("base-creating", lambda _task_id: _source_url(), validate_source=lambda _: None)

    assert client.create_calls == []
    assert _state(redis.values["causyn:topaz:base-creating"])["phase"] == "indeterminate"


@pytest.mark.asyncio
async def test_corrupt_state_is_fail_closed_and_does_not_create() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-corrupt"] = "{not-json"
    client = FakeClient(["topaz-corrupt"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    with pytest.raises(TopazIndeterminateError):
        await adapter.advance("base-corrupt", lambda _task_id: _source_url(), validate_source=lambda _: None)

    assert client.create_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_overrides",
    [
        {"phase": "submitted", "provider_task_id": None},
        {"phase": "completed", "provider_task_id": None},
        {"phase": "submitted", "provider_task_id": "topaz-1", "confirmed_submission_count": 0},
        {"phase": "submitted", "confirmed_submission_count": 2},
        {"phase": "preparing", "lease_until": 200.0, "confirmed_submission_count": 1, "provider_task_id": None},
        {"phase": "creating", "lease_until": 200.0, "confirmed_submission_count": 1, "provider_task_id": None},
        {
            "phase": "retryable_failure",
            "failure_message": "provider_failure",
            "confirmed_submission_count": 1,
            "provider_task_id": None,
        },
        {
            "phase": "failed",
            "failure_message": "provider_failure",
            "confirmed_submission_count": 1,
            "provider_task_id": None,
        },
        {"phase": "preparing", "lease_until": 0},
        {"phase": "retryable_failure", "failure_message": None},
    ],
)
async def test_semantically_corrupt_state_is_fail_closed_without_create(
    state_overrides: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    redis = FakeRedis()
    state: dict[str, object] = {
        "target_resolution": "2k",
        "attempt_index": 1,
        "confirmed_submission_count": 1,
        "provider_task_id": "topaz-1",
        "account_ref": "account-1",
        "phase": "submitted",
        "updated_at": 100.0,
        "lease_until": 0,
        "failure_message": None,
    }
    state.update(state_overrides)
    redis.values["causyn:topaz:base-semantic-corrupt"] = json.dumps(state)
    client = FakeClient(["topaz-semantic-corrupt"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(TopazIndeterminateError):
            await adapter.advance(
                "base-semantic-corrupt", lambda _task_id: _source_url(), validate_source=lambda _: None
            )

    assert client.create_calls == []
    assert any(record.message == "causyn Topaz state is corrupt" for record in caplog.records)


@pytest.mark.asyncio
async def test_terminal_provider_failure_preserves_submission_evidence_before_next_attempt() -> None:
    redis = FakeRedis()
    first_client = FakeClient(["topaz-1"], poll_outcomes=[{"status": 3, "failed_reason": "provider failed"}])
    second_client = FakeClient(["topaz-2"])
    clients = {"account-1": first_client, "account-2": second_client}
    adapter = TopazVideoAdapter(
        TopazAccountPool(
            (
                TopazAccount("account-1", "token-1", "webid-1"),
                TopazAccount("account-2", "token-2", "webid-2"),
            )
        ),
        client_factory=lambda account: clients[account.account_ref],
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    first = await adapter.advance(
        "base-terminal-roundtrip", lambda _task_id: _source_url(), validate_source=lambda _: None
    )

    assert first.status == "in_progress"
    retryable = _state(redis.values["causyn:topaz:base-terminal-roundtrip"])
    assert retryable == {
        "account_ref": "account-1",
        "attempt_index": 1,
        "confirmed_submission_count": 1,
        # The provider's own poll `failed_reason`, kept verbatim. This used to
        # assert the "provider_failure" placeholder, i.e. it pinned the very
        # discard that made the 2026-09-01 production failure undiagnosable.
        "failure_message": "provider failed",
        "lease_until": 0,
        "phase": "retryable_failure",
        "provider_task_id": "topaz-1",
        "target_resolution": "2k",
        "updated_at": 100.0,
    }

    second = await adapter.advance(
        "base-terminal-roundtrip", lambda _task_id: _source_url(), validate_source=lambda _: None
    )

    assert second.status == "completed"
    completed = _state(redis.values["causyn:topaz:base-terminal-roundtrip"])
    assert completed["attempt_index"] == 2
    assert completed["confirmed_submission_count"] == 2
    assert completed["account_ref"] == "account-2"
    assert completed["provider_task_id"] == "topaz-2"
    assert first_client.create_calls == [
        {
            "prompt": "",
            "count": 1,
            "textList": [],
            "imageList": [],
            "videoList": ["https://libtv.example/source.mp4"],
            "audioList": [],
            "resolution": "2K",
            "specifiedModel": "prob-4",
            "slowmo": "1",
        }
    ]
    assert second_client.create_calls == [
        {
            "prompt": "",
            "count": 1,
            "textList": [],
            "imageList": [],
            "videoList": ["https://libtv.example/source.mp4"],
            "audioList": [],
            "resolution": "2K",
            "specifiedModel": "prob-4",
            "slowmo": "1",
        }
    ]


@pytest.mark.asyncio
async def test_cas_failure_reloads_current_state_instead_of_reporting_lost_terminal_result() -> None:
    redis = FakeRedis()
    client = FakeClient([ProviderTransportError("429", crossed_create_boundary=False)])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )
    assert (
        await adapter.advance("base-cas", lambda _task_id: _source_url(), validate_source=lambda _: None)
    ).status == "in_progress"
    redis.cas_fail = True
    result = await adapter.advance("base-cas", lambda _task_id: _source_url(), validate_source=lambda _: None)
    assert result.status == "in_progress"
    assert _state(redis.values["causyn:topaz:base-cas"])["phase"] == "retryable_failure"


@pytest.mark.asyncio
async def test_each_retry_gets_a_fresh_source_url() -> None:
    redis = FakeRedis()
    client = FakeClient([ProviderTransportError("429", crossed_create_boundary=False)] * MAX_ATTEMPTS)
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )
    urls = [f"https://platform.example/source-{index}.mp4" for index in range(MAX_ATTEMPTS)]

    async def source_url(_task_id: str) -> str:
        return urls.pop(0)

    for _ in range(MAX_ATTEMPTS):
        await adapter.advance("base-fresh", source_url, validate_source=lambda url: assert_source(url))

    assert client.import_calls == [
        "https://platform.example/source-0.mp4",
        "https://platform.example/source-1.mp4",
        "https://platform.example/source-2.mp4",
        "https://platform.example/source-3.mp4",
    ]


@pytest.mark.asyncio
async def test_concurrent_initial_claim_only_imports_and_creates_once() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-race-initial"] = TopazState(
        target_resolution="2k",
        attempt_index=1,
        confirmed_submission_count=0,
        provider_task_id=None,
        account_ref="account-1",
        phase="preparing",
        updated_at=1.0,
        lease_until=2.0,
    ).to_json()
    redis.cas_barrier = asyncio.Event()
    client = FakeClient(["topaz-race-initial"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    results = await asyncio.gather(
        adapter.advance("base-race-initial", lambda _task_id: _source_url(), validate_source=lambda _: None),
        adapter.advance("base-race-initial", lambda _task_id: _source_url(), validate_source=lambda _: None),
    )

    assert [result.status for result in results] == ["completed", "completed"]
    assert len(client.import_calls) == 1
    assert len(client.create_calls) == 1
    assert _state(redis.values["causyn:topaz:base-race-initial"])["confirmed_submission_count"] == 1


@pytest.mark.asyncio
async def test_concurrent_submitted_polls_reload_completed_cas_winner() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-race-completed"] = TopazState(
        target_resolution="2k",
        attempt_index=1,
        confirmed_submission_count=1,
        provider_task_id="topaz-existing",
        account_ref="account-1",
        phase="submitted",
        updated_at=1.0,
        lease_until=0,
    ).to_json()
    redis.cas_barrier = asyncio.Event()
    client = FakeClient([])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    results = await asyncio.gather(
        adapter.advance("base-race-completed", lambda _task_id: _source_url(), validate_source=lambda _: None),
        adapter.advance("base-race-completed", lambda _task_id: _source_url(), validate_source=lambda _: None),
    )

    assert [result.status for result in results] == ["completed", "completed"]
    state = _state(redis.values["causyn:topaz:base-race-completed"])
    assert state["phase"] == "completed"
    assert state["provider_task_id"] == "topaz-existing"
    assert client.create_calls == []


@pytest.mark.asyncio
async def test_concurrent_next_attempt_claim_has_one_creator_and_loser_reloads_preparing() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-race-next"] = TopazState(
        target_resolution="2k",
        attempt_index=1,
        confirmed_submission_count=0,
        provider_task_id=None,
        account_ref="account-1",
        phase="retryable_failure",
        updated_at=1.0,
        lease_until=0,
        failure_message="provider_failure",
    ).to_json()
    redis.cas_barrier = asyncio.Event()
    client = FakeClient(["topaz-race-next"])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token", "webid"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    results = await asyncio.gather(
        adapter.advance("base-race-next", lambda _task_id: _source_url(), validate_source=lambda _: None),
        adapter.advance("base-race-next", lambda _task_id: _source_url(), validate_source=lambda _: None),
    )

    assert [result.status for result in results] == ["completed", "completed"]
    assert len(client.create_calls) == 1
    assert _state(redis.values["causyn:topaz:base-race-next"])["attempt_index"] == 2


@pytest.mark.asyncio
async def test_concurrent_content_uses_persisted_account_ref() -> None:
    redis = FakeRedis()
    redis.values["causyn:topaz:base-race-content"] = TopazState(
        target_resolution="2k",
        attempt_index=2,
        confirmed_submission_count=1,
        provider_task_id="topaz-content",
        account_ref="account-2",
        phase="completed",
        updated_at=1.0,
        lease_until=0,
    ).to_json()
    client = FakeClient([])
    account_refs: list[str] = []

    def client_factory(account: TopazAccount) -> FakeClient:
        account_refs.append(account.account_ref)
        return client

    async def content_getter(_client: TopazClient, url: str) -> bytes:
        assert url == "https://libtv.example/topaz-content.mp4"
        return b"content"

    adapter = TopazVideoAdapter(
        TopazAccountPool(
            (
                TopazAccount("account-1", "token-1", "webid-1"),
                TopazAccount("account-2", "token-2", "webid-2"),
            )
        ),
        client_factory=client_factory,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
        content_getter=content_getter,
    )

    contents = await asyncio.gather(adapter.content("base-race-content"), adapter.content("base-race-content"))

    assert contents == [b"content", b"content"]
    assert account_refs == ["account-2"]


@pytest.mark.asyncio
async def test_libtv_client_topaz_import_bypasses_upload_cache_for_each_fresh_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Persistence(LibTVPersistence):
        def __init__(self) -> None:
            super().__init__(None)
            self.lookup_calls = 0
            self.store_calls = 0

        async def cached_upload(self, account_key: str, source_key: str) -> str:
            self.lookup_calls += 1
            return "https://libtv-res.liblib.art/old-cached-source"

        async def store_upload(self, account_key: str, source_key: str, cdn_url: str, size_bytes: int) -> None:
            self.store_calls += 1

    persistence = Persistence()

    async def cached_url_alive(_url: str) -> bool:
        return True

    fetched: list[str] = []
    uploads: list[bytes] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return url.encode()

    async def upload(buffer: bytes, filename: str) -> str:
        uploads.append(buffer)
        return f"https://libtv-res.liblib.art/new-{len(uploads)}"

    monkeypatch.setattr("litellm.llms.libtv.client.url_alive", cached_url_alive)
    client = LibTVClient(token="token", webid="webid", http_get=fetch, persistence=persistence)
    monkeypatch.setattr(client, "aupload_media", upload)

    first = await client.aensure_libtv_url(
        "url", "https://platform.example/source-1.mp4?signature=one", None, "source.mp4", allow_cache=False
    )
    second = await client.aensure_libtv_url(
        "url", "https://platform.example/source-2.mp4?signature=two", None, "source.mp4", allow_cache=False
    )
    cached = await client.aensure_libtv_url(
        "url", "https://platform.example/source-3.mp4?signature=three", None, "source.mp4", allow_cache=True
    )

    assert (first, second, cached) == (
        "https://libtv-res.liblib.art/new-1",
        "https://libtv-res.liblib.art/new-2",
        "https://libtv-res.liblib.art/old-cached-source",
    )
    assert fetched == [
        "https://platform.example/source-1.mp4?signature=one",
        "https://platform.example/source-2.mp4?signature=two",
    ]
    assert uploads == [
        b"https://platform.example/source-1.mp4?signature=one",
        b"https://platform.example/source-2.mp4?signature=two",
    ]
    assert persistence.lookup_calls == 1


@pytest.mark.asyncio
async def test_libtv_paid_video_create_uses_post_once_once_after_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200
        text = "{}"
        headers: dict[str, str] = {}

        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict[str, object]:
            return self.payload

    class AsyncClient:
        def __init__(self) -> None:
            self.post_calls: list[str] = []
            self.post_once_calls: list[str] = []

        async def post(self, url: str, **_: object) -> Response:
            path = url.split("api.liblib.tv", 1)[-1]
            self.post_calls.append(path)
            if path == "/api/canvas/project/create":
                return Response({"code": 0, "data": {"projectMeta": {"uuid": "project-1"}}})
            return Response({"code": 0, "data": {}})

        async def post_once(self, url: str, **_: object) -> Response:
            path = url.split("api.liblib.tv", 1)[-1]
            self.post_once_calls.append(path)
            raise RuntimeError("connection lost after paid boundary")

    fake = AsyncClient()
    client = LibTVClient(token="token", webid="webid", persistence=None)
    monkeypatch.setattr(client, "async_client", fake)

    with pytest.raises(ProviderTransportError) as exc_info:
        await client.acreate(
            "topaz-video-upscaler",
            "topazlabs",
            "video",
            {"prompt": "", "videoList": ["https://libtv.example/source.mp4"]},
            "causyn-2k",
            allow_cached_project_retry=False,
            paid_submission=True,
        )

    assert exc_info.value.crossed_create_boundary is True
    assert fake.post_once_calls == ["/api/task/generation/create"]
    assert fake.post_calls.count("/api/task/generation/create") == 0


async def _source_url() -> str:
    return "https://platform.example/source.mp4"


def assert_source(url: str) -> None:
    assert url.startswith("https://platform.example/source-")


# --- failure-reason preservation (2026-09-02) -------------------------------
#
# `_retryable` used to take the provider's error text as `_message` and throw it
# away, writing the literal "provider_failure" into the state instead. Nothing
# else logged it either -- topaz.py's only warnings are about a corrupt/missing
# state machine. So a production 2K failure left three layers of generic text
# and no cause anywhere: the platform's `error_message`, litellm's "causyn 2K
# processing failed", and this state's "provider_failure".
#
# The reason now survives IN THE STATE AND THE LOG ONLY. The API-facing string
# must stay generic -- see 6214f3e146 ("hide 2K provider details in errors"),
# which deliberately renamed "Topaz" out of the caller-visible messages.


@pytest.mark.asyncio
async def test_create_rejection_keeps_the_provider_reason_in_state() -> None:
    redis = FakeRedis()
    client = FakeClient([ProviderRejected("算力不足")])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token-1", "webid-1"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    await adapter.advance(
        "base-reason-create", lambda _task_id: _source_url(), validate_source=lambda _: None
    )

    state = _state(redis.values["causyn:topaz:base-reason-create"])
    assert "算力不足" in str(state["failure_message"])


@pytest.mark.asyncio
async def test_source_import_failure_keeps_the_provider_reason_in_state() -> None:
    # The branch production actually hit: confirmed_submission_count stayed 0
    # and provider_task_id stayed null, so the failure was at source import or
    # create -- never at poll.
    redis = FakeRedis()
    # RuntimeError is in _TOPAZ_SOURCE_ERRORS; the import branch never sees a
    # ProviderRejected/ProviderTransportError (those come from acreate).
    client = FakeClient([], import_outcomes=[RuntimeError("source import refused")])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token-1", "webid-1"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    await adapter.advance(
        "base-reason-import", lambda _task_id: _source_url(), validate_source=lambda _: None
    )

    state = _state(redis.values["causyn:topaz:base-reason-import"])
    assert "source import refused" in str(state["failure_message"])
    assert state["confirmed_submission_count"] == 0
    assert state["provider_task_id"] is None


@pytest.mark.asyncio
async def test_the_reason_is_logged_with_enough_context_to_act_on(caplog) -> None:
    redis = FakeRedis()
    client = FakeClient([ProviderRejected("算力不足")])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token-1", "webid-1"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    with caplog.at_level(logging.WARNING):
        await adapter.advance(
            "base-reason-log", lambda _task_id: _source_url(), validate_source=lambda _: None
        )

    records = [r for r in caplog.records if r.message == "causyn Topaz attempt failed"]
    assert records, "the retryable path must log the provider reason"
    extra = records[0]
    # Enough context to act without a second query: which task, which attempt,
    # which account, and what the provider actually said.
    assert getattr(extra, "causyn_task_id") == "base-reason-log"
    assert getattr(extra, "attempt_index") == 1
    assert getattr(extra, "account_ref") == "account-1"
    assert "算力不足" in getattr(extra, "failure_message")


@pytest.mark.asyncio
async def test_exhausted_attempts_keep_the_last_reason_not_a_placeholder() -> None:
    redis = FakeRedis()
    accounts = TopazAccountPool(
        (
            TopazAccount("account-1", "token-1", "webid-1"),
            TopazAccount("account-2", "token-2", "webid-2"),
        )
    )
    client = FakeClient([ProviderRejected("算力不足")] * MAX_ATTEMPTS)
    adapter = TopazVideoAdapter(
        accounts,
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    for _ in range(MAX_ATTEMPTS):
        advance = await adapter.advance(
            "base-reason-exhausted", lambda _task_id: _source_url(), validate_source=lambda _: None
        )

    state = _state(redis.values["causyn:topaz:base-reason-exhausted"])
    assert advance.status == "failed"
    assert state["phase"] == "failed"
    # The terminal state is the one an operator reads after the fact; a
    # placeholder here is exactly what made the production incident unreadable.
    assert "算力不足" in str(state["failure_message"])


@pytest.mark.asyncio
async def test_a_blank_provider_reason_falls_back_to_the_placeholder() -> None:
    # `_optional_state_string` rejects an empty string, so a provider that
    # raises with no text must not be able to write an unloadable state.
    redis = FakeRedis()
    client = FakeClient([ProviderRejected("")])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token-1", "webid-1"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    await adapter.advance(
        "base-reason-blank", lambda _task_id: _source_url(), validate_source=lambda _: None
    )

    state = _state(redis.values["causyn:topaz:base-reason-blank"])
    assert state["failure_message"] == "provider_failure"
    # Must round-trip: a state that cannot be re-loaded strands the task.
    assert TopazState.from_json(redis.values["causyn:topaz:base-reason-blank"]) is not None


@pytest.mark.asyncio
async def test_a_pathological_provider_reason_is_bounded() -> None:
    redis = FakeRedis()
    client = FakeClient([ProviderRejected("x" * 50_000)])
    adapter = TopazVideoAdapter(
        TopazAccountPool((TopazAccount("account-1", "token-1", "webid-1"),)),
        client_factory=lambda _account: client,
        redis_factory=lambda: redis,
        now=lambda: 100.0,
        state_ttl=3600,
    )

    await adapter.advance(
        "base-reason-long", lambda _task_id: _source_url(), validate_source=lambda _: None
    )

    state = _state(redis.values["causyn:topaz:base-reason-long"])
    assert len(str(state["failure_message"])) <= MAX_FAILURE_MESSAGE_CHARS


def test_the_advance_result_cannot_carry_the_provider_reason() -> None:
    """Structural guard on the internal/external boundary.

    The reason is allowed to live in the Redis state and the log. `TopazAdvance`
    is what crosses back into `handler.py`, which turns it into a caller-visible
    error -- so if the reason ever became a field here, the next well-meaning
    change would plumb it into that message and undo 6214f3e1 ("hide 2K provider
    details in errors").
    """
    fields = set(TopazAdvance.__dataclass_fields__)

    assert fields == {"status", "provider_task_id", "result_url", "attempt_index"}
