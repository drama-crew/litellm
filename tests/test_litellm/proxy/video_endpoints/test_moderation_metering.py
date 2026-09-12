import asyncio
import os
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from prisma import Prisma
from redis.asyncio import Redis

from litellm.proxy.video_endpoints.moderation_metering import (
    BillingBinding,
    MeteringStore,
    PhaseEvent,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("MODERATION_METERING_POSTGRES_URL") or not os.getenv("MODERATION_METERING_REDIS_URL"),
    reason="isolated PostgreSQL and Redis required",
)


@pytest_asyncio.fixture(loop_scope="function")
async def store():
    schema = "meter_" + uuid4().hex
    url = os.environ["MODERATION_METERING_POSTGRES_URL"]
    control = Prisma(datasource={"url": url})
    db = Prisma(datasource={"url": url + "?schema=" + schema + "&connection_limit=8"})
    redis = Redis.from_url(os.environ["MODERATION_METERING_REDIS_URL"], decode_responses=True)
    namespace = schema + ":"
    try:
        await control.connect()
        await control.execute_raw(f'CREATE SCHEMA "{schema}"')
        await db.connect()
        migration = Path(
            "litellm-proxy-extras/litellm_proxy_extras/migrations/20260913000000_moderation_metering/migration.sql"
        )
        admission = migration.parent.parent / "20260913010000_legacy_financial_admission" / "migration.sql"
        for statement in (migration.read_text() + admission.read_text()).split(";"):
            if statement.strip():
                await db.execute_raw(statement)
        for statement in (
            'CREATE TABLE "LiteLLM_VerificationToken" (token text primary key, user_id text, team_id text, organization_id text, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_TeamTable" (team_id text primary key, organization_id text, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_EndUserTable" (user_id text primary key, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_TagTable" (tag_name text primary key, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_UserTable" (user_id text primary key, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_OrganizationTable" (organization_id text primary key, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_TeamMembership" (user_id text, team_id text, spend double precision default 0, total_spend double precision default 0, primary key(user_id,team_id))',
            "INSERT INTO \"LiteLLM_VerificationToken\" (token,user_id,team_id) VALUES ('key','user','team')",
            "INSERT INTO \"LiteLLM_TeamTable\" (team_id) VALUES ('team')",
            "INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('user')",
            "INSERT INTO \"LiteLLM_TeamMembership\" (user_id,team_id) VALUES ('user','team')",
        ):
            await db.execute_raw(statement)
        yield MeteringStore.from_client(db, redis, namespace=namespace), db, redis, namespace
    finally:
        try:
            try:
                async for key in redis.scan_iter(match=namespace + "*"):
                    await redis.delete(key)
            finally:
                await redis.aclose()
        finally:
            try:
                if db.is_connected():
                    await db.disconnect()
            finally:
                if control.is_connected():
                    try:
                        await control.execute_raw(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                    finally:
                        await control.disconnect()


def binding():
    return BillingBinding(
        intent_id="intent",
        request_digest="digest",
        fingerprint="key",
        user_id="user",
        team_id="team",
        model="video",
        expected_phases=("submit", "completion"),
    )


def event(phase="completion", amount="20"):
    return PhaseEvent(
        binding=binding(),
        request_id="public-video:intent:" + phase,
        phase=phase,
        provider="libtv",
        deployment_id="deployment",
        native_id="native",
        provider_task_id="provider-task",
        amount=None if amount is None else Decimal(amount),
        finalized=amount is not None,
    )


@pytest.mark.asyncio
async def test_actual_receipt_concurrent_replay_and_complete_zero_phase(store):
    meter, db, redis, prefix = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event("submit", "0"))
    await meter.persist(event())
    await asyncio.gather(*(meter.run_once() for _ in range(6)))
    envelope = await meter.settlement(binding())
    assert envelope.complete
    assert envelope.total_actual == Decimal("20")
    assert len(envelope.receipts) == 2
    for _ in range(2):
        await meter.persist(event())
        await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 20.0}]
    assert float(await redis.get(prefix + "spend:team:team")) == 20


@pytest.mark.asyncio
async def test_unknown_cost_and_incomplete_manifest_never_mean_zero(store):
    meter, _, _, _ = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event("submit", "0"))
    await meter.persist(event(amount=None))
    await meter.run_once()
    assert not (await meter.settlement(binding())).complete
    await meter.persist(event(amount="0"))
    await meter.run_once()
    result = await meter.settlement(binding())
    assert result.complete and result.total_actual == 0


@pytest.mark.asyncio
async def test_legacy_additive_counter_is_preserved(store):
    meter, _, redis, prefix = store
    await redis.set(prefix + "spend:team:team", "100")
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())
    await meter.run_once()
    assert float(await redis.get(prefix + "spend:team:team")) == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["user", "team", "missing"])
async def test_wrong_debit_identity_rolls_back_without_receipt(store, change):
    meter, db, _, _ = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())
    if change == "missing":
        await db.execute_raw('DELETE FROM "LiteLLM_UserTable"')
    else:
        await db.execute_raw(f"UPDATE \"LiteLLM_VerificationToken\" SET {change}_id='other'")
    with pytest.raises(ValueError):
        await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 0.0}]
    assert not (await meter.settlement(binding())).complete


@pytest.mark.asyncio
async def test_payload_and_actor_replay_conflicts(store):
    meter, _, _, _ = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())
    with pytest.raises(ValueError):
        await meter.persist(event(amount="21"))
    with pytest.raises(ValueError):
        await prepare_meter(meter, binding().model_copy(update={"user_id": "attacker"}), {})


@pytest.mark.asyncio
async def test_evicted_active_counter_requires_reconciliation(store):
    meter, db, redis, prefix = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())
    await redis.delete(prefix + "spend:team:team")
    with pytest.raises(ValueError):
        await meter.run_once()
    rows = await db.query_raw(
        "SELECT status FROM \"LiteLLM_ModerationMeteringCounter\" WHERE counter_key='spend:team:team'"
    )
    assert rows == [{"status": "reconciliation_required"}]
    with pytest.raises(ValueError):
        await meter.assert_admission(("spend:team:team",))


@pytest.mark.asyncio
async def test_sql_rollback_after_redis_adjustment_recovers_same_event(store):
    meter, db, redis, prefix = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())

    class FailingTransaction:
        def __init__(self, transaction):
            self.transaction = transaction

        async def query_raw(self, sql, *args):
            return await self.transaction.query_raw(sql, *args)

        async def execute_raw(self, sql, *args):
            if "SET status='settled'" in sql:
                raise RuntimeError("synthetic SQL failure after debit")
            return await self.transaction.execute_raw(sql, *args)

    @asynccontextmanager
    async def transactions():
        async with meter.transactions() as tx:
            yield FailingTransaction(tx)

    interrupted = MeteringStore(meter.db, transactions, meter.redis, namespace=prefix)
    with pytest.raises(RuntimeError):
        await interrupted.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 0.0}]
    assert float(await redis.get(prefix + "spend:team:team")) == 20
    await meter.assert_admission(("spend:team:team",))
    assert (await db.query_raw('SELECT DISTINCT status FROM "LiteLLM_ModerationMeteringCounter"')) == [
        {"status": "ready"}
    ]
    await db.execute_raw('UPDATE "LiteLLM_ModerationMeteringPhase" SET available_at=NOW()')
    await MeteringStore.from_client(db, redis, namespace=prefix).run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 20.0}]
    assert float(await redis.get(prefix + "spend:team:team")) == 20


@pytest.mark.asyncio
async def test_redis_lost_reply_does_not_repeat_reservation_adjustment(store):

    meter, db, redis, prefix = store
    await redis.set(prefix + "spend:team:team", 100)
    await prepare_meter(meter, binding(), {"spend:team:team": Decimal(5)})
    await meter.persist(event("submit"))

    class LostReply:
        async def eval(self, script, numkeys, *args):
            value = await redis.eval(script, numkeys, *args)
            from litellm.proxy.spend_tracking.protected_budget import APPLY_OPERATION

            if script == APPLY_OPERATION:
                raise RuntimeError("synthetic lost Lua reply")
            return value

    interrupted = MeteringStore(meter.db, meter.transactions, LostReply(), namespace=prefix)
    with pytest.raises(RuntimeError):
        await interrupted.run_once()
    assert float(await redis.get(prefix + "spend:team:team")) == 120
    await db.execute_raw('UPDATE "LiteLLM_ModerationMeteringPhase" SET available_at=NOW()')
    await meter.run_once()
    assert float(await redis.get(prefix + "spend:team:team")) == 120
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 20.0}]


@pytest.mark.asyncio
async def test_lost_sql_commit_reply_replays_receipt_without_cache_call(store):
    meter, db, redis, prefix = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())

    class ObservedTransaction:
        def __init__(self, tx):
            self.tx = tx
            self.committed_receipt = False

        async def query_raw(self, sql, *args):
            return await self.tx.query_raw(sql, *args)

        async def execute_raw(self, sql, *args):
            result = await self.tx.execute_raw(sql, *args)
            if "SET status='committed'" in sql:
                self.committed_receipt = True
            return result

    @asynccontextmanager
    async def lost_reply():
        async with meter.transactions() as tx:
            observed = ObservedTransaction(tx)
            yield observed
        if observed.committed_receipt:
            raise RuntimeError("synthetic lost SQL commit reply")

    interrupted = MeteringStore(meter.db, lost_reply, meter.redis, namespace=prefix)
    with pytest.raises(RuntimeError):
        await interrupted.run_once()
    await meter.persist(event())
    assert not await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 20.0}]
    assert float(await redis.get(prefix + "spend:team:team")) == 20


@pytest.mark.asyncio
async def test_old_backup_and_phase_identity_mismatch_fail_closed(store):
    meter, db, redis, prefix = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())
    await meter.run_once()
    with pytest.raises(ValueError, match="provider binding"):
        await meter.persist(event("submit", "0").model_copy(update={"native_id": "different-task"}))
    await redis.hset(prefix + "moderation:counter:spend:team:team", "seq", "0")
    with pytest.raises(ValueError):
        await meter.assert_admission(("spend:team:team",))
    assert (
        await db.query_raw(
            "SELECT status FROM \"LiteLLM_ModerationMeteringCounter\" WHERE counter_key='spend:team:team'"
        )
    ) == [{"status": "reconciliation_required"}]


@pytest.mark.asyncio
async def test_team_organization_binding_cannot_change_before_debit(store):
    meter, db, _, _ = store
    await prepare_meter(meter, binding(), {})
    await meter.persist(event())
    await db.execute_raw("UPDATE \"LiteLLM_TeamTable\" SET organization_id='foreign-org'")
    with pytest.raises(ValueError, match="organization"):
        await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{"spend": 0.0}]


@pytest.mark.asyncio
async def test_unknown_placeholder_rejects_noncanonical_request_id(store):
    meter, _, _, _ = store
    await prepare_meter(meter, binding(), {})
    with pytest.raises(ValueError, match="request"):
        await meter.persist(event().model_copy(update={"request_id": "unbound-phase"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [None, "end_user", "tag"])
async def test_frozen_end_user_and_tag_actual_debit_is_atomic(store, missing):
    meter, db, redis, prefix = store
    await db.execute_raw("INSERT INTO \"LiteLLM_EndUserTable\" (user_id) VALUES ('customer')")
    await db.execute_raw("INSERT INTO \"LiteLLM_TagTable\" (tag_name) VALUES ('campaign')")
    bound = BillingBinding(**{**binding().model_dump(), "end_user_id": "customer", "tag_ids": ("campaign",)})
    await prepare_meter(meter, bound, {})
    phase = event().model_copy(update={"binding": bound})
    await meter.persist(phase)
    if missing is not None:
        table = "LiteLLM_EndUserTable" if missing == "end_user" else "LiteLLM_TagTable"
        await db.execute_raw(f'DELETE FROM "{table}"')
        with pytest.raises(ValueError, match="identity"):
            await meter.run_once()
        assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 0.0}]
        assert float(await redis.get(prefix + "spend:team:team")) == 0
        return
    await meter.run_once()
    for table in ("LiteLLM_EndUserTable", "LiteLLM_TagTable"):
        assert await db.query_raw(f'SELECT spend FROM "{table}"') == [{"spend": 20.0}]
    for key in ("spend:end_user:customer", "spend:tag:campaign"):
        assert float(await redis.get(prefix + key)) == 20
    await meter.persist(phase)
    assert not await meter.run_once()


def protected_store(meter):
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore

    return ProtectedBudgetStore(meter.db, meter.transactions, meter.redis, namespace=meter.namespace)


def cutover_receipt(*, redis_value="100", sql_value="0"):
    from datetime import datetime, timezone
    from litellm.proxy.spend_tracking.protected_budget import (
        BudgetIdentity,
        CounterBaseline,
        CutoverReceipt,
        WriterEvidence,
    )

    return CutoverReceipt(
        receipt_id="maintenance-1",
        operator="test-operator",
        boundary=datetime(2026, 9, 1, tzinfo=timezone.utc),
        writers=(
            WriterEvidence(
                writer_id="synthetic-writer",
                version_sha="a" * 40,
                fence="maintenance-lock-1",
                state="terminated",
                evidence_sha256="b" * 64,
            ),
        ),
        admission_closed=True,
        reset_writers_fenced=True,
        inventory_complete=True,
        unresolved_unversioned_reservations=0,
        unflushed_unversioned_operations=0,
        inventory_sha256="c" * 64,
        baselines=(
            CounterBaseline(
                target=BudgetIdentity(kind="team", identity="team"),
                redis_value=Decimal(redis_value) if redis_value is not None else None,
                sql_value=Decimal(sql_value),
                period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_protected_vertical_slice_cutover_reserve_debit_and_rollover(store):
    from datetime import datetime, timedelta, timezone

    meter, db, redis, prefix = store
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100, ex=60)
    await authority.register_cutover(cutover_receipt())
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    await authority.mutate(
        "reserve-1",
        ("spend:team:team",),
        kind="reserve",
        amount=Decimal(5),
        reservation_id="request-1",
        valid_until=expiry,
    )
    await authority.mutate(
        "reserve-1",
        ("spend:team:team",),
        kind="reserve",
        amount=Decimal(5),
        reservation_id="request-1",
        valid_until=expiry,
    )
    assert float(await redis.get(prefix + "spend:team:team")) == 105
    await authority.reset(
        "reset-1", "spend:team:team", boundary=datetime(2026, 9, 1, tzinfo=timezone.utc), reset_at=None
    )
    assert float(await redis.get(prefix + "spend:team:team")) == 5
    await authority.mutate(
        "actual-1", ("spend:team:team",), kind="debit", amount=Decimal(20), reservation_id="request-1"
    )
    assert float(await redis.get(prefix + "spend:team:team")) == 20
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 20.0}]
    await authority.reset(
        "reset-1", "spend:team:team", boundary=datetime(2026, 9, 1, tzinfo=timezone.utc), reset_at=None
    )
    await authority.mutate(
        "actual-1", ("spend:team:team",), kind="debit", amount=Decimal(20), reservation_id="request-1"
    )
    assert float(await redis.get(prefix + "spend:team:team")) == 20
    assert await redis.ttl(prefix + "spend:team:team") == -1
    assert await redis.ttl(prefix + "moderation:counter:spend:team:team") == -1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["reserve", "debit", "reset"])
async def test_protected_lost_redis_response_and_restart_recover_exact_operation(store, kind):
    from datetime import datetime, timedelta, timezone
    from litellm.proxy.spend_tracking.protected_budget import APPLY_OPERATION, ProtectedBudgetStore

    meter, db, redis, prefix = store
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())

    class LostReply:
        async def eval(self, script, numkeys, *args):
            result = await redis.eval(script, numkeys, *args)
            if script == APPLY_OPERATION:
                raise ConnectionError("synthetic Lua reply lost")
            return result

    interrupted = ProtectedBudgetStore(meter.db, meter.transactions, LostReply(), namespace=prefix)
    with pytest.raises(ConnectionError):
        if kind == "reset":
            await interrupted.reset(
                "lost-1", "spend:team:team", boundary=datetime(2026, 9, 1, tzinfo=timezone.utc), reset_at=None
            )
        else:
            await interrupted.mutate(
                "lost-1",
                ("spend:team:team",),
                kind=kind,
                amount=Decimal(5),
                reservation_id="request-1" if kind == "reserve" else None,
                valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
            )
    await protected_store(meter).assert_admission(("spend:team:team",))
    assert (await authority.operation("lost-1")).status == "committed"
    expected = 0 if kind == "reset" else 105
    assert float(await redis.get(prefix + "spend:team:team")) == expected
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 5.0 if kind == "debit" else 0.0}]
    await authority.mutate("later", ("spend:team:team",), kind="debit", amount=Decimal(3))
    await authority.recover("lost-1")
    assert float(await redis.get(prefix + "spend:team:team")) == expected + 3


@pytest.mark.asyncio
async def test_protected_reset_sql_rollback_does_not_clear_later_increments(store):
    from datetime import datetime, timezone
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore

    meter, db, redis, prefix = store
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())

    class FailingTransaction:
        def __init__(self, tx):
            self.tx = tx

        async def query_raw(self, sql, *args):
            return await self.tx.query_raw(sql, *args)

        async def execute_raw(self, sql, *args):
            if "SET status='committed'" in sql:
                raise ConnectionError("synthetic reset SQL rollback")
            return await self.tx.execute_raw(sql, *args)

    @asynccontextmanager
    async def transactions():
        async with meter.transactions() as tx:
            yield FailingTransaction(tx)

    interrupted = ProtectedBudgetStore(meter.db, transactions, meter.redis, namespace=prefix)
    with pytest.raises(ConnectionError):
        await interrupted.reset(
            "reset-1", "spend:team:team", boundary=datetime(2026, 9, 1, tzinfo=timezone.utc), reset_at=None
        )
    assert float(await redis.get(prefix + "spend:team:team")) == 0
    await authority.assert_admission(("spend:team:team",))
    await authority.mutate("new-period", ("spend:team:team",), kind="debit", amount=Decimal(7))
    await authority.recover("reset-1")
    assert float(await redis.get(prefix + "spend:team:team")) == 7
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 7.0}]


@pytest.mark.asyncio
async def test_cutover_sql_intent_precedes_redis_and_lost_register_reply_recovers(store):
    from litellm.proxy.spend_tracking.protected_budget import REGISTER_COUNTER, ProtectedBudgetStore

    meter, db, redis, prefix = store
    await redis.set(prefix + "spend:team:team", 100)

    class LostReply:
        async def eval(self, script, numkeys, *args):
            result = await redis.eval(script, numkeys, *args)
            if script == REGISTER_COUNTER:
                assert await db.query_raw('SELECT status FROM "LiteLLM_BudgetCutover"') == [{"status": "pending"}]
                raise ConnectionError("synthetic registration reply lost")
            return result

    interrupted = ProtectedBudgetStore(meter.db, meter.transactions, LostReply(), namespace=prefix)
    with pytest.raises(ConnectionError):
        await interrupted.register_cutover(cutover_receipt())
    authority = protected_store(meter)
    await authority.register_cutover(cutover_receipt())
    await authority.assert_admission(("spend:team:team",))
    assert await db.query_raw('SELECT status FROM "LiteLLM_BudgetCutover"') == [{"status": "ready"}]


@pytest.mark.asyncio
async def test_unproven_redis_ahead_is_quarantined(store):
    from litellm.proxy.spend_tracking.protected_budget import BudgetReconciliation

    meter, db, redis, prefix = store
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())
    await redis.hset(
        prefix + "moderation:counter:spend:team:team", mapping={"seq": 1, "event": "invented", "payload": "invented"}
    )
    with pytest.raises(BudgetReconciliation):
        await authority.assert_admission(("spend:team:team",))
    assert await db.query_raw('SELECT status FROM "LiteLLM_ModerationMeteringCounter"') == [
        {"status": "reconciliation_required"}
    ]


async def prepare_meter(meter, bound, reservations):
    from datetime import datetime, timedelta, timezone
    from litellm.proxy.spend_tracking.protected_budget import BudgetIdentity, CounterBaseline

    authority = protected_store(meter)
    targets = (
        BudgetIdentity(kind="key", identity=bound.fingerprint),
        BudgetIdentity(kind="user", identity=bound.user_id),
        BudgetIdentity(kind="team", identity=bound.team_id),
        BudgetIdentity(kind="team_member", identity=bound.user_id, team_id=bound.team_id),
        *((BudgetIdentity(kind="org", identity=bound.organization_id),) if bound.organization_id else ()),
        *((BudgetIdentity(kind="end_user", identity=bound.end_user_id),) if bound.end_user_id else ()),
        *(BudgetIdentity(kind="tag", identity=tag) for tag in bound.tag_ids),
    )
    existing = await authority.states(meter.db, tuple(t.counter_key for t in targets))
    if len(existing) != len(targets):
        async with meter.transactions() as tx:
            balances = await meter._identity(tx, bound)
        baselines = tuple(
            [
                CounterBaseline(
                    target=target,
                    redis_value=await meter.redis.get(meter.namespace + target.counter_key),
                    sql_value=balances[target.counter_key],
                    period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
                )
                for target in targets
            ]
        )
        await authority.register_cutover(
            cutover_receipt().model_copy(update={"receipt_id": bound.intent_id, "baselines": baselines})
        )
    reservation_id = "reservation:" + bound.intent_id if reservations else None
    if reservation_id:
        for key, amount in reservations.items():
            await authority.mutate(
                "reserve:" + bound.intent_id + ":" + key,
                (key,),
                kind="reserve",
                amount=amount,
                reservation_id=reservation_id,
                valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
            )
    await meter.prepare(bound, reservations, reservation_id=reservation_id)


@pytest.mark.asyncio
async def test_new_entity_insert_creates_birth_proof_without_repeated_maintenance(store):
    from litellm.proxy.spend_tracking.protected_budget import BudgetPending

    meter, db, redis, prefix = store
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())
    assert not await authority.register_birth("spend:user:user")
    await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('born-user')")
    assert await authority.register_birth("spend:user:born-user")
    await authority.mutate("born-debit", ("spend:user:born-user",), kind="debit", amount=Decimal(4))
    assert float(await redis.get(prefix + "spend:user:born-user")) == 4
    assert await authority.register_birth("spend:user:born-user")
    await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('unversioned-user')")
    await db.execute_raw("UPDATE \"LiteLLM_UserTable\" SET spend=2 WHERE user_id='unversioned-user'")
    with pytest.raises(BudgetPending, match="unversioned"):
        await authority.register_birth("spend:user:unversioned-user")


@pytest.mark.asyncio
async def test_shared_callback_mixed_dimensions_reservation_reset_and_writer_mode(store, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy import proxy_server
    from litellm.proxy._types import Litellm_EntityType, LiteLLM_TeamTable, SpendUpdateQueueItem
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.hooks.proxy_track_cost_callback import _update_database_and_spend_counters
    from litellm.proxy.spend_tracking.budget_reservation import _BudgetCounter, _reserve_counter
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

    meter, db, redis, prefix = store
    await db.execute_raw('ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN budget_reset_at timestamptz')
    await db.execute_raw("UPDATE \"LiteLLM_TeamTable\" SET budget_reset_at='2026-09-01'")
    cache = RedisCache(host="127.0.0.1", port=39462, namespace=prefix[:-1])
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache(redis_cache=cache))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
    try:
        await redis.set(prefix + "spend:team:team", 100)
        await protected_store(meter).register_cutover(cutover_receipt())
        counter = _BudgetCounter(
            counter_key="spend:team:team", max_budget=1000, fallback_spend=0, entity_type="team", entity_id="team"
        )
        assert await _reserve_counter(counter, 5, reservation_id="shared-request") == 105
        team = LiteLLM_TeamTable(
            team_id="team", spend=0, budget_duration="1d", budget_reset_at=datetime(2026, 9, 1, tzinfo=timezone.utc)
        )
        await ResetBudgetJob._reset_budget_common(team, datetime.now(timezone.utc), "team")
        assert float(await redis.get(prefix + "spend:team:team")) == 5
        writer = DBSpendUpdateWriter()

        class SharedWriter:
            async def update_database(self, **kwargs):
                await writer._enqueue_cumulative_spend(
                    SpendUpdateQueueItem(
                        entity_type=Litellm_EntityType.TEAM, entity_id="team", response_cost=kwargs["response_cost"]
                    )
                )
                await writer._enqueue_cumulative_spend(
                    SpendUpdateQueueItem(
                        entity_type=Litellm_EntityType.USER, entity_id="user", response_cost=kwargs["response_cost"]
                    )
                )

        async def callback():
            await _update_database_and_spend_counters(
                proxy_logging_obj=SimpleNamespace(db_spend_update_writer=SharedWriter()),
                increment_spend_counters=proxy_server.increment_spend_counters,
                user_api_key=None,
                user_id="user",
                end_user_id=None,
                team_id="team",
                org_id=None,
                kwargs={"litellm_call_id": "shared-call", "response_cost": 20},
                completion_response=None,
                start_time=None,
                end_time=None,
                response_cost=20,
                budget_reservation={
                    "reservation_id": "shared-request",
                    "reserved_cost": 5,
                    "entries": [
                        {"counter_key": "spend:team:team", "reservation_id": "shared-request", "reserved_cost": 5}
                    ],
                },
            )

        await callback()
        await callback()
        assert float(await redis.get(prefix + "spend:team:team")) == 20
        assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 20.0}]
        buffered = await writer.spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions()
        assert not buffered["team_list_transactions"]
        assert buffered["user_list_transactions"] == {"user": 40.0}
        assert runtime.PROJECTION.get() is None
        await proxy_server.reseed_spend_counter_from_db("spend:team:team")
        assert float(await redis.get(prefix + "spend:team:team")) == 20
        monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "false")
        with pytest.raises(ValueError, match="configuration mismatch"):
            await proxy_server._increment_spend_counter_cache("spend:team:team", 5)
        assert float(await redis.get(prefix + "spend:team:team")) == 20
    finally:
        await cache.disconnect()


@pytest.mark.asyncio
async def test_all_dimensions_phase_receipt_and_window_rollover(store):
    import json
    from datetime import datetime, timedelta, timezone
    from litellm.proxy.spend_tracking.protected_budget import BudgetIdentity, CounterBaseline
    from litellm.proxy.video_endpoints.moderation_metering import BillingWindow

    meter, db, redis, prefix = store
    await db.execute_raw("INSERT INTO \"LiteLLM_OrganizationTable\" (organization_id) VALUES ('org')")
    await db.execute_raw("INSERT INTO \"LiteLLM_EndUserTable\" (user_id) VALUES ('customer')")
    await db.execute_raw("INSERT INTO \"LiteLLM_TagTable\" (tag_name) VALUES ('campaign')")
    await db.execute_raw("UPDATE \"LiteLLM_VerificationToken\" SET organization_id='org'")
    await db.execute_raw("UPDATE \"LiteLLM_TeamTable\" SET organization_id='org'")
    window_json = json.dumps([{"budget_duration": "1d", "reset_at": "2026-09-01T00:00:00+00:00", "max_budget": 100}])
    for table in ("LiteLLM_VerificationToken", "LiteLLM_TeamTable"):
        await db.execute_raw(f'ALTER TABLE "{table}" ADD COLUMN budget_limits jsonb')
        await db.execute_raw(f'UPDATE "{table}" SET budget_limits=$1::jsonb', window_json)
    targets = (
        BudgetIdentity(kind="key", identity="key"),
        BudgetIdentity(kind="user", identity="user"),
        BudgetIdentity(kind="team", identity="team"),
        BudgetIdentity(kind="org", identity="org"),
        BudgetIdentity(kind="team_member", identity="user", team_id="team"),
        BudgetIdentity(kind="end_user", identity="customer"),
        BudgetIdentity(kind="tag", identity="campaign"),
        BudgetIdentity(kind="key", identity="key", window="1d"),
        BudgetIdentity(kind="team", identity="team", window="1d"),
    )
    authority = protected_store(meter)
    baselines = tuple(
        CounterBaseline(
            target=t, redis_value=None, sql_value=Decimal(0), period_start=datetime(2026, 8, 1, tzinfo=timezone.utc)
        )
        for t in targets
    )
    await authority.register_cutover(cutover_receipt().model_copy(update={"baselines": baselines}))
    keys = tuple(sorted(t.counter_key for t in targets))
    await authority.mutate(
        "reserve-all",
        keys,
        kind="reserve",
        amount=Decimal(5),
        reservation_id="all",
        valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    for target in targets:
        await authority.reset(
            "reset:" + target.counter_key,
            target.counter_key,
            boundary=datetime(2026, 9, 1, tzinfo=timezone.utc),
            reset_at=datetime(2026, 9, 2, tzinfo=timezone.utc) if target.window else None,
        )
        assert float(await redis.get(prefix + target.counter_key)) == 5
    bound = binding().model_copy(
        update={
            "organization_id": "org",
            "end_user_id": "customer",
            "tag_ids": ("campaign",),
            "windows": tuple(
                BillingWindow(
                    kind=t.kind,
                    identity=t.identity,
                    duration="1d",
                    window_start=datetime(2026, 8, 31, tzinfo=timezone.utc),
                    reset_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                )
                for t in targets
                if t.window
            ),
        }
    )
    await meter.prepare(bound, {key: Decimal(5) for key in keys}, reservation_id="all")
    await meter.persist(event("submit", "20").model_copy(update={"binding": bound}))
    await meter.run_once()
    for target in targets:
        assert float(await redis.get(prefix + target.counter_key)) == 20
        if not target.window:
            assert await db.query_raw(f'SELECT spend FROM "{target.table}"') == [{"spend": 20.0}]
    assert (await meter.settlement(bound)).receipts[0].amount == Decimal(20)


@pytest.mark.asyncio
async def test_expired_unknown_reservation_has_no_new_period_subtraction(store):
    from datetime import datetime, timedelta, timezone

    meter, db, redis, prefix = store
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())
    await authority.mutate(
        "short-reserve",
        ("spend:team:team",),
        kind="reserve",
        amount=Decimal(5),
        reservation_id="unknown",
        valid_until=datetime.now(timezone.utc) + timedelta(seconds=0.5),
    )
    await asyncio.sleep(0.6)
    await authority.reset("expire-reset", "spend:team:team", boundary=datetime.now(timezone.utc), reset_at=None)
    assert float(await redis.get(prefix + "spend:team:team")) == 0
    await authority.mutate(
        "late-actual", ("spend:team:team",), kind="debit", amount=Decimal(3), reservation_id="unknown"
    )
    assert float(await redis.get(prefix + "spend:team:team")) == 3
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 3.0}]


@pytest.mark.asyncio
async def test_new_window_configuration_has_durable_birth(store):
    meter, db, redis, prefix = store
    await db.execute_raw('ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN budget_limits jsonb')
    await redis.set(prefix + "spend:team:team", 100)
    authority = protected_store(meter)
    await authority.register_cutover(cutover_receipt())
    await db.execute_raw(
        'UPDATE "LiteLLM_TeamTable" SET budget_limits=\'[{"budget_duration":"1d","max_budget":10,"reset_at":"2026-09-15T00:00:00Z"}]\''
    )
    assert await authority.register_birth("spend:team:team:window:1d")
    await authority.mutate("new-window-actual", ("spend:team:team:window:1d",), kind="debit", amount=Decimal(3))
    assert float(await redis.get(prefix + "spend:team:team:window:1d")) == 3


@pytest.mark.asyncio
async def test_prepare_rejects_wrong_durable_reservation_binding(store):
    from datetime import datetime, timedelta, timezone

    meter, db, redis, prefix = store
    await prepare_meter(meter, binding(), {})
    authority = protected_store(meter)
    await authority.mutate(
        "binding-reserve",
        ("spend:team:team",),
        kind="reserve",
        amount=Decimal(3),
        reservation_id="owned-reservation",
        valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    with pytest.raises(ValueError, match="reservation binding"):
        await meter.prepare(
            binding().model_copy(update={"intent_id": "other-intent"}),
            {"spend:team:team": Decimal(8)},
            reservation_id="owned-reservation",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("event_kind", ["causyn", "image"])
async def test_reservation_adjustment_and_partial_identity_outbox_use_actual_authority(store, monkeypatch, event_kind):
    from datetime import datetime, timezone, timedelta
    from types import SimpleNamespace
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy import proxy_server
    from litellm.llms.causyn.context_ir_budget import settle_reservation
    from litellm.llms.libtv.billing_outbox import CausynBillingEvent, LibTVBillingReconciler, enqueue_causyn_billing

    meter, db, redis, prefix = store
    cache = RedisCache(host="127.0.0.1", port=39462, namespace=prefix[:-1])
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache(redis_cache=cache))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
    try:
        authority = protected_store(meter)
        await redis.set(prefix + "spend:team:team", 100)
        await redis.set(prefix + "spend:user:user", 105)
        await authority.register_cutover(cutover_receipt())
        await authority.mutate(
            "ir-reserve",
            ("spend:team:team",),
            kind="reserve",
            amount=Decimal(5),
            reservation_id="ir",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        reservation = {
            "reserved_cost": 5,
            "reservation_id": "ir",
            "entries": [
                {"counter_key": "spend:team:team", "reserved_cost": 5, "reservation_id": "ir"},
                {"counter_key": "spend:user:user", "reserved_cost": 5},
            ],
        }
        await settle_reservation("ir-task", reservation, 3)
        await authority.reset("ir-reset", "spend:team:team", boundary=datetime.now(timezone.utc), reset_at=None)
        await settle_reservation("ir-task", reservation, 3)
        assert float(await redis.get(prefix + "spend:team:team")) == 3
        assert float(await redis.get(prefix + "spend:user:user")) == 103
        assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 0.0}]
        from litellm.llms.libtv import billing_outbox

        await create_financial_projection_tables(db)
        stream = prefix + "outbox"
        monkeypatch.setattr(billing_outbox, "CAUSYN_BILLING_STREAM_KEY", stream)
        monkeypatch.setattr(billing_outbox, "CAUSYN_BILLING_MARKER_PREFIX", prefix + "enqueued:")
        if event_kind == "causyn":
            await enqueue_causyn_billing(
                redis,
                CausynBillingEvent(
                    provider_task_id="ir-task",
                    response_cost=3,
                    team_id="team",
                    user_id="user",
                    task_type="h3_context_ir",
                    reservation_id="ir",
                ),
            )
        else:
            import json
            from litellm.llms.libtv.billing_outbox import ImageBillingEvent

            await redis.xadd(
                stream,
                {
                    "payload": json.dumps(
                        ImageBillingEvent(
                            deployment_id="dep",
                            provider_task_id="ir-task",
                            response_cost=3,
                            team_id="team",
                            user_id="user",
                        ).to_dict()
                    )
                },
            )
        for consumer in ("one", "restart"):
            worker = LibTVBillingReconciler(
                redis, SimpleNamespace(db=db), stream_key=stream, consumer_group=prefix + "group", consumer=consumer
            )
            assert await worker.reconcile_once() == (1 if consumer == "one" else 0)
            assert (await redis.xpending(stream, prefix + "group"))["pending"] == 0
        for table in (
            "LiteLLM_UserTable",
            "LiteLLM_TeamTable",
            "LiteLLM_DailyUserSpend",
            "LiteLLM_DailyTeamSpend",
            "LiteLLM_SpendLogs",
        ):
            assert await db.query_raw(f'SELECT spend FROM "{table}"') == [{"spend": 3.0}]
        assert await db.query_raw('SELECT spend FROM "LiteLLM_VerificationToken"') == [{"spend": 0.0}]
        assert float(await redis.get(prefix + "spend:team:team")) == (3 if event_kind == "causyn" else 6)

    finally:
        await cache.disconnect()


@pytest.mark.asyncio
async def test_cutover_cli_replay_and_missing_evidence_fails_closed(store, monkeypatch, tmp_path):
    from scripts.protected_budget_cutover import execute
    from litellm.proxy.spend_tracking.protected_budget import BudgetReconciliation

    meter, db, redis, prefix = store
    await redis.set(prefix + "spend:team:team", 100)
    receipt_path = tmp_path / "operator-attested.json"
    receipt_path.write_text(cutover_receipt().model_dump_json())
    monkeypatch.setenv("COUNTER_TEST_SQL", os.environ["MODERATION_METERING_POSTGRES_URL"] + "?schema=" + prefix[:-1])
    monkeypatch.setenv("COUNTER_TEST_REDIS", os.environ["MODERATION_METERING_REDIS_URL"])
    await execute(receipt_path, prefix, "COUNTER_TEST_SQL", "COUNTER_TEST_REDIS")
    await execute(receipt_path, prefix, "COUNTER_TEST_SQL", "COUNTER_TEST_REDIS")
    await redis.delete(prefix + "spend:team:team")
    with pytest.raises(BudgetReconciliation):
        await execute(receipt_path, prefix, "COUNTER_TEST_SQL", "COUNTER_TEST_REDIS")
    assert (await protected_store(meter).states(meter.db, ("spend:team:team",)))[0].status == "reconciliation_required"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,identity,team_id",
    [
        ("key", "key", None),
        ("org", "org", None),
        ("team_member", "user", "team"),
        ("tag", "tag", None),
        ("end_user", "end", None),
    ],
)
async def test_linked_shared_reset_carries_zero_sql_spend_reservation(store, monkeypatch, kind, identity, team_id):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.proxy.spend_tracking.protected_budget import BudgetIdentity, CounterBaseline
    from litellm.proxy.video_endpoints.moderation_metering_runtime import reset_linked_registered

    meter, db, redis, prefix = store
    target = BudgetIdentity(kind=kind, identity=identity, team_id=team_id)
    await db.execute_raw('CREATE TABLE "LiteLLM_BudgetTable" (budget_id text primary key,budget_reset_at timestamptz)')
    await db.execute_raw("INSERT INTO \"LiteLLM_BudgetTable\" VALUES ('linked','2026-09-01')")
    if kind in ("org", "tag", "end_user"):
        await db.execute_raw(f'INSERT INTO "{target.table}" ({target.column}) VALUES ($1)', identity)
    cache = RedisCache(host="127.0.0.1", port=39462, namespace=prefix[:-1])
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache(redis_cache=cache))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
    try:
        authority = protected_store(meter)
        await authority.register_cutover(
            cutover_receipt().model_copy(
                update={
                    "baselines": (
                        CounterBaseline(
                            target=target, sql_value=0, period_start=datetime(2026, 8, 1, tzinfo=timezone.utc)
                        ),
                    )
                }
            )
        )
        await authority.mutate(
            "linked-reserve",
            (target.counter_key,),
            kind="reserve",
            amount=Decimal(5),
            reservation_id="linked-r",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        class Table:
            async def find_many(self, where):
                assert "spend" not in where
                return [
                    SimpleNamespace(
                        **{target.column: identity, "budget_id": "linked", **({"team_id": team_id} if team_id else {})}
                    )
                ]

            async def update_many(self, where, data):
                assert where["NOT"]["OR"] == [{target.column: identity, **({"team_id": team_id} if team_id else {})}]
                return 0

        job = ResetBudgetJob(proxy_logging_obj=SimpleNamespace(), prisma_client=SimpleNamespace(db=db))
        if kind == "end_user":
            assert await reset_linked_registered(target.counter_key, "linked")
        else:
            await job._cascade_reset_spend_for_budget_link(
                budgets_to_reset=[SimpleNamespace(budget_id="linked")],
                table=Table(),
                counter_key_fn=lambda row: target.counter_key,
                log_subject="synthetic",
                extra_where={"spend": {"gt": 0}},
            )
        assert float(await redis.get(prefix + target.counter_key)) == 5
        await authority.mutate(
            "linked-actual", (target.counter_key,), kind="debit", amount=Decimal(3), reservation_id="linked-r"
        )
        assert await reset_linked_registered(target.counter_key, "linked")
        assert float(await redis.get(prefix + target.counter_key)) == 3
        assert await db.query_raw(f'SELECT spend FROM "{target.table}"') == [{"spend": 3.0}]
    finally:
        await cache.disconnect()


@pytest.mark.asyncio
async def test_cutover_sql_rollback_never_creates_orphan_redis_generation(store):
    from contextlib import asynccontextmanager
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore

    meter, db, redis, prefix = store
    await redis.set(prefix + "spend:team:team", 100)

    @asynccontextmanager
    async def rollback():
        async with meter.transactions() as tx:
            yield tx
            raise RuntimeError("cutover SQL rollback")

    authority = ProtectedBudgetStore(meter.db, rollback, meter.redis, namespace=prefix)
    with pytest.raises(RuntimeError, match="cutover SQL rollback"):
        await authority.register_cutover(cutover_receipt())
    assert await db.query_raw('SELECT * FROM "LiteLLM_ModerationMeteringCounter"') == []
    assert await db.query_raw('SELECT * FROM "LiteLLM_BudgetCutover"') == []
    assert not await redis.exists(prefix + "moderation:counter:spend:team:team")
    assert float(await redis.get(prefix + "spend:team:team")) == 100
    await protected_store(meter).register_cutover(cutover_receipt())


@pytest.mark.asyncio
async def test_shared_window_reset_stale_job_replay_preserves_later_actual(store, monkeypatch):
    from copy import deepcopy
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.proxy.spend_tracking.protected_budget import BudgetIdentity, CounterBaseline

    meter, db, redis, prefix = store
    await db.execute_raw('ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN budget_limits jsonb')
    await db.execute_raw(
        'UPDATE "LiteLLM_TeamTable" SET budget_limits=\'[{"budget_duration":"1d","max_budget":10,"reset_at":"2026-09-01T00:00:00Z"}]\''
    )
    target = BudgetIdentity(kind="team", identity="team", window="1d")
    cache = RedisCache(host="127.0.0.1", port=39462, namespace=prefix[:-1])
    dual = DualCache(redis_cache=cache)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", dual)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
    try:
        authority = protected_store(meter)
        await authority.register_cutover(
            cutover_receipt().model_copy(
                update={
                    "baselines": (
                        CounterBaseline(
                            target=target, sql_value=0, period_start=datetime(2026, 8, 1, tzinfo=timezone.utc)
                        ),
                    )
                }
            )
        )
        await authority.mutate(
            "window-reserve",
            (target.counter_key,),
            kind="reserve",
            amount=Decimal(5),
            reservation_id="window-r",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        window = {"budget_duration": "1d", "max_budget": 10, "reset_at": "2026-09-01T00:00:00Z"}
        for amount in (None, Decimal(3)):
            if amount is not None:
                await authority.mutate(
                    "window-actual", (target.counter_key,), kind="debit", amount=amount, reservation_id="window-r"
                )
            assert not await ResetBudgetJob._reset_expired_window(
                window=deepcopy(window), counter_key=target.counter_key, now=datetime.now(), spend_counter_cache=dual
            )
            assert float(await redis.get(prefix + target.counter_key)) == (5 if amount is None else 3)
        assert await redis.ttl(prefix + target.counter_key) == -1
    finally:
        await cache.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,amount", [("resize", Decimal(3)), ("release", Decimal(0))])
async def test_adjustment_lost_reply_recovery_and_old_replay_after_reset(store, kind, amount):
    from datetime import datetime, timedelta, timezone
    from litellm.proxy.spend_tracking.protected_budget import ProtectedBudgetStore, APPLY_OPERATION

    meter, db, redis, prefix = store
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())
    await authority.mutate(
        "adjust-reserve",
        ("spend:team:team",),
        kind="reserve",
        amount=Decimal(5),
        reservation_id="adjust-r",
        valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    class LostReply:
        async def eval(self, script, numkeys, *args):
            result = await redis.eval(script, numkeys, *args)
            if script == APPLY_OPERATION:
                raise ConnectionError("lost adjustment reply")
            return result

    lossy = ProtectedBudgetStore(meter.db, meter.transactions, LostReply(), namespace=prefix)
    with pytest.raises(ConnectionError, match="lost adjustment reply"):
        await lossy.mutate("adjust", ("spend:team:team",), kind=kind, amount=amount, reservation_id="adjust-r")
    await protected_store(meter).recover("adjust")
    assert float(await redis.get(prefix + "spend:team:team")) == 100 + float(amount)
    await authority.reset("adjust-reset", "spend:team:team", boundary=datetime.now(timezone.utc), reset_at=None)
    await authority.mutate("another-actual", ("spend:team:team",), kind="debit", amount=Decimal(7))
    await authority.mutate("adjust", ("spend:team:team",), kind=kind, amount=amount, reservation_id="adjust-r")
    assert float(await redis.get(prefix + "spend:team:team")) == 7 + float(amount)
    await authority.mutate(
        "adjust-actual", ("spend:team:team",), kind="debit", amount=Decimal(2), reservation_id="adjust-r"
    )
    assert float(await redis.get(prefix + "spend:team:team")) == 9
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 9.0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("submit_amount", [Decimal(0), Decimal(2)])
async def test_fix1_partial_phases_retain_reservation_until_complete_manifest(store, submit_amount):
    from datetime import datetime, timezone

    meter, db, redis, prefix = store
    await prepare_meter(meter, binding(), {"spend:team:team": Decimal(5)})
    await meter.persist(event("submit", str(submit_amount)))
    await meter.run_once()
    expected_remaining = 5 - submit_amount
    rows = await db.query_raw('SELECT remaining,status FROM "LiteLLM_BudgetReservation"')
    assert Decimal(str(rows[0]["remaining"])) == expected_remaining
    assert rows[0]["status"] == "active"
    assert float(await redis.get(prefix + "spend:team:team")) == 5
    assert not (await meter.settlement(binding())).complete
    authority = protected_store(meter)
    await authority.reset("phase-reset", "spend:team:team", boundary=datetime.now(timezone.utc), reset_at=None)
    assert float(await redis.get(prefix + "spend:team:team")) == float(expected_remaining)
    await meter.persist(event("completion", "3"))
    await meter.run_once()
    assert float(await redis.get(prefix + "spend:team:team")) == 3
    assert await db.query_raw('SELECT remaining,status FROM "LiteLLM_BudgetReservation"') == [
        {"remaining": 0, "status": "settled"}
    ]
    assert (await meter.settlement(binding())).complete
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 3.0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("released", [False, True])
async def test_fix1_late_context_ir_reconciles_expired_or_released_without_actual_sql(store, monkeypatch, released):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy import proxy_server
    from litellm.llms.causyn.context_ir_budget import settle_reservation

    meter, db, redis, prefix = store
    cache = RedisCache(host="127.0.0.1", port=39462, namespace=prefix[:-1])
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache(redis_cache=cache))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
    try:
        await redis.set(prefix + "spend:team:team", 100)
        authority = protected_store(meter)
        await authority.register_cutover(cutover_receipt())
        await authority.mutate(
            "late-reserve",
            ("spend:team:team",),
            kind="reserve",
            amount=Decimal(5),
            reservation_id="late-r",
            valid_until=datetime.now(timezone.utc) + timedelta(seconds=0.3),
        )
        if released:
            await authority.mutate(
                "late-release", ("spend:team:team",), kind="release", amount=Decimal(0), reservation_id="late-r"
            )
        await asyncio.sleep(0.35)
        await authority.reset("late-reset", "spend:team:team", boundary=datetime.now(timezone.utc), reset_at=None)
        reservation = {
            "reservation_id": "late-r",
            "reserved_cost": 5,
            "entries": [{"counter_key": "spend:team:team", "reservation_id": "late-r", "reserved_cost": 5}],
        }
        await settle_reservation("late-task", reservation, 3)
        await settle_reservation("late-task", reservation, 3)
        assert float(await redis.get(prefix + "spend:team:team")) == 3
        assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 0.0}]
        await authority.reset("late-next-reset", "spend:team:team", boundary=datetime.now(timezone.utc), reset_at=None)
        await settle_reservation("late-task", reservation, 3)
        assert float(await redis.get(prefix + "spend:team:team")) == 0
    finally:
        await cache.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entrypoint,scenario",
    [
        ("callback", "global-ready"),
        ("direct", "global-born"),
        ("direct", "global-unregistered"),
        ("callback", "new-window"),
        ("direct", "new-window"),
        ("callback", "missing-id"),
        ("direct", "missing-id"),
    ],
)
async def test_fix1_real_writer_dimensions_and_original_operation_identity(store, monkeypatch, entrypoint, scenario):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import litellm
    from litellm.caching import DualCache, RedisCache
    from litellm.proxy import proxy_server
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.hooks.proxy_track_cost_callback import _update_database_and_spend_counters
    from litellm.proxy.spend_tracking import spend_tracking_utils
    from litellm.proxy.spend_tracking.protected_budget import BudgetIdentity, CounterBaseline

    meter, db, redis, prefix = store
    await db.execute_raw("INSERT INTO \"LiteLLM_EndUserTable\" (user_id) VALUES ('end')")
    if scenario in ("global-ready", "global-unregistered"):
        await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('global-budget')")
    await db.execute_raw('ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN budget_limits jsonb')
    cache = RedisCache(host="127.0.0.1", port=39462, namespace=prefix[:-1])
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache(redis_cache=cache))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=db))
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "global-budget")
    monkeypatch.setattr(litellm, "max_budget", 100 if scenario.startswith("global") else 0)
    monkeypatch.setenv("DRAMA_PROTECTED_BUDGETS_ENABLED", "true")
    writer = DBSpendUpdateWriter()
    spend_log = AsyncMock()
    monkeypatch.setattr(writer, "_insert_spend_log_to_db", spend_log)
    for dimension in ("user", "end_user", "agent", "team", "org", "tag"):
        monkeypatch.setattr(writer, "add_spend_log_transaction_to_daily_" + dimension + "_transaction", AsyncMock())
    completed = asyncio.Event()
    original_batch = writer._batch_database_updates

    async def batch(**kwargs):
        try:
            await original_batch(**kwargs)
        finally:
            completed.set()

    monkeypatch.setattr(writer, "_batch_database_updates", batch)
    identifier = "" if scenario == "missing-id" else "fix1-actual"
    monkeypatch.setattr(
        spend_tracking_utils,
        "get_logging_payload",
        lambda **kwargs: {
            "startTime": "2026-09-01T00:00:00",
            "endTime": "2026-09-01T00:00:01",
            "request_id": identifier,
            "request_tags": "[]",
            "model": "synthetic",
        },
    )
    try:
        authority = protected_store(meter)
        await redis.set(prefix + "spend:team:team", 100)
        await authority.register_cutover(cutover_receipt())
        if scenario == "global-ready":
            await authority.register_cutover(
                cutover_receipt().model_copy(
                    update={
                        "receipt_id": "global-cutover",
                        "baselines": (
                            CounterBaseline(
                                target=BudgetIdentity(kind="user", identity="global-budget"),
                                sql_value=0,
                                period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
                            ),
                        ),
                    }
                )
            )
        elif scenario == "global-born":
            await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('global-budget')")
        elif scenario == "new-window":
            await db.execute_raw(
                'UPDATE "LiteLLM_TeamTable" SET budget_limits=\'[{"budget_duration":"1d","max_budget":10,"reset_at":"2026-09-20T00:00:00Z"}]\''
            )

        async def invoke():
            completed.clear()
            args = {
                "user_id": "user",
                "end_user_id": "end",
                "team_id": "team",
                "org_id": None,
                "kwargs": {"litellm_call_id": identifier, "response_cost": 3},
                "completion_response": None,
                "start_time": None,
                "end_time": None,
                "response_cost": 3,
            }
            if entrypoint == "direct":
                await writer.update_database(token=None, **args)
            else:
                await _update_database_and_spend_counters(
                    proxy_logging_obj=SimpleNamespace(db_spend_update_writer=writer),
                    increment_spend_counters=proxy_server.increment_spend_counters,
                    user_api_key=None,
                    budget_reservation=None,
                    **args,
                )
            await asyncio.wait_for(completed.wait(), 2)

        for _ in range(2):
            if scenario == "missing-id":
                with pytest.raises(ValueError, match="operation identity"):
                    await invoke()
            else:
                await invoke()
        if scenario == "missing-id":
            assert await db.query_raw('SELECT operation_id FROM "LiteLLM_BudgetOperation"') == []
            assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 0.0}]
        elif scenario == "global-unregistered":
            assert await db.query_raw("SELECT spend FROM \"LiteLLM_UserTable\" WHERE user_id='global-budget'") == [
                {"spend": 0.0}
            ]
        elif scenario.startswith("global"):
            assert await db.query_raw("SELECT spend FROM \"LiteLLM_UserTable\" WHERE user_id='global-budget'") == [
                {"spend": 3.0}
            ]
            assert float(await redis.get(prefix + "spend:user:global-budget")) == 3
        else:
            assert float(await redis.get(prefix + "spend:team:team:window:1d")) == 3
        buffered = await writer.spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions()
        if scenario == "missing-id":
            spend_log.assert_not_awaited()
            assert not buffered["user_list_transactions"]
        else:
            assert buffered["end_user_list_transactions"] == {"end": 6.0}
            assert not buffered["team_list_transactions"]
            if scenario == "global-unregistered":
                assert buffered["user_list_transactions"] == {"user": 6.0, "global-budget": 6.0}
    finally:
        await cache.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("phases", [("submit", "completion"), ("completion", "submit")])
async def test_fix1_parallel_phase_receipts_freeze_final_disposition_independent_of_order(store, phases):
    meter, db, redis, prefix = store
    bound = binding().model_copy(update={"expected_phases": phases})
    await prepare_meter(meter, bound, {"spend:team:team": Decimal(5)})
    for phase, amount in (("submit", "2"), ("completion", "3")):
        await meter.persist(event(phase, amount).model_copy(update={"binding": bound}))
    await asyncio.gather(*(meter.run_once() for _ in range(4)))
    assert (await meter.settlement(bound)).complete
    assert float(await redis.get(prefix + "spend:team:team")) == 5
    assert await db.query_raw('SELECT remaining,status FROM "LiteLLM_BudgetReservation"') == [
        {"remaining": 0, "status": "settled"}
    ]
    operations = await db.query_raw(
        "SELECT payload->'reservation_final' AS final FROM \"LiteLLM_BudgetOperation\" WHERE payload->>'kind'='debit'"
    )
    assert sorted(row["final"] for row in operations) == [False, True]
    for phase in phases:
        from litellm.proxy.video_endpoints.moderation_metering import request_id

        await protected_store(meter).recover(request_id(bound, phase))
    assert float(await redis.get(prefix + "spend:team:team")) == 5


async def create_financial_projection_tables(db):
    await db.execute_raw(
        'CREATE TABLE "LiteLLM_SpendLogs" (request_id text primary key,call_type text,api_key text,spend double precision,"startTime" timestamptz,"endTime" timestamptz,model text,model_id text,model_group text,custom_llm_provider text,"user" text,team_id text,organization_id text,end_user text,metadata jsonb,request_tags jsonb,prompt_tokens integer default 0,completion_tokens integer default 0,total_tokens integer default 0)'
    )
    for table, dimension in [
        ("LiteLLM_DailyUserSpend", "user_id"),
        ("LiteLLM_DailyTeamSpend", "team_id"),
        ("LiteLLM_DailyOrganizationSpend", "organization_id"),
        ("LiteLLM_DailyEndUserSpend", "end_user_id"),
        ("LiteLLM_DailyTagSpend", "tag"),
    ]:
        await db.execute_raw(
            f'CREATE TABLE "{table}" (id text primary key,{dimension} text,date text,api_key text,model text,model_group text,custom_llm_provider text,mcp_namespaced_tool_name text,endpoint text,spend double precision default 0,api_requests bigint default 0,successful_requests bigint default 0,prompt_tokens bigint default 0,completion_tokens bigint default 0,updated_at timestamptz,UNIQUE({dimension},date,api_key,model,custom_llm_provider,mcp_namespaced_tool_name,endpoint))'
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("event_kind", ["image", "causyn"])
async def test_entry_actual_outbox_commits_mixed_dimensions_and_projections_once(store, monkeypatch, event_kind):
    from datetime import datetime, timezone
    import json
    from types import SimpleNamespace
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
    from litellm.llms.libtv.billing_outbox import ImageBillingEvent, CausynBillingEvent, LibTVBillingReconciler

    meter, db, redis, prefix = store
    await create_financial_projection_tables(db)
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setattr(
        runtime, "counter_mode", __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value=True)
    )
    fields = dict(
        deployment_id="deployment",
        provider_task_id="actual-task",
        response_cost=3,
        api_key="key",
        team_id="team",
        user_id="user",
        occurred_at=datetime(2026, 9, 13, tzinfo=timezone.utc).isoformat(),
    )
    event = (
        ImageBillingEvent(**fields, scale=4, project_id="project", artifact_id="artifact", attribution_user_id="owner")
        if event_kind == "image"
        else CausynBillingEvent(**fields)
    )
    stream = prefix + "entry-outbox"
    await redis.xadd(stream, {"payload": json.dumps(event.to_dict())})
    worker = LibTVBillingReconciler(redis, SimpleNamespace(db=db), stream_key=stream, consumer_group=prefix + "group")
    assert await worker.reconcile_once() == 1
    assert (await redis.xpending(stream, prefix + "group"))["pending"] == 0
    await asyncio.gather(*(worker._commit_event(event) for _ in range(3)))
    for table in [
        "LiteLLM_TeamTable",
        "LiteLLM_UserTable",
        "LiteLLM_VerificationToken",
        "LiteLLM_TeamMembership",
        "LiteLLM_DailyUserSpend",
        "LiteLLM_DailyTeamSpend",
        "LiteLLM_SpendLogs",
    ]:
        assert await db.query_raw(f'SELECT spend FROM "{table}"') == [{"spend": 3.0}]
    assert float(await redis.get(prefix + "spend:team:team")) == 103
    assert await db.query_raw('SELECT status FROM "LiteLLM_BudgetOperation"') == [{"status": "committed"}]
    metadata = (await db.query_raw('SELECT metadata FROM "LiteLLM_SpendLogs"'))[0]["metadata"]
    if event_kind == "image":
        assert {
            key: metadata[key] for key in ("libtv_billing_key", "scale", "project_id", "artifact_id", "user_id")
        } == {
            "libtv_billing_key": event.billing_key,
            "scale": 4,
            "project_id": "project",
            "artifact_id": "artifact",
            "user_id": "owner",
        }
    else:
        assert metadata["causyn_billing_key"] == event.billing_key
        assert metadata["provider"] == event.provider


@pytest.mark.asyncio
async def test_entry_phase_projection_rolls_back_with_actual_then_recovers(store):
    from datetime import datetime, timezone
    from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts

    meter, db, redis, prefix = store
    await create_financial_projection_tables(db)
    await prepare_meter(meter, binding(), {})
    facts = BillingFacts(
        started_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        ended_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        route="avideo_status",
        pricing=(("output_cost_per_second", Decimal("4")),),
        duration_seconds=5,
    )
    terminal = event().model_copy(update={"facts": facts})
    await meter.persist(terminal)
    await db.execute_raw(
        'ALTER TABLE "LiteLLM_DailyTeamSpend" ADD CONSTRAINT synthetic_projection_failure CHECK(spend<0)'
    )
    with pytest.raises(Exception):
        await meter.run_once()
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 0.0}]
    assert await db.query_raw('SELECT request_id FROM "LiteLLM_SpendLogs"') == []
    await db.execute_raw('ALTER TABLE "LiteLLM_DailyTeamSpend" DROP CONSTRAINT synthetic_projection_failure')
    await db.execute_raw('UPDATE "LiteLLM_ModerationMeteringPhase" SET available_at=NOW()')
    await meter.run_once()
    await meter.persist(terminal)
    await meter.run_once()
    assert await db.query_raw('SELECT spend FROM "LiteLLM_DailyTeamSpend"') == [{"spend": 20.0}]
    assert float(await redis.get(prefix + "spend:team:team")) == 20


@pytest.mark.asyncio
async def test_settlement_endpoint_recovers_private_event_and_never_regresses_unknown(store, monkeypatch):
    import time
    import jwt
    import httpx
    from fastapi import FastAPI
    from litellm.proxy.video_endpoints import moderation_execution as execution, moderation_metering_runtime as runtime

    meter, db, redis, prefix = store
    bound = binding()
    await prepare_meter(meter, bound, {})
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", "synthetic-settlement-secret-long-enough-32-bytes")
    private = [event("submit", "0"), event("completion", "20")]
    proof = dict(
        binding=bound.model_dump(mode="json"),
        native_id="native",
        events={e.phase: e.model_dump(mode="json") for e in private},
    )

    async def platform(request, method, path, payload):
        assert path == "/intents/intent/settlement-authority"
        return proof

    monkeypatch.setattr(execution.bridge, "platform", platform)
    app = FastAPI()
    app.include_router(execution.router)

    def ticket(purpose):
        return jwt.encode(
            dict(
                intent_id="intent",
                request_digest="digest",
                input_digest="input",
                model="video",
                policy_version="v1",
                policy_digest="policy",
                purpose=purpose,
                iat=int(time.time()),
                exp=int(time.time()) + 60,
                aud="moderation-fork",
            ),
            "synthetic-settlement-secret-long-enough-32-bytes",
            algorithm="HS256",
        )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://synthetic") as client:
        assert (
            await client.post("/internal/moderation/settlement", json={"ticket": ticket("settlement")})
        ).status_code == 401
        headers = {"Authorization": "Bearer synthetic-settlement-secret-long-enough-32-bytes"}
        assert (
            await client.post("/internal/moderation/settlement", headers=headers, json={"ticket": ticket("collect")})
        ).status_code == 403
        response = await client.post(
            "/internal/moderation/settlement", headers=headers, json={"ticket": ticket("settlement")}
        )
        assert response.status_code == 200 and response.json()["complete"] is False
        consumer = runtime.RecoveryConsumer(meter, interval=0.01)
        await consumer.start()
        try:
            for _ in range(100):
                if (await meter.settlement(bound)).complete:
                    break
                await asyncio.sleep(0.02)
        finally:
            await consumer.stop()
        proof["events"]["completion"] = (
            private[1].model_copy(update={"amount": None, "finalized": False}).model_dump(mode="json")
        )
        response = await client.post(
            "/internal/moderation/settlement", headers=headers, json={"ticket": ticket("settlement")}
        )
        assert response.status_code == 200
        assert response.json()["complete"] is True and Decimal(response.json()["total_actual"]) == Decimal("20")
        assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 20.0}]
        assert float(await redis.get(prefix + "spend:team:team")) == 20


@pytest.mark.asyncio
async def test_missing_moderated_outbox_proof_does_not_starve_valid_event(store, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
    from litellm.llms.libtv.billing_outbox import CausynBillingEvent, LibTVBillingReconciler

    meter, db, redis, prefix = store
    await create_financial_projection_tables(db)
    await redis.set(prefix + "spend:team:team", 100)
    await protected_store(meter).register_cutover(cutover_receipt())
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setattr(runtime, "counter_mode", AsyncMock(return_value=True))
    fields = dict(team_id="team", user_id="user", response_cost=3)
    missing = CausynBillingEvent(provider_task_id="missing-proof", moderation_intent_id="moderated", **fields)
    valid = CausynBillingEvent(provider_task_id="valid", **fields)
    stream = prefix + "actual-outbox"
    for event in (missing, valid):
        await redis.xadd(stream, {"payload": json.dumps(event.to_dict())})
    worker = LibTVBillingReconciler(redis, SimpleNamespace(db=db), stream_key=stream, consumer_group=prefix + "group")
    assert await worker.reconcile_once() == 1
    assert await worker.reconcile_once() == 0
    assert (await redis.xpending(stream, prefix + "group"))["pending"] == 1
    assert await db.query_raw('SELECT request_id,spend FROM "LiteLLM_SpendLogs"') == [
        {"request_id": valid.request_id, "spend": 3.0}
    ]


@pytest_asyncio.fixture(loop_scope="function")
async def production_store():
    schema = "entry_schema_" + uuid4().hex
    url = os.environ["MODERATION_METERING_POSTGRES_URL"]
    control = Prisma(datasource={"url": url})
    db = Prisma(datasource={"url": url + "?schema=" + schema + "&connection_limit=8"})
    redis = Redis.from_url(os.environ["MODERATION_METERING_REDIS_URL"], decode_responses=True)
    prefix = schema + ":"
    try:
        await control.connect()
        await control.execute_raw(f'CREATE SCHEMA "{schema}"')
        process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "--no-sync",
            "prisma",
            "db",
            "push",
            "--schema",
            "litellm/proxy/schema.prisma",
            "--skip-generate",
            env={**os.environ, "DATABASE_URL": url + "?schema=" + schema},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 60)
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
        assert process.returncode == 0, (stdout.decode(), stderr.decode())
        await db.connect()
        await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\"(user_id) VALUES('user')")
        await db.execute_raw("INSERT INTO \"LiteLLM_TeamTable\"(team_id) VALUES('team')")
        await db.execute_raw(
            "INSERT INTO \"LiteLLM_VerificationToken\"(token,user_id,team_id) VALUES('key','user','team')"
        )
        await db.execute_raw("INSERT INTO \"LiteLLM_TeamMembership\"(user_id,team_id) VALUES('user','team')")
        yield MeteringStore.from_client(db, redis, namespace=prefix), db, redis, prefix
    finally:
        try:
            async for key in redis.scan_iter(match=prefix + "*"):
                await redis.delete(key)
        finally:
            await redis.aclose()
            if db.is_connected():
                await db.disconnect()
            if control.is_connected():
                await control.execute_raw(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                await control.disconnect()


@pytest.mark.asyncio
async def test_phase_projection_matches_complete_production_prisma_schema(production_store):
    from datetime import datetime, timezone
    from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts

    meter, db, redis, prefix = production_store
    await prepare_meter(meter, binding(), {})
    facts = BillingFacts(
        started_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        ended_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        route="avideo_generation",
        prompt_tokens=30,
        completion_tokens=20,
    )
    phase = event().model_copy(update={"facts": facts})
    await meter.persist(phase)
    await asyncio.gather(*(meter.run_once() for _ in range(3)))
    await meter.persist(phase)
    await meter.run_once()
    assert await db.query_raw('SELECT spend,prompt_tokens,completion_tokens,total_tokens FROM "LiteLLM_SpendLogs"') == [
        {"spend": 20.0, "prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}
    ]
    for table in ("LiteLLM_DailyUserSpend", "LiteLLM_DailyTeamSpend"):
        assert await db.query_raw(f'SELECT spend,prompt_tokens,completion_tokens,api_requests FROM "{table}"') == [
            {"spend": 20.0, "prompt_tokens": 30, "completion_tokens": 20, "api_requests": 1}
        ]
    assert float(await redis.get(prefix + "spend:team:team")) == 20

    from datetime import timedelta

    for updates in (
        {"prompt_tokens": 31},
        {"pricing": (("output_cost_per_task", Decimal(25)),)},
        {"ended_at": facts.ended_at + timedelta(seconds=1)},
    ):
        altered = phase.model_copy(update={"facts": facts.model_copy(update=updates)})
        assert altered.digest() != phase.digest()
        with pytest.raises(ValueError, match="replay conflict"):
            await meter.persist(altered)
    assert await db.query_raw('SELECT spend,api_requests FROM "LiteLLM_DailyTeamSpend"') == [
        {"spend": 20.0, "api_requests": 1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_first", [False, True])
async def test_image_own_reservation_and_actual_outbox_commute_once(store, monkeypatch, terminal_first):
    from datetime import datetime, timezone, timedelta
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.llms.libtv.billing_outbox import ImageBillingEvent, LibTVBillingReconciler
    from litellm.types.utils import ImageResponse

    meter, db, redis, prefix = store
    await create_financial_projection_tables(db)
    await redis.set(prefix + "spend:team:team", 100)
    authority = protected_store(meter)
    await authority.register_cutover(cutover_receipt())
    await authority.mutate(
        "image-own-reserve",
        ("spend:team:team",),
        kind="reserve",
        amount=Decimal(5),
        reservation_id="image-own",
        valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setattr(runtime, "counter_mode", AsyncMock(return_value=True))
    reservation = {
        "reservation_id": "image-own",
        "reserved_cost": 5,
        "entries": [{"counter_key": "spend:team:team", "reservation_id": "image-own", "reserved_cost": 5}],
    }
    from litellm.llms.libtv.handler import LibTVLLM

    handler = LibTVLLM()
    provider = SimpleNamespace(aresolve_model_spec=AsyncMock(return_value={}))
    submit = AsyncMock(return_value=SimpleNamespace(to_dict=lambda: {"task_id": "image-own-task"}))
    monkeypatch.setattr(handler, "_make_client", lambda *args, **kwargs: provider)
    monkeypatch.setattr(handler, "asubmit_image_upscale", submit)
    response = await handler.aimage_generation(
        model="topaz-image-upscaler",
        prompt="",
        model_response=ImageResponse(data=[]),
        api_key="synthetic",
        api_base=None,
        optional_params={"libtv_image_upscale_submit": True},
        logging_obj=None,
        client=object(),
    )
    submit.assert_awaited_once()
    assert response._hidden_params["response_cost"] == 0
    assert getattr(response, runtime.DEFERRED_KEY) is runtime.Ownership.DEFERRED_IMAGE

    event = ImageBillingEvent(
        deployment_id="deployment",
        provider_task_id="image-own-task",
        response_cost=3,
        api_key="key",
        team_id="team",
        user_id="user",
    )
    worker = LibTVBillingReconciler(redis, SimpleNamespace(db=db))

    async def release():
        await _ProxyDBLogger()._PROXY_track_cost_callback(
            {"litellm_params": {"metadata": {"user_api_key_budget_reservation": reservation}}}, response
        )

    if terminal_first:
        await worker._commit_event(event)
        await release()
    else:
        await release()
        await worker._commit_event(event)
    await worker._commit_event(event)
    await release()
    assert float(await redis.get(prefix + "spend:team:team")) == 103
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 3.0}]
    assert await db.query_raw('SELECT request_id,spend FROM "LiteLLM_SpendLogs"') == [
        {"request_id": event.request_id, "spend": 3.0}
    ]
    assert reservation["finalized"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted,collect_first", [(False, False), (True, False), (True, True)])
async def test_standalone_context_ir_endpoint_rewrite_outbox_production_schema_once(
    production_store, monkeypatch, admitted, collect_first
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from fastapi import Request
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.video_endpoints import (
        context_ir_endpoints as endpoint,
        moderation_bridge,
        moderation_metering_entry as entry,
        moderation_metering_runtime as runtime,
    )
    from litellm.llms.causyn import context_ir, context_ir_store
    from litellm.llms.causyn.h3_prompt import AUTH_MODEL, ContextIRRequest, RewriteResult, RewriteUsage
    from litellm.llms.libtv import billing_outbox

    meter, db, redis, prefix = production_store
    fingerprint = "b" * 64
    await db.execute_raw('UPDATE "LiteLLM_VerificationToken" SET token=$1', fingerprint)
    baseline = binding().model_copy(
        update={
            "intent_id": "seed",
            "fingerprint": fingerprint,
            "model": AUTH_MODEL,
            "actor_user_id": "actor",
            "expected_phases": ("completion",),
        }
    )
    await prepare_meter(meter, baseline, {})
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setattr(runtime, "counter_mode", AsyncMock(return_value=True))
    monkeypatch.setattr(moderation_bridge, "submit", AsyncMock(return_value=None))
    monkeypatch.setattr(context_ir, "get_transfer_redis", lambda url: redis)
    monkeypatch.setattr(context_ir_store, "PENDING", prefix + "pending")
    monkeypatch.setattr(context_ir_store, "READY", prefix + "ready")
    monkeypatch.setattr(context_ir_store, "task_key", lambda task_id: prefix + "task:" + task_id)
    monkeypatch.setattr(context_ir_store, "owner_key", lambda owner: prefix + "owner:" + owner)
    monkeypatch.setattr(billing_outbox, "CAUSYN_BILLING_STREAM_KEY", prefix + "outbox")
    monkeypatch.setattr(billing_outbox, "CAUSYN_BILLING_MARKER_PREFIX", prefix + "enqueued:")
    rewrite = AsyncMock(
        return_value=RewriteResult(
            prompt="integrated_multimodal_description: [Shot 1] A cat walks.\noverall_soundscape: Quiet.\nnon_diegetic_music: None.",
            usage=RewriteUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50, cost=0.001),
            system_sha256="a" * 64,
        )
    )
    import time

    clock = [time.time()]
    monkeypatch.setattr(context_ir, "time", SimpleNamespace(time=lambda: clock[0], time_ns=lambda: int(clock[0] * 1e9)))
    delivered_events = []

    async def settle(task):
        persisted = await service.store.get(task.id)
        phase = context_ir.financial_event(persisted)
        if phase:
            delivered_events.append(phase)
        await context_ir.settle_task(task)
        clock[0] += 5

    service = context_ir.ContextIRService(context_ir_store.ContextIRStore(redis), rewrite=rewrite, settle=settle)
    request = Request(
        {"type": "http", "method": "POST", "path": "/v2/h3_context_ir", "headers": [], "query_string": b""}
    )
    request.scope["causyn_context_ir_spec"] = ContextIRRequest.model_validate(
        {"model": "MiniMax-H3", "content": [{"type": "text", "text": "A cat walks."}], "duration": 5, "ratio": "16:9"}
    )
    if admitted:
        await db.execute_raw(
            'UPDATE "LiteLLM_TeamTable" SET budget_limits=$1::jsonb',
            json.dumps([{"budget_duration": "1d", "max_budget": 100, "reset_at": "2026-09-20T00:00:00Z"}]),
        )
        entry.attest(
            request, {"intent_id": "entry", "request_digest": "digest", "actor_user_id": "actor", "model": AUTH_MODEL}
        )
    response = await endpoint.create_context_ir(
        request, UserAPIKeyAuth(api_key=fingerprint, user_id="user", team_id="team"), service
    )
    task_id = response["task_id"]
    await service.process(task_id)
    await service.process(task_id)
    task = await service.store.get(task_id)
    assert task.status == "succeeded" and task.settled and task.price == 4
    assert rewrite.await_count == 1
    assert await redis.xlen(prefix + "outbox") == 1
    worker = billing_outbox.LibTVBillingReconciler(
        redis, SimpleNamespace(db=db), stream_key=prefix + "outbox", consumer_group=prefix + "group"
    )
    collected_events = []

    async def collect():
        if not admitted:
            return
        from litellm.proxy.video_endpoints import moderation_execution as execution
        from litellm.proxy.video_endpoints.moderation_metering import PhaseEvent

        monkeypatch.setattr(context_ir, "get_context_ir_service", lambda: service)
        monkeypatch.setattr(execution, "authorize", lambda *args: SimpleNamespace(intent_id="entry"))

        async def platform(request, method, path, body):
            if path.endswith("/execution"):
                return {
                    "state": "submitted",
                    "route": "context_ir",
                    "native_id": task_id,
                    "principal": {"fingerprint": fingerprint, "user_id": "user", "team_id": "team"},
                }
            if path.endswith("/financial-event"):
                phase = PhaseEvent.model_validate(body["metering_event"])
                collected_events.append(phase)
                assert phase.digest() == delivered_events[0].digest()
                await meter.persist(phase)
                await meter.run_once()
            return {}

        monkeypatch.setattr(execution.bridge, "platform", platform)
        await execution.collect(execution.Ticket(ticket="synthetic"), request)

    if collect_first:
        await collect()
    assert await worker.reconcile_once() == 1
    clock[0] += 5
    await collect()
    await context_ir.settle_task(task)
    await collect()
    assert await worker.reconcile_once() == 0
    if admitted:
        assert all(phase == delivered_events[0] for phase in collected_events)
        assert task.updated_at > delivered_events[0].facts.ended_at.timestamp()
    assert await db.query_raw('SELECT spend,prompt_tokens,completion_tokens,total_tokens FROM "LiteLLM_SpendLogs"') == [
        {"spend": 4.0, "prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}
    ]
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 4.0}]
    assert float(await redis.get(prefix + "spend:team:team")) == 4
    if admitted:
        bound = await meter.binding("entry")
        assert bound.expected_phases == ("completion",)
        assert len(bound.windows) == 1
        assert (await meter.settlement(bound)).total_actual == Decimal(4)
        assert float(await redis.get(prefix + "spend:team:team:window:1d")) == 4
    else:
        assert task.metering_binding_json is None
        assert await db.query_raw('SELECT intent_id FROM "LiteLLM_ModerationMeteringTask"') == [{"intent_id": "seed"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("event_kind", ["image", "causyn", "context_ir"])
@pytest.mark.parametrize("global_state", ["ready", "born", "unregistered"])
async def test_entry_fix1_actual_outbox_freezes_global_identity(store, monkeypatch, event_kind, global_state):
    import json
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
    from litellm.proxy.spend_tracking.protected_budget import BudgetIdentity, CounterBaseline
    from litellm.llms.libtv.billing_outbox import ImageBillingEvent, CausynBillingEvent, LibTVBillingReconciler

    meter, db, redis, prefix = store
    await create_financial_projection_tables(db)
    authority = protected_store(meter)
    if global_state != "born":
        await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('global-budget')")
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())
    if global_state == "born":
        await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('global-budget')")
    global_receipt = cutover_receipt().model_copy(
        update={
            "receipt_id": "global-cutover",
            "baselines": (
                CounterBaseline(
                    target=BudgetIdentity(kind="user", identity="global-budget"),
                    sql_value=0,
                    period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
                ),
            ),
        }
    )
    if global_state == "ready":
        await authority.register_cutover(global_receipt)
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setattr(runtime, "counter_mode", AsyncMock(return_value=True))
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "global-budget")
    monkeypatch.setattr(litellm, "max_budget", 100)
    fields = dict(
        deployment_id="deployment",
        provider_task_id="actual-task",
        response_cost=3,
        api_key="key",
        team_id="team",
        user_id="user",
        occurred_at="2026-09-13T00:00:00+00:00",
    )
    event = (
        ImageBillingEvent(**fields)
        if event_kind == "image"
        else CausynBillingEvent(
            **fields, task_type="h3_context_ir" if event_kind == "context_ir" else "video_generation"
        )
    )
    stream = prefix + "global-outbox"
    await redis.xadd(stream, {"payload": json.dumps(event.to_dict())})
    worker = LibTVBillingReconciler(redis, SimpleNamespace(db=db), stream_key=stream, consumer_group=prefix + "group")
    if global_state == "ready":
        await asyncio.gather(*(worker._commit_event(event) for _ in range(3)))
    if global_state == "unregistered":
        assert await worker.reconcile_once() == 0
        assert (await redis.xpending(stream, prefix + "group"))["pending"] == 1
        assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 0.0}]
        assert await db.query_raw('SELECT request_id FROM "LiteLLM_SpendLogs"') == []
        monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "different-global")
        monkeypatch.setattr(litellm, "max_budget", 0)
        await authority.register_cutover(global_receipt)
        assert await worker.reconcile_once() == 1
        assert (await redis.xpending(stream, prefix + "group"))["pending"] == 0
    else:
        assert await worker.reconcile_once() == 1
    await worker._commit_event(event)
    operation = await authority.operation("legacy:" + event.request_id)
    assert operation.payload.projection.global_user_id == "global-budget"
    from litellm.proxy.video_endpoints.moderation_metering_projection import freeze_legacy

    projection = operation.payload.projection
    for changes in (
        {"fingerprint": "other-key"},
        {"user_id": "other-user"},
        {"team_id": "other-team"},
        {"provider": "other-provider"},
        {"deployment_id": "other-deployment"},
        {"provider_task_id": "other-task"},
        {"amount": Decimal(4)},
        {"facts": projection.facts.model_copy(update={"prompt_tokens": 42})},
    ):
        with pytest.raises(ValueError, match="admission replay conflict"):
            await freeze_legacy(meter.db, projection.model_copy(update=changes))
    assert len(await db.query_raw('SELECT request_id FROM "LiteLLM_LegacyFinancialAdmission"')) == 1
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "different-global")
    monkeypatch.setattr(litellm, "max_budget", 0)
    await asyncio.gather(*(worker._commit_event(event) for _ in range(3)))
    assert (await authority.operation("legacy:" + event.request_id)).payload_hash == operation.payload_hash
    assert await db.query_raw("SELECT spend FROM \"LiteLLM_UserTable\" WHERE user_id='global-budget'") == [
        {"spend": 3.0}
    ]
    assert float(await redis.get(prefix + "spend:user:global-budget")) == 3
    assert float(await redis.get(prefix + "spend:team:team")) == 103
    assert await db.query_raw('SELECT spend FROM "LiteLLM_SpendLogs"') == [{"spend": 3.0}]


async def accepted_capture_entry(store, monkeypatch, failure, source="public", proof=None):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import httpx
    import litellm
    from fastapi import FastAPI, Request, Response
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.video_endpoints import moderation_execution as execution, moderation_metering_entry as entry
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime, endpoints, openapi_log_capture
    from litellm.types.videos.main import VideoObject
    from litellm.types.videos.utils import encode_video_id_with_provider
    from litellm.utils import client
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    meter, db, redis, prefix = store
    for table in ("LiteLLM_VerificationToken", "LiteLLM_TeamTable"):
        await db.execute_raw(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS budget_limits jsonb')
    bound = binding().model_copy(update={"fingerprint": "b" * 64, "actor_user_id": "actor"})
    if proof is not None:
        bound = BillingBinding(**proof, expected_phases=("submit", "completion"))
    await db.execute_raw('UPDATE "LiteLLM_VerificationToken" SET token=$1 WHERE token=$2', bound.fingerprint, "key")
    await prepare_meter(meter, bound, {})
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setattr(litellm, "max_budget", 0)
    monkeypatch.setenv("DRAMA_MODERATION_SERVICE_TOKEN", "synthetic-entry-fix1-secret-32-bytes")
    monkeypatch.setenv("DRAMA_MODERATION_PLATFORM_URL", "http://platform.synthetic.invalid")
    auth = UserAPIKeyAuth(api_key=bound.fingerprint, user_id=bound.user_id, team_id=bound.team_id)
    monkeypatch.setattr(execution, "authenticate", AsyncMock(return_value=auth))
    monkeypatch.setattr(execution, "authorize", lambda *args: SimpleNamespace(intent_id=bound.intent_id))
    state = {"state": "input_moderation", "not_sent": [], "receipts": [], "entry_capture": 0}
    admission = {
        "intent_id": bound.intent_id,
        "request_digest": bound.request_digest,
        "actor_user_id": bound.actor_user_id,
        "model": bound.model,
        "generation_id": bound.generation_id,
    }

    async def platform(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/begin"):
            if state["state"] != "input_moderation":
                return httpx.Response(200, json={"acquired": False, "state": state["state"]})
            state["state"] = "submission_unknown"
            return httpx.Response(
                200,
                json={
                    "acquired": True,
                    "token": "attempt",
                    "metering": admission,
                    "credential": "synthetic",
                    "route": "avideo_generation",
                    "request": {"model": bound.model, "prompt": "synthetic"},
                },
            )
        if request.url.path.endswith("/not-sent"):
            state["not_sent"].append(payload)
            state["state"] = "input_moderation"
            return httpx.Response(200, json={})
        assert request.url.path.endswith(("/receipt", "/generation-receipt"))
        state["receipts"].append(payload)
        if runtime.CONTEXT.get() is not None:
            state["entry_capture"] += 1
        if failure == "platform4xx":
            return httpx.Response(403, json={"error": "synthetic lost admission"})
        error = {
            "connect": httpx.ConnectError,
            "connect_timeout": httpx.ConnectTimeout,
            "pool_timeout": httpx.PoolTimeout,
        }.get(failure, httpx.ConnectError)("synthetic receipt loss", request=request)
        if failure in ("nested_cause", "nested_context"):
            try:
                raise error
            except httpx.ConnectError:
                if failure == "nested_cause":
                    raise RuntimeError("nested financial transport") from error
                raise RuntimeError("nested financial transport") from None
        raise error

    app = FastAPI()
    app.state.moderation_transport = httpx.MockTransport(platform)
    app.include_router(execution.router)
    responses = []
    attempts = []

    @client
    async def avideo_generation(**kwargs):
        attempts.append("provider")
        if failure == "provider_connect":
            raise httpx.ConnectError("synthetic provider not reached")
        if failure == "provider_rejected":
            rejected = httpx.Response(429, request=httpx.Request("POST", "http://provider.synthetic.invalid"))
            rejected.raise_for_status()
        response = VideoObject(
            id=encode_video_id_with_provider("native", "openai", "deployment"), object="video", status="queued"
        )
        response._hidden_params["response_cost"] = 3.75
        responses.append(response)
        return response

    monkeypatch.setattr(litellm, "avideo_generation", avideo_generation)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "user_model", "openai/synthetic")
    logging = SimpleNamespace(litellm_call_id="entry-call")

    async def pre_call(self, **kwargs):
        self.data.update(model="openai/synthetic", caching=False)
        return self.data, logging

    monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "_pre_call_with_fallbacks", pre_call)
    monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "get_custom_headers", lambda **kwargs: {})
    proxy_logging = SimpleNamespace(
        during_call_hook=AsyncMock(),
        update_request_status=AsyncMock(),
        post_call_success_hook=AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
        post_call_response_headers_hook=AsyncMock(return_value={}),
        post_call_failure_hook=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", proxy_logging)
    monkeypatch.setattr(openapi_log_capture, "start", AsyncMock(return_value=None))
    try:
        if source == "public":
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fork"
            ) as http:
                first = await http.post("/internal/moderation/submit", json={"ticket": "synthetic"})
                if failure.startswith("provider_"):
                    assert first.status_code == (429 if failure == "provider_rejected" else 500)
                    assert len(attempts) == 1 and responses == []
                    assert state["not_sent"] == [
                        {"token": "attempt", "outcome": "not_sent" if failure == "provider_connect" else "rejected"}
                    ]
                    assert state["receipts"] == []
                    return None, bound, state
                assert first.status_code == 200, (first.text, state, len(responses))
                second = await http.post("/internal/moderation/submit", json={"ticket": "synthetic"})
                assert second.status_code == 200
                assert state["state"] == "submission_unknown"
        else:
            request = execution.execution_request(
                Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/videos",
                        "headers": [],
                        "query_string": b"",
                        "app": app,
                    }
                ),
                {"model": bound.model},
                "/v1/videos",
            )
            entry.attest(request, admission)
            request.scope["moderation_producer_ticket"] = "synthetic-producer"
            returned = await endpoints.video_generation(request, Response(), None, auth)
            assert returned is responses[0]
        assert len(responses) == 1
        assert state["entry_capture"] == 1
        assert state["not_sent"] == []
        assert runtime.private_event(responses[0]) is not None
        phase = await meter.phase(bound.intent_id, "submit")
        assert phase.native_id == responses[0].id
        assert phase.amount == Decimal("3.75")
        state["provider_calls"] = len(responses)
        return responses[0], bound, state
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["connect", "connect_timeout", "pool_timeout", "nested_cause", "nested_context", "platform4xx"]
)
@pytest.mark.parametrize("source", ["public", "studio"])
async def test_entry_fix1_capture_failure_after_actual_invoke_never_reopens_dispatch(
    store, monkeypatch, failure, source
):
    await create_financial_projection_tables(store[1])
    await accepted_capture_entry(store, monkeypatch, failure, source)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider_connect", "provider_rejected"])
async def test_entry_fix1_actual_pre_provider_failure_keeps_trusted_outcome(store, monkeypatch, failure):
    await accepted_capture_entry(store, monkeypatch, failure)


@pytest.mark.asyncio
async def test_entry_fix1_historical_operation_precedes_new_global_configuration(store, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
    from litellm.llms.libtv.billing_outbox import ImageBillingEvent, LibTVBillingReconciler

    meter, db, redis, prefix = store
    await create_financial_projection_tables(db)
    await db.execute_raw("INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('global-budget')")
    authority = protected_store(meter)
    await redis.set(prefix + "spend:team:team", 100)
    await authority.register_cutover(cutover_receipt())
    monkeypatch.setattr(runtime, "store", lambda: meter)
    monkeypatch.setattr(runtime, "counter_mode", AsyncMock(return_value=True))
    monkeypatch.setattr(litellm, "max_budget", 0)
    monkeypatch.setattr(proxy_server, "litellm_proxy_budget_name", "global-budget")
    event = ImageBillingEvent(
        deployment_id="deployment", provider_task_id="historical", response_cost=3, team_id="team", user_id="user"
    )
    worker = LibTVBillingReconciler(redis, SimpleNamespace(db=db))
    await worker._commit_event(event)
    original = await authority.operation("legacy:" + event.request_id)
    assert original.payload.projection.global_user_id is None
    await db.execute_raw('DELETE FROM "LiteLLM_LegacyFinancialAdmission" WHERE request_id=$1', event.request_id)
    monkeypatch.setattr(litellm, "max_budget", 100)
    await worker._commit_event(event)
    assert (await authority.operation("legacy:" + event.request_id)).payload_hash == original.payload_hash
    assert await db.query_raw("SELECT spend FROM \"LiteLLM_UserTable\" WHERE user_id='global-budget'") == [
        {"spend": 0.0}
    ]
    assert await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"') == [{"spend": 3.0}]
    assert await redis.get(prefix + "spend:user:global-budget") is None
    assert float(await redis.get(prefix + "spend:team:team")) == 103
