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
    not os.getenv('MODERATION_METERING_POSTGRES_URL') or not os.getenv('MODERATION_METERING_REDIS_URL'),
    reason='isolated PostgreSQL and Redis required',
)


@pytest_asyncio.fixture(loop_scope='function')
async def store():
    schema = 'meter_' + uuid4().hex
    url = os.environ['MODERATION_METERING_POSTGRES_URL']
    control = Prisma(datasource={'url': url})
    db = Prisma(datasource={'url': url + '?schema=' + schema + '&connection_limit=8'})
    redis = Redis.from_url(os.environ['MODERATION_METERING_REDIS_URL'], decode_responses=True)
    namespace = schema + ':'
    try:
        await control.connect()
        await control.execute_raw(f'CREATE SCHEMA "{schema}"')
        await db.connect()
        migration = Path('litellm-proxy-extras/litellm_proxy_extras/migrations/20260913000000_moderation_metering/migration.sql')
        for statement in migration.read_text().split(';'):
            if statement.strip():
                await db.execute_raw(statement)
        for statement in (
            'CREATE TABLE "LiteLLM_VerificationToken" (token text primary key, user_id text, team_id text, organization_id text, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_TeamTable" (team_id text primary key, organization_id text, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_UserTable" (user_id text primary key, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_OrganizationTable" (organization_id text primary key, spend double precision default 0)',
            'CREATE TABLE "LiteLLM_TeamMembership" (user_id text, team_id text, spend double precision default 0, total_spend double precision default 0, primary key(user_id,team_id))',
            'INSERT INTO "LiteLLM_VerificationToken" (token,user_id,team_id) VALUES (\'key\',\'user\',\'team\')',
            'INSERT INTO "LiteLLM_TeamTable" (team_id) VALUES (\'team\')',
            'INSERT INTO "LiteLLM_UserTable" (user_id) VALUES (\'user\')',
            'INSERT INTO "LiteLLM_TeamMembership" (user_id,team_id) VALUES (\'user\',\'team\')',
        ):
            await db.execute_raw(statement)
        yield MeteringStore.from_client(db, redis, namespace=namespace), db, redis, namespace
    finally:
        async for key in redis.scan_iter(match=namespace + '*'):
            await redis.delete(key)
        await redis.aclose()
        if db.is_connected():
            await db.disconnect()
        if control.is_connected():
            await control.execute_raw(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await control.disconnect()


def binding():
    return BillingBinding(intent_id='intent', request_digest='digest', fingerprint='key', user_id='user',
                          team_id='team', model='video', expected_phases=('submit', 'completion'))


def event(phase='completion', amount='20'):
    return PhaseEvent(binding=binding(), request_id='public-video:intent:' + phase, phase=phase,
                      provider='libtv', deployment_id='deployment', native_id='native',
                      provider_task_id='provider-task', amount=None if amount is None else Decimal(amount),
                      finalized=amount is not None)


@pytest.mark.asyncio
async def test_actual_receipt_concurrent_replay_and_complete_zero_phase(store):
    meter, db, redis, prefix = store
    await meter.prepare(binding(), {})
    await meter.persist(event('submit', '0'))
    await meter.persist(event())
    await asyncio.gather(*(meter.run_once() for _ in range(6)))
    envelope = await meter.settlement(binding())
    assert envelope.complete
    assert envelope.total_actual == Decimal('20')
    assert len(envelope.receipts) == 2
    for _ in range(2):
        await meter.persist(event())
        await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{'spend': 20.0}]
    assert float(await redis.get(prefix + 'spend:team:team')) == 20


@pytest.mark.asyncio
async def test_unknown_cost_and_incomplete_manifest_never_mean_zero(store):
    meter, _, _, _ = store
    await meter.prepare(binding(), {})
    await meter.persist(event('submit', '0'))
    await meter.persist(event(amount=None))
    await meter.run_once()
    assert not (await meter.settlement(binding())).complete
    await meter.persist(event(amount='0'))
    await meter.run_once()
    result = await meter.settlement(binding())
    assert result.complete and result.total_actual == 0


@pytest.mark.asyncio
async def test_legacy_additive_counter_is_preserved(store):
    meter, _, redis, prefix = store
    await redis.set(prefix + 'spend:team:team', '100')
    await meter.prepare(binding(), {})
    await meter.persist(event())
    await meter.run_once()
    assert float(await redis.get(prefix + 'spend:team:team')) == 120


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['user', 'team', 'missing'])
async def test_wrong_debit_identity_rolls_back_without_receipt(store, change):
    meter, db, _, _ = store
    await meter.prepare(binding(), {})
    await meter.persist(event())
    if change == 'missing':
        await db.execute_raw('DELETE FROM "LiteLLM_UserTable"')
    else:
        await db.execute_raw(f'UPDATE "LiteLLM_VerificationToken" SET {change}_id=\'other\'')
    with pytest.raises(ValueError):
        await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{'spend': 0.0}]
    assert not (await meter.settlement(binding())).complete


@pytest.mark.asyncio
async def test_payload_and_actor_replay_conflicts(store):
    meter, _, _, _ = store
    await meter.prepare(binding(), {})
    await meter.persist(event())
    with pytest.raises(ValueError):
        await meter.persist(event(amount='21'))
    with pytest.raises(ValueError):
        await meter.prepare(binding().model_copy(update={'user_id': 'attacker'}), {})


@pytest.mark.asyncio
async def test_evicted_active_counter_requires_reconciliation(store):
    meter, db, redis, prefix = store
    await meter.prepare(binding(), {})
    await meter.persist(event())
    await redis.delete(prefix + 'spend:team:team')
    with pytest.raises(ValueError):
        await meter.run_once()
    rows = await db.query_raw('SELECT status FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=\'spend:team:team\'')
    assert rows == [{'status': 'reconciliation_required'}]
    with pytest.raises(ValueError):
        await meter.assert_admission(('spend:team:team',))


@pytest.mark.asyncio
async def test_sql_rollback_after_redis_adjustment_recovers_same_event(store):
    meter, db, redis, prefix = store
    await meter.prepare(binding(), {})
    await meter.persist(event())

    class FailingTransaction:
        def __init__(self, transaction):
            self.transaction = transaction

        async def query_raw(self, sql, *args):
            return await self.transaction.query_raw(sql, *args)

        async def execute_raw(self, sql, *args):
            if 'SET status=\'settled\'' in sql:
                raise RuntimeError('synthetic SQL failure after debit')
            return await self.transaction.execute_raw(sql, *args)

    @asynccontextmanager
    async def transactions():
        async with meter.transactions() as tx:
            yield FailingTransaction(tx)

    interrupted = MeteringStore(meter.db, transactions, meter.redis, namespace=prefix)
    with pytest.raises(RuntimeError):
        await interrupted.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{'spend': 0.0}]
    assert float(await redis.get(prefix + 'spend:team:team')) == 20
    with pytest.raises(ValueError, match='pending'):
        await meter.assert_admission(('spend:team:team',))
    assert (await db.query_raw('SELECT DISTINCT status FROM "LiteLLM_ModerationMeteringCounter"')) == [{'status': 'ready'}]
    await db.execute_raw('UPDATE "LiteLLM_ModerationMeteringPhase" SET available_at=NOW()')
    await MeteringStore.from_client(db, redis, namespace=prefix).run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{'spend': 20.0}]
    assert float(await redis.get(prefix + 'spend:team:team')) == 20


@pytest.mark.asyncio
async def test_redis_lost_reply_does_not_repeat_reservation_adjustment(store):
    from litellm.proxy.video_endpoints import moderation_metering_cache as cache

    meter, db, redis, prefix = store
    await redis.set(prefix + 'spend:team:team', 105)
    await meter.prepare(binding(), {'spend:team:team': Decimal(5)})
    await meter.persist(event('submit'))

    class LostReply:
        async def eval(self, script, numkeys, *args):
            value = await redis.eval(script, numkeys, *args)
            if script == cache.APPLY:
                raise RuntimeError('synthetic lost Lua reply')
            return value

    interrupted = MeteringStore(meter.db, meter.transactions, LostReply(), namespace=prefix)
    with pytest.raises(RuntimeError):
        await interrupted.run_once()
    assert float(await redis.get(prefix + 'spend:team:team')) == 120
    await db.execute_raw('UPDATE "LiteLLM_ModerationMeteringPhase" SET available_at=NOW()')
    await meter.run_once()
    assert float(await redis.get(prefix + 'spend:team:team')) == 120
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{'spend': 20.0}]


@pytest.mark.asyncio
async def test_lost_sql_commit_reply_replays_receipt_without_cache_call(store):
    meter, db, redis, prefix = store
    await meter.prepare(binding(), {})
    await meter.persist(event())

    @asynccontextmanager
    async def lost_reply():
        async with meter.transactions() as tx:
            yield tx
        raise RuntimeError('synthetic lost SQL commit reply')

    interrupted = MeteringStore(meter.db, lost_reply, meter.redis, namespace=prefix)
    with pytest.raises(RuntimeError):
        await interrupted.run_once()
    await meter.persist(event())
    assert not await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{'spend': 20.0}]
    assert float(await redis.get(prefix + 'spend:team:team')) == 20


@pytest.mark.asyncio
async def test_old_backup_and_phase_identity_mismatch_fail_closed(store):
    meter, db, redis, prefix = store
    await meter.prepare(binding(), {})
    await meter.persist(event())
    await meter.run_once()
    with pytest.raises(ValueError, match='provider binding'):
        await meter.persist(event('submit', '0').model_copy(update={'native_id': 'different-task'}))
    await redis.hset(prefix + 'moderation:counter:spend:team:team', 'seq', '0')
    with pytest.raises(ValueError):
        await meter.assert_admission(('spend:team:team',))
    assert (await db.query_raw('SELECT status FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=\'spend:team:team\'')) == [{'status': 'reconciliation_required'}]


@pytest.mark.asyncio
async def test_team_organization_binding_cannot_change_before_debit(store):
    meter, db, _, _ = store
    await meter.prepare(binding(), {})
    await meter.persist(event())
    await db.execute_raw('UPDATE "LiteLLM_TeamTable" SET organization_id=\'foreign-org\'')
    with pytest.raises(ValueError, match='organization'):
        await meter.run_once()
    assert (await db.query_raw('SELECT spend FROM "LiteLLM_TeamTable"')) == [{'spend': 0.0}]


@pytest.mark.asyncio
async def test_unknown_placeholder_rejects_noncanonical_request_id(store):
    meter, _, _, _ = store
    await meter.prepare(binding(), {})
    with pytest.raises(ValueError, match='request'):
        await meter.persist(event().model_copy(update={'request_id': 'unbound-phase'}))
