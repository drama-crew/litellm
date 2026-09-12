from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.proxy.video_endpoints import moderation_metering_cache as cache
from litellm.proxy.video_endpoints.openapi_log_capture import RawDatabase, raw_method
from litellm.proxy.video_endpoints.openapi_logs import Database


class BillingBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    intent_id: str
    generation_id: str | None = None
    request_digest: str
    fingerprint: str
    user_id: str
    actor_user_id: str | None = None
    team_id: str
    organization_id: str | None = None
    model: str
    expected_phases: tuple[str, ...]


class PhaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    binding: BillingBinding
    request_id: str
    phase: str
    provider: str
    deployment_id: str
    native_id: str
    provider_task_id: str
    amount: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    unit: Literal['USD'] = 'USD'
    finalized: bool = False

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class SettlementEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)
    binding: BillingBinding
    receipts: tuple[PhaseEvent, ...]
    complete: bool
    total_actual: Decimal | None
    unit: Literal['USD'] = 'USD'


class CounterRow(cache.CounterVersion):
    status: str


class PhaseRow(BaseModel):
    request_id: str
    payload: PhaseEvent
    payload_hash: str
    status: str
    lease_token: str | None = None
    lease_until: datetime | None = None
    receipt: PhaseEvent | None = None


class TaskRow(BaseModel):
    binding: BillingBinding
    reservations: dict[str, Decimal]


class IdentityRow(BaseModel):
    token: str
    user_id: str | None
    team_id: str | None
    organization_id: str | None
    spend: Decimal


class SpendRow(BaseModel):
    spend: Decimal


class TeamIdentity(BaseModel):
    organization_id: str | None


class CounterFailure(ValueError):
    def __init__(self, counter_keys: tuple[str, ...], reason: str):
        super().__init__('moderation metering cache reconciliation required: ' + reason)
        self.counter_keys = counter_keys


def _json(value: dict[str, JsonValue]) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def base_counter_keys(binding: BillingBinding) -> tuple[str, ...]:
    return tuple(sorted((
        'spend:key:' + binding.fingerprint,
        'spend:user:' + binding.user_id,
        'spend:team:' + binding.team_id,
        'spend:team_member:' + binding.user_id + ':' + binding.team_id,
        *(('spend:org:' + binding.organization_id,) if binding.organization_id else ()),
    )))


def request_id(binding: BillingBinding, phase: str) -> str:
    return 'public-video:' + binding.intent_id + ':' + phase


class MeteringStore:
    def __init__(
        self, db: Database, transactions: Callable[[], AbstractAsyncContextManager[Database]],
        redis: cache.RedisCommands, *, namespace: str = ''
    ):
        self.db = db
        self.transactions = transactions
        self.redis = redis
        self.namespace = namespace

    @classmethod
    def from_client(cls, client: object, redis: cache.RedisCommands, *, namespace: str = '') -> MeteringStore:
        db = getattr(client, 'db', client)
        read = raw_method(db, 'query_raw')
        write = raw_method(db, 'execute_raw')
        factory = TypeAdapter(Callable[[], object]).validate_python(getattr(db, 'tx'))

        @asynccontextmanager
        async def transactions() -> AsyncIterator[Database]:
            async with cast(AbstractAsyncContextManager[object], factory()) as transaction:
                yield RawDatabase(raw_method(transaction, 'query_raw'), raw_method(transaction, 'execute_raw'))

        return cls(RawDatabase(read, write), transactions, redis, namespace=namespace)

    async def _identity(self, tx: Database, binding: BillingBinding) -> dict[str, Decimal]:
        rows = TypeAdapter(list[IdentityRow]).validate_python(await tx.query_raw(
            'SELECT token,user_id,team_id,organization_id,spend FROM "LiteLLM_VerificationToken" WHERE token=$1 FOR UPDATE',
            binding.fingerprint
        ))
        if len(rows) != 1 or (rows[0].user_id, rows[0].team_id, rows[0].organization_id) != (
            binding.user_id, binding.team_id, binding.organization_id
        ):
            raise ValueError('moderation metering key identity mismatch')
        teams = TypeAdapter(list[TeamIdentity]).validate_python(await tx.query_raw(
            'SELECT organization_id FROM "LiteLLM_TeamTable" WHERE team_id=$1 FOR UPDATE', binding.team_id
        ))
        if len(teams) != 1 or teams[0].organization_id != binding.organization_id:
            raise ValueError('moderation metering team organization mismatch')
        sources = (
            ('spend:user:' + binding.user_id, '"LiteLLM_UserTable"', 'user_id', binding.user_id),
            ('spend:team:' + binding.team_id, '"LiteLLM_TeamTable"', 'team_id', binding.team_id),
            *((('spend:org:' + binding.organization_id, '"LiteLLM_OrganizationTable"', 'organization_id', binding.organization_id),)
              if binding.organization_id else ()),
        )
        members = TypeAdapter(list[SpendRow]).validate_python(await tx.query_raw(
            'SELECT spend FROM "LiteLLM_TeamMembership" WHERE user_id=$1 AND team_id=$2 FOR UPDATE',
            binding.user_id, binding.team_id
        ))
        if len(members) != 1:
            raise ValueError('moderation metering member identity missing')

        async def balance(table: str, column: str, identity: str) -> Decimal:
            result = TypeAdapter(list[SpendRow]).validate_python(await tx.query_raw(
                f'SELECT COALESCE(spend,0) AS spend FROM {table} WHERE {column}=$1 FOR UPDATE', identity
            ))
            if len(result) != 1:
                raise ValueError('moderation metering debit identity missing')
            return result[0].spend

        return {
            'spend:key:' + binding.fingerprint: rows[0].spend,
            'spend:team_member:' + binding.user_id + ':' + binding.team_id: members[0].spend,
            **{key: await balance(table, column, identity) for key, table, column, identity in sources},
        }

    async def prepare(self, binding: BillingBinding, reservations: dict[str, Decimal]) -> None:
        if not binding.expected_phases or len(set(binding.expected_phases)) != len(binding.expected_phases):
            raise ValueError('unique billing phases required')
        try:
            async with self.transactions() as tx:
                await tx.execute_raw('SELECT pg_advisory_xact_lock(hashtext($1))', 'moderation:' + binding.intent_id)
                old = TypeAdapter(list[TaskRow]).validate_python(await tx.query_raw(
                    'SELECT binding,reservations FROM "LiteLLM_ModerationMeteringTask" WHERE intent_id=$1', binding.intent_id
                ))
                if old:
                    if old[0].binding != binding or old[0].reservations != reservations:
                        raise ValueError('moderation metering admission replay conflict')
                    return
                balances = await self._identity(tx, binding)
                if set(reservations) - set(balances):
                    raise ValueError('unsupported moderation reservation counter')
                for key in sorted(balances):
                    inserted = await tx.execute_raw(
                        'INSERT INTO "LiteLLM_ModerationMeteringCounter" (counter_key,generation) VALUES ($1,$2) '
                        'ON CONFLICT (counter_key) DO NOTHING', key, uuid4().hex
                    )
                    versions = TypeAdapter(list[CounterRow]).validate_python(await tx.query_raw(
                        'SELECT * FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=$1 FOR UPDATE', key
                    ))
                    version = versions[0]
                    if version.status != 'ready':
                        raise CounterFailure((key,), 'held')
                    status = await cache.register(self.redis, version, balances[key], namespace=self.namespace, fresh=bool(inserted))
                    if status != 'ok':
                        raise CounterFailure((key,), status)
                await tx.execute_raw(
                    'INSERT INTO "LiteLLM_ModerationMeteringTask" (intent_id,binding,reservations) VALUES ($1,$2::jsonb,$3::jsonb)',
                    binding.intent_id, binding.model_dump_json(), _json({key: str(value) for key, value in reservations.items()})
                )
                for phase in binding.expected_phases:
                    placeholder = PhaseEvent(binding=binding, request_id=request_id(binding, phase), phase=phase,
                                             provider='', deployment_id='', native_id='', provider_task_id='')
                    await tx.execute_raw(
                        'INSERT INTO "LiteLLM_ModerationMeteringPhase" (request_id,intent_id,phase,payload,payload_hash,status) '
                        'VALUES ($1,$2,$3,$4::jsonb,$5,\'unknown\')', placeholder.request_id, binding.intent_id,
                        phase, placeholder.model_dump_json(), placeholder.digest()
                    )
        except CounterFailure as error:
            await self.quarantine(error.counter_keys)
            raise

    async def persist(self, event: PhaseEvent) -> None:
        if event.request_id != request_id(event.binding, event.phase):
            raise ValueError('moderation billing request identity mismatch')
        if event.phase not in event.binding.expected_phases or not event.native_id or not event.deployment_id:
            raise ValueError('complete provider billing identity required')
        if event.finalized and event.amount is None:
            raise ValueError('finalized phase requires authoritative amount')
        async with self.transactions() as tx:
            tasks = TypeAdapter(list[TaskRow]).validate_python(await tx.query_raw(
                'SELECT binding,reservations FROM "LiteLLM_ModerationMeteringTask" WHERE intent_id=$1 FOR UPDATE',
                event.binding.intent_id
            ))
            if not tasks or tasks[0].binding != event.binding:
                raise ValueError('unattested metering event')
            siblings = TypeAdapter(list[PhaseRow]).validate_python(await tx.query_raw(
                'SELECT * FROM "LiteLLM_ModerationMeteringPhase" WHERE intent_id=$1', event.binding.intent_id
            ))
            if any((sibling.payload.provider, sibling.payload.deployment_id, sibling.payload.native_id,
                    sibling.payload.provider_task_id) != (event.provider, event.deployment_id, event.native_id,
                                                         event.provider_task_id) for sibling in siblings if sibling.payload.native_id):
                raise ValueError('moderation billing provider binding changed')
            old = TypeAdapter(list[PhaseRow]).validate_python(await tx.query_raw(
                'SELECT * FROM "LiteLLM_ModerationMeteringPhase" WHERE intent_id=$1 AND phase=$2 FOR UPDATE',
                event.binding.intent_id, event.phase
            ))
            if old:
                previous = old[0]
                if previous.payload == event:
                    return
                updates = {'amount': event.amount, 'finalized': event.finalized}
                if not previous.payload.native_id:
                    updates = {**updates, 'provider': event.provider, 'deployment_id': event.deployment_id,
                               'native_id': event.native_id, 'provider_task_id': event.provider_task_id}
                if previous.status != 'unknown' or previous.payload.model_copy(update=updates) != event:
                    raise ValueError('moderation billing payload replay conflict')
                await tx.execute_raw(
                    'UPDATE "LiteLLM_ModerationMeteringPhase" SET payload=$2::jsonb,payload_hash=$3,status=$4 WHERE request_id=$1',
                    event.request_id, event.model_dump_json(), event.digest(), 'pending' if event.finalized else 'unknown'
                )
                return
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_ModerationMeteringPhase" (request_id,intent_id,phase,payload,payload_hash,status) '
                'VALUES ($1,$2,$3,$4::jsonb,$5,$6)', event.request_id, event.binding.intent_id, event.phase,
                event.model_dump_json(), event.digest(), 'pending' if event.finalized else 'unknown'
            )

    async def quarantine(self, counter_keys: tuple[str, ...]) -> None:
        for key in counter_keys:
            await self.db.execute_raw(
                'UPDATE "LiteLLM_ModerationMeteringCounter" SET status=\'reconciliation_required\',updated_at=NOW() WHERE counter_key=$1', key
            )

    async def assert_admission(self, counter_keys: tuple[str, ...]) -> None:
        for key in sorted(counter_keys):
            rows = TypeAdapter(list[CounterRow]).validate_python(await self.db.query_raw(
                'SELECT * FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=$1', key
            ))
            if not rows:
                continue
            if rows[0].status != 'ready':
                raise CounterFailure((key,), 'held')
            status = await cache.check(self.redis, rows[0], namespace=self.namespace)
            if status == 'ahead':
                raise ValueError('moderation metering adjustment commit pending')
            if status != 'ok':
                await self.quarantine((key,))
                raise CounterFailure((key,), status)

    async def run_once(self) -> bool:
        token = uuid4().hex
        claimed = TypeAdapter(list[PhaseRow]).validate_python(await self.db.query_raw(
            'UPDATE "LiteLLM_ModerationMeteringPhase" SET status=\'running\',lease_token=$1,lease_until=NOW()+INTERVAL \'20 seconds\',attempts=attempts+1 '
            'WHERE request_id=(SELECT request_id FROM "LiteLLM_ModerationMeteringPhase" '
            'WHERE (status=\'pending\' OR (status=\'running\' AND lease_until<NOW())) AND available_at<=NOW() '
            'ORDER BY created_at,request_id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *', token
        ))
        if not claimed:
            return False
        row = claimed[0]
        try:
            await self._apply(row.request_id, token)
        except CounterFailure as error:
            await self.quarantine(error.counter_keys)
            await self.db.execute_raw(
                'UPDATE "LiteLLM_ModerationMeteringPhase" SET status=\'reconciliation_required\' WHERE request_id=$1 AND lease_token=$2',
                row.request_id, token
            )
            raise
        except Exception:
            await self.db.execute_raw(
                'UPDATE "LiteLLM_ModerationMeteringPhase" SET status=\'pending\',available_at=NOW()+INTERVAL \'2 seconds\' '
                'WHERE request_id=$1 AND lease_token=$2 AND status=\'running\'', row.request_id, token
            )
            raise
        return True

    async def _apply(self, request_id: str, token: str) -> None:
        async with self.transactions() as tx:
            rows = TypeAdapter(list[PhaseRow]).validate_python(await tx.query_raw(
                'SELECT * FROM "LiteLLM_ModerationMeteringPhase" WHERE request_id=$1 FOR UPDATE', request_id
            ))
            row = rows[0]
            if row.status == 'settled':
                return
            if row.status != 'running' or row.lease_token != token or row.lease_until is None:
                raise ValueError('moderation billing lease lost')
            event = row.payload
            if event.amount is None or not event.finalized:
                raise ValueError('phase has no authoritative amount')
            balances = await self._identity(tx, event.binding)
            tasks = TypeAdapter(list[TaskRow]).validate_python(await tx.query_raw(
                'SELECT binding,reservations FROM "LiteLLM_ModerationMeteringTask" WHERE intent_id=$1', event.binding.intent_id
            ))
            versions = TypeAdapter(list[CounterRow]).validate_python(await tx.query_raw(
                'SELECT * FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=ANY($1::text[]) ORDER BY counter_key FOR UPDATE',
                list(sorted(balances))
            ))
            if len(versions) != len(balances) or any(version.status != 'ready' for version in versions):
                raise CounterFailure(tuple(balances), 'held')
            counters = tuple(cache.CounterVersion(
                counter_key=version.counter_key, generation=version.generation, committed_seq=version.committed_seq,
                reserved=tasks[0].reservations.get(version.counter_key, Decimal(0)) if event.phase == event.binding.expected_phases[0] else Decimal(0)
            ) for version in versions)
            status = await cache.adjust(
                self.redis, counters, event_id=event.request_id, payload_hash=row.payload_hash,
                amount=event.amount, lease_until_ms=int(row.lease_until.timestamp() * 1000), namespace=self.namespace
            )
            if status in ('ahead', 'lease'):
                raise ValueError('moderation metering predecessor or lease pending')
            if status != 'ok':
                raise CounterFailure(tuple(balances), status)
            await self._debit(tx, event)
            await tx.execute_raw(
                'UPDATE "LiteLLM_ModerationMeteringCounter" SET committed_seq=committed_seq+1,updated_at=NOW() WHERE counter_key=ANY($1::text[])',
                list(sorted(balances))
            )
            changed = await tx.execute_raw(
                'UPDATE "LiteLLM_ModerationMeteringPhase" SET status=\'settled\',receipt=payload '
                'WHERE request_id=$1 AND lease_token=$2 AND lease_until>NOW() AND status=\'running\'', request_id, token
            )
            if changed != 1:
                raise ValueError('moderation billing lease expired')

    async def _debit(self, tx: Database, event: PhaseEvent) -> None:
        binding = event.binding
        targets = (
            ('"LiteLLM_VerificationToken"', 'token', binding.fingerprint),
            ('"LiteLLM_TeamTable"', 'team_id', binding.team_id),
            ('"LiteLLM_UserTable"', 'user_id', binding.user_id),
            *((('"LiteLLM_OrganizationTable"', 'organization_id', binding.organization_id),) if binding.organization_id else ()),
        )
        for table, column, identity in targets:
            count = await tx.execute_raw(
                f'UPDATE {table} SET spend=COALESCE(spend,0)+$2::double precision WHERE {column}=$1', identity, str(event.amount)
            )
            if count != 1:
                raise ValueError('moderation metering debit row disappeared')
        if await tx.execute_raw(
            'UPDATE "LiteLLM_TeamMembership" SET spend=spend+$3::double precision,total_spend=total_spend+$3::double precision '
            'WHERE user_id=$1 AND team_id=$2', binding.user_id, binding.team_id, str(event.amount)
        ) != 1:
            raise ValueError('moderation metering debit membership disappeared')

    async def settlement(self, binding: BillingBinding) -> SettlementEnvelope:
        tasks = TypeAdapter(list[TaskRow]).validate_python(await self.db.query_raw(
            'SELECT binding,reservations FROM "LiteLLM_ModerationMeteringTask" WHERE intent_id=$1', binding.intent_id
        ))
        if not tasks or tasks[0].binding != binding:
            raise ValueError('moderation settlement binding mismatch')
        rows = TypeAdapter(list[PhaseRow]).validate_python(await self.db.query_raw(
            'SELECT * FROM "LiteLLM_ModerationMeteringPhase" WHERE intent_id=$1 ORDER BY phase', binding.intent_id
        ))
        receipts = tuple(row.receipt for row in rows if row.status == 'settled' and row.receipt is not None)
        complete = set(receipt.phase for receipt in receipts) == set(binding.expected_phases)
        return SettlementEnvelope(
            binding=binding, receipts=receipts, complete=complete,
            total_actual=sum((receipt.amount or Decimal(0) for receipt in receipts), Decimal(0)) if complete else None
        )

    async def binding(self, intent_id: str) -> BillingBinding:
        tasks = TypeAdapter(list[TaskRow]).validate_python(await self.db.query_raw(
            'SELECT binding,reservations FROM "LiteLLM_ModerationMeteringTask" WHERE intent_id=$1', intent_id
        ))
        if not tasks:
            raise ValueError('moderation metering admission missing')
        return tasks[0].binding
