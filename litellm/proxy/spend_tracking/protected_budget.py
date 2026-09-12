from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

from litellm.proxy.video_endpoints.moderation_metering_cache import RedisCommands, keys
from litellm.proxy.video_endpoints.openapi_logs import Database


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BudgetIdentity(FrozenModel):
    kind: Literal["key", "user", "team", "org", "team_member", "end_user", "tag"]
    identity: str = Field(min_length=1)
    team_id: str | None = None
    window: str | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> BudgetIdentity:
        if (self.kind == "team_member") != (self.team_id is not None):
            raise ValueError("membership requires exact user and team")
        if self.window is not None and self.kind not in ("key", "team"):
            raise ValueError("window requires key or team")
        return self

    @property
    def counter_key(self) -> str:
        return (
            "spend:"
            + self.kind
            + ":"
            + self.identity
            + (":" + self.team_id if self.team_id else "")
            + (":window:" + self.window if self.window else "")
        )

    @property
    def table(self) -> str:
        return {
            "key": "LiteLLM_VerificationToken",
            "user": "LiteLLM_UserTable",
            "team": "LiteLLM_TeamTable",
            "org": "LiteLLM_OrganizationTable",
            "team_member": "LiteLLM_TeamMembership",
            "end_user": "LiteLLM_EndUserTable",
            "tag": "LiteLLM_TagTable",
        }[self.kind]

    @property
    def column(self) -> str:
        return {
            "key": "token",
            "user": "user_id",
            "team": "team_id",
            "org": "organization_id",
            "team_member": "user_id",
            "end_user": "user_id",
            "tag": "tag_name",
        }[self.kind]


class WriterEvidence(FrozenModel):
    writer_id: str = Field(min_length=1)
    version_sha: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    fence: str = Field(min_length=1)
    state: Literal["terminated", "drained_and_fenced"]
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CounterBaseline(FrozenModel):
    target: BudgetIdentity
    redis_value: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    sql_value: Decimal = Field(ge=0, allow_inf_nan=False)
    period_start: datetime


class CutoverReceipt(FrozenModel):
    receipt_id: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    boundary: datetime
    writers: tuple[WriterEvidence, ...] = Field(min_length=1)
    admission_closed: Literal[True]
    reset_writers_fenced: Literal[True]
    inventory_complete: Literal[True]
    unresolved_unversioned_reservations: Literal[0]
    unflushed_unversioned_operations: Literal[0]
    inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    baselines: tuple[CounterBaseline, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> CutoverReceipt:
        if self.boundary.tzinfo is None or any(b.period_start.tzinfo is None for b in self.baselines):
            raise ValueError("cutover timestamps require explicit timezone")
        if any(b.period_start > self.boundary for b in self.baselines):
            raise ValueError("counter period begins after cutover boundary")
        if len({w.writer_id for w in self.writers}) != len(self.writers):
            raise ValueError("duplicate writer evidence")
        if len({b.target.counter_key for b in self.baselines}) != len(self.baselines):
            raise ValueError("duplicate counter baseline")
        if any(b.redis_value is not None and b.redis_value < b.sql_value for b in self.baselines):
            raise ValueError("baseline requires reconciliation")
        return self

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class CounterState(FrozenModel):
    counter_key: str
    generation: str
    committed_seq: int
    status: str
    target: BudgetIdentity
    period_start: datetime
    pending_operation: str | None = None


class ReservationState(FrozenModel):
    reservation_id: str
    counter_key: str
    generation: str
    remaining: Decimal
    valid_until: datetime
    status: str


class CounterChange(FrozenModel):
    before: CounterState
    generation: str
    delta: Decimal
    reset_value: Decimal | None = None
    debit: Decimal = Decimal(0)
    reset_at: datetime | None = None
    period_start: datetime | None = None
    reservation: ReservationState | None = None


class OperationPayload(FrozenModel):
    kind: Literal["reserve", "resize", "release", "debit", "reset"]
    changes: tuple[CounterChange, ...]
    amount: Decimal
    reservation_id: str | None = None
    phase_request_id: str | None = None
    phase_hash: str | None = None

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class OperationRow(FrozenModel):
    operation_id: str
    payload: OperationPayload
    payload_hash: str
    status: str
    fence: int
    lease_until: datetime | None = None


class ReceiptRow(FrozenModel):
    payload: CutoverReceipt
    payload_hash: str
    status: str


class BirthRow(FrozenModel):
    counter_key: str
    birth_id: str
    target: BudgetIdentity
    initial_spend: Decimal
    created_at: datetime


class WindowRow(BaseModel):
    budget_limits: JsonValue


class WindowBudget(BaseModel):
    budget_duration: str
    reset_at: datetime


class Balance(FrozenModel):
    spend: Decimal


class BudgetPending(ValueError):
    pass


class BudgetReconciliation(BudgetPending):
    pass


class BudgetBusy(BudgetPending):
    pass


REGISTER_COUNTER = """
local guard = KEYS[2]
if redis.call('EXISTS', guard) == 1 then
  if redis.call('HGET', guard, 'generation') ~= ARGV[1] or redis.call('HGET', guard, 'cutover') ~= ARGV[2] then return 'conflict' end
  if redis.call('EXISTS', KEYS[1]) == 0 then return 'missing' end
  return 'ok'
end
local value = redis.call('GET', KEYS[1])
if ARGV[3] == 'absent' then
  if value then return 'baseline' end
elseif not value or tonumber(value) ~= tonumber(ARGV[3]) then return 'baseline' end
redis.call('SET', KEYS[1], ARGV[4])
redis.call('HSET', guard, 'generation', ARGV[1], 'seq', '0', 'cutover', ARGV[2], 'fence', '0')
redis.call('PERSIST', guard)
redis.call('SET', KEYS[3], '"protected-v1"')
return 'ok'
"""

APPLY_OPERATION = """
local now = redis.call('TIME')
if tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000) >= tonumber(ARGV[4]) then return 'lease' end
for i = 1, #KEYS, 2 do
  local a = 5 + math.floor((i-1)/2) * 5
  if redis.call('EXISTS', KEYS[i]) == 0 then return 'missing' end
  local gen = redis.call('HGET', KEYS[i+1], 'generation')
  local seq = tonumber(redis.call('HGET', KEYS[i+1], 'seq'))
  if seq == tonumber(ARGV[a+1])+1 then
    if gen ~= ARGV[a+2] or redis.call('HGET', KEYS[i+1], 'event') ~= ARGV[1] or redis.call('HGET', KEYS[i+1], 'payload') ~= ARGV[2] then return 'unproven_ahead' end
  elseif gen ~= ARGV[a] or seq ~= tonumber(ARGV[a+1]) then return 'evidence' end
  if tonumber(redis.call('HGET', KEYS[i+1], 'fence') or '0') > tonumber(ARGV[3]) and redis.call('HGET', KEYS[i+1], 'event') == ARGV[1] then return 'fence' end
end
for i = 1, #KEYS, 2 do
  local a = 5 + math.floor((i-1)/2) * 5
  if tonumber(redis.call('HGET', KEYS[i+1], 'seq')) == tonumber(ARGV[a+1]) then
    if ARGV[a+4] ~= 'increment' then redis.call('SET', KEYS[i], ARGV[a+4])
    else redis.call('INCRBYFLOAT', KEYS[i], ARGV[a+3]) end
    redis.call('HSET', KEYS[i+1], 'generation', ARGV[a+2], 'seq', tonumber(ARGV[a+1])+1, 'event', ARGV[1], 'payload', ARGV[2])
  end
  redis.call('HSET', KEYS[i+1], 'fence', ARGV[3])
  redis.call('PERSIST', KEYS[i])
  redis.call('PERSIST', KEYS[i+1])
end
return 'ok'
"""

CHECK_COUNTER = """
if redis.call('EXISTS', KEYS[1]) == 0 then return 'missing' end
if redis.call('HGET', KEYS[2], 'generation') ~= ARGV[1] then return 'generation' end
if tonumber(redis.call('HGET', KEYS[2], 'seq')) ~= tonumber(ARGV[2]) then return 'sequence' end
redis.call('PERSIST', KEYS[1])
redis.call('PERSIST', KEYS[2])
return 'ok'
"""


class ProtectedBudgetStore:
    def __init__(
        self,
        db: Database,
        transactions: Callable[[], AbstractAsyncContextManager[Database]],
        redis: RedisCommands,
        *,
        namespace: str = "",
    ):
        self.db = db
        self.transactions = transactions
        self.redis = redis
        self.namespace = namespace

    async def states(
        self, tx: Database, counter_keys: tuple[str, ...], *, lock: bool = False
    ) -> tuple[CounterState, ...]:
        return TypeAdapter(tuple[CounterState, ...]).validate_python(
            await tx.query_raw(
                "SELECT counter_key,generation,committed_seq,status,target,period_start,pending_operation "
                'FROM "LiteLLM_ModerationMeteringCounter" WHERE counter_key=ANY($1::text[]) ORDER BY counter_key'
                + (" FOR UPDATE" if lock else ""),
                list(counter_keys),
            )
        )

    async def quarantine(self, counter_keys: tuple[str, ...], reason: str) -> None:
        await self.db.execute_raw(
            "UPDATE \"LiteLLM_ModerationMeteringCounter\" SET status='reconciliation_required',failure_reason=$2 "
            "WHERE counter_key=ANY($1::text[])",
            list(counter_keys),
            reason,
        )

    async def _balance(self, tx: Database, target: BudgetIdentity) -> Decimal:
        if target.window is not None:
            rows = TypeAdapter(tuple[WindowRow, ...]).validate_python(
                await tx.query_raw(
                    f'SELECT budget_limits FROM "{target.table}" WHERE {target.column}=$1 FOR UPDATE',
                    target.identity,
                )
            )
            if len(rows) != 1:
                raise BudgetPending("window identity missing")
            raw = rows[0].budget_limits
            windows = (
                TypeAdapter(tuple[WindowBudget, ...]).validate_json(raw)
                if isinstance(raw, str)
                else TypeAdapter(tuple[WindowBudget, ...]).validate_python(raw)
            )
            if len(tuple(w for w in windows if w.budget_duration == target.window)) != 1:
                raise BudgetPending("window budget identity missing")
            return Decimal(0)
        suffix = " AND team_id=$2" if target.team_id is not None else ""
        args = (target.identity, target.team_id) if target.team_id is not None else (target.identity,)
        rows = TypeAdapter(tuple[Balance, ...]).validate_python(
            await tx.query_raw(
                f'SELECT COALESCE(spend,0) AS spend FROM "{target.table}" WHERE {target.column}=$1{suffix} FOR UPDATE',
                *args,
            )
        )
        if len(rows) != 1:
            raise BudgetPending("budget identity missing")
        return rows[0].spend

    async def _install_birth_guards(self, tx: Database) -> None:
        await tx.execute_raw("""
CREATE OR REPLACE FUNCTION protected_budget_birth() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE item jsonb; target jsonb; counter text; budget jsonb; windows jsonb; old_windows jsonb;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM "LiteLLM_BudgetCutover" WHERE status='ready') THEN RETURN NEW; END IF;
  item := to_jsonb(NEW);
  target := jsonb_strip_nulls(jsonb_build_object('kind',TG_ARGV[0],'identity',item->>TG_ARGV[1],
               'team_id', CASE WHEN TG_ARGV[0]='team_member' THEN item->>'team_id' ELSE NULL END));
  counter := 'spend:' || TG_ARGV[0] || ':' || (item->>TG_ARGV[1]) ||
             CASE WHEN TG_ARGV[0]='team_member' THEN ':' || (item->>'team_id') ELSE '' END;
  IF TG_OP='INSERT' THEN
  INSERT INTO "LiteLLM_BudgetBirth" (counter_key,birth_id,target,initial_spend)
    VALUES(counter,txid_current()::text || ':' || counter,target,COALESCE((item->>'spend')::numeric,0))
    ON CONFLICT(counter_key) DO NOTHING;
  END IF;
  IF TG_ARGV[0] IN ('key','team') AND item->>'budget_limits' IS NOT NULL THEN
    windows := CASE WHEN jsonb_typeof(item->'budget_limits')='string'
                    THEN (item->>'budget_limits')::jsonb ELSE item->'budget_limits' END;
    old_windows := CASE WHEN TG_OP='INSERT' THEN '[]'::jsonb
                        WHEN jsonb_typeof(to_jsonb(OLD)->'budget_limits')='string'
                        THEN (to_jsonb(OLD)->>'budget_limits')::jsonb
                        ELSE COALESCE(NULLIF(to_jsonb(OLD)->'budget_limits','null'::jsonb),'[]'::jsonb) END;
    FOR budget IN SELECT value FROM jsonb_array_elements(windows) LOOP
      IF EXISTS (SELECT 1 FROM jsonb_array_elements(old_windows) old_budget
                 WHERE old_budget->>'budget_duration'=budget->>'budget_duration') THEN CONTINUE; END IF;
      INSERT INTO "LiteLLM_BudgetBirth" (counter_key,birth_id,target,initial_spend)
        VALUES(counter || ':window:' || (budget->>'budget_duration'),
               txid_current()::text || ':' || counter || ':window:' || (budget->>'budget_duration'),
               target || jsonb_build_object('window',budget->>'budget_duration'),0)
        ON CONFLICT(counter_key) DO NOTHING;
    END LOOP;
  END IF;
  RETURN NEW;
END $$
""")
        for kind in ("key", "user", "team", "org", "team_member", "end_user", "tag"):
            target = BudgetIdentity(kind=kind, identity="trigger", team_id="trigger" if kind == "team_member" else None)
            await tx.execute_raw(f'DROP TRIGGER IF EXISTS protected_budget_birth ON "{target.table}"')
            await tx.execute_raw(
                f'CREATE TRIGGER protected_budget_birth AFTER INSERT OR UPDATE ON "{target.table}" '
                f"FOR EACH ROW EXECUTE FUNCTION protected_budget_birth('{kind}','{target.column}')"
            )

    async def register_birth(self, counter_key: str) -> bool:
        async with self.transactions() as tx:
            await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtext($1))", "budget-birth:" + counter_key)
            births = TypeAdapter(tuple[BirthRow, ...]).validate_python(
                await tx.query_raw(
                    'SELECT * FROM "LiteLLM_BudgetBirth" WHERE counter_key=$1 FOR UPDATE',
                    counter_key,
                )
            )
            if not births:
                return False
            birth = births[0]
            existing = await self.states(tx, (counter_key,), lock=True)
            if not existing:
                if await self._balance(tx, birth.target) != birth.initial_spend:
                    raise BudgetPending("birth has unversioned spend; reconciliation required")
                await tx.execute_raw(
                    'INSERT INTO "LiteLLM_ModerationMeteringCounter" '
                    "(counter_key,generation,status,target,period_start) VALUES ($1,$2,'registering',$3::jsonb,$4::timestamptz)",
                    counter_key,
                    uuid4().hex,
                    birth.target.model_dump_json(),
                    birth.created_at,
                )
            elif existing[0].status == "ready":
                return True
            elif existing[0].status != "registering":
                raise BudgetPending("birth counter requires reconciliation")
        state = (await self.states(self.db, (counter_key,)))[0]
        status = await self.redis.eval(
            REGISTER_COUNTER,
            3,
            *keys(counter_key, self.namespace),
            self.namespace + "protected:mode",
            state.generation,
            "birth:" + birth.birth_id,
            "absent",
            str(birth.initial_spend),
        )
        if status != "ok":
            raise BudgetPending("birth counter evidence conflict: " + str(status))
        await self.db.execute_raw(
            "UPDATE \"LiteLLM_ModerationMeteringCounter\" SET status='ready' WHERE counter_key=$1 AND status='registering'",
            counter_key,
        )
        return True

    async def register_cutover(self, receipt: CutoverReceipt) -> None:
        async with self.transactions() as tx:
            await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtext($1))", "protected-budget-cutover")
            previous = TypeAdapter(tuple[ReceiptRow, ...]).validate_python(
                await tx.query_raw(
                    'SELECT payload,payload_hash,status FROM "LiteLLM_BudgetCutover" WHERE receipt_id=$1',
                    receipt.receipt_id,
                )
            )
            if previous:
                if previous[0].payload != receipt:
                    raise BudgetPending("cutover receipt conflict")
                if previous[0].status == "ready":
                    return
            else:
                await self._install_birth_guards(tx)
                await tx.execute_raw(
                    'INSERT INTO "LiteLLM_BudgetCutover" (receipt_id,payload,payload_hash) VALUES ($1,$2::jsonb,$3)',
                    receipt.receipt_id,
                    receipt.model_dump_json(),
                    receipt.digest(),
                )
                for baseline in sorted(receipt.baselines, key=lambda value: value.target.counter_key):
                    if await self._balance(tx, baseline.target) != baseline.sql_value:
                        raise BudgetPending("cutover SQL baseline changed")
                    if await self.states(tx, (baseline.target.counter_key,), lock=True):
                        raise BudgetPending("counter already registered; explicit reconciliation required")
                    await tx.execute_raw(
                        'INSERT INTO "LiteLLM_ModerationMeteringCounter" '
                        "(counter_key,generation,status,target,period_start,cutover_id) VALUES ($1,$2,'registering',$3::jsonb,$4::timestamptz,$5)",
                        baseline.target.counter_key,
                        uuid4().hex,
                        baseline.target.model_dump_json(),
                        baseline.period_start,
                        receipt.receipt_id,
                    )
        states = await self.states(self.db, tuple(b.target.counter_key for b in receipt.baselines))
        by_key = {b.target.counter_key: b for b in receipt.baselines}
        for state in states:
            baseline = by_key[state.counter_key]
            result = await self.redis.eval(
                REGISTER_COUNTER,
                3,
                *keys(state.counter_key, self.namespace),
                self.namespace + "protected:mode",
                state.generation,
                receipt.digest(),
                str(baseline.redis_value) if baseline.redis_value is not None else "absent",
                str(baseline.redis_value if baseline.redis_value is not None else baseline.sql_value),
            )
            if result != "ok":
                raise BudgetPending("cutover Redis baseline/evidence mismatch: " + str(result))
        async with self.transactions() as tx:
            await tx.execute_raw(
                "UPDATE \"LiteLLM_ModerationMeteringCounter\" SET status='ready' WHERE cutover_id=$1 AND status='registering'",
                receipt.receipt_id,
            )
            await tx.execute_raw(
                "UPDATE \"LiteLLM_BudgetCutover\" SET status='ready' WHERE receipt_id=$1", receipt.receipt_id
            )

    async def assert_admission(self, counter_keys: tuple[str, ...]) -> tuple[CounterState, ...]:
        for counter_key in sorted(counter_keys):
            await self.register_birth(counter_key)
        states = await self.states(self.db, counter_keys)
        for state in states:
            if state.status != "ready":
                raise BudgetPending("budget " + state.status)
            if state.pending_operation:
                await self.recover(state.pending_operation)
        refreshed = await self.states(self.db, counter_keys)
        for state in refreshed:
            result = await self.redis.eval(
                CHECK_COUNTER,
                2,
                *keys(state.counter_key, self.namespace),
                state.generation,
                str(state.committed_seq),
            )
            if result != "ok":
                current = (await self.states(self.db, (state.counter_key,)))[0]
                if current != state or current.pending_operation:
                    raise BudgetBusy("counter operation changed during admission")
                changed = await self.db.execute_raw(
                    "UPDATE \"LiteLLM_ModerationMeteringCounter\" SET status='reconciliation_required',failure_reason=$4 "
                    "WHERE counter_key=$1 AND generation=$2 AND committed_seq=$3 AND pending_operation IS NULL AND status='ready'",
                    state.counter_key,
                    state.generation,
                    state.committed_seq,
                    str(result),
                )
                if changed != 1:
                    raise BudgetBusy("counter operation changed before reconciliation")
                raise BudgetReconciliation("counter evidence lost: " + str(result))
        return refreshed

    async def operation(self, operation_id: str) -> OperationRow | None:
        rows = TypeAdapter(tuple[OperationRow, ...]).validate_python(
            await self.db.query_raw(
                'SELECT operation_id,payload,payload_hash,status,fence,lease_until FROM "LiteLLM_BudgetOperation" WHERE operation_id=$1',
                operation_id,
            )
        )
        return rows[0] if rows else None

    async def _persist_operation(self, tx: Database, operation_id: str, payload: OperationPayload) -> None:
        await tx.execute_raw(
            'INSERT INTO "LiteLLM_BudgetOperation" (operation_id,payload,payload_hash) VALUES ($1,$2::jsonb,$3)',
            operation_id,
            payload.model_dump_json(),
            payload.digest(),
        )
        for change in payload.changes:
            if (
                await tx.execute_raw(
                    'UPDATE "LiteLLM_ModerationMeteringCounter" SET pending_operation=$2 '
                    "WHERE counter_key=$1 AND pending_operation IS NULL AND status='ready'",
                    change.before.counter_key,
                    operation_id,
                )
                != 1
            ):
                raise BudgetBusy("predecessor must finish first")

    async def mutate(
        self,
        operation_id: str,
        counter_keys: tuple[str, ...],
        *,
        kind: Literal["reserve", "resize", "release", "debit"],
        amount: Decimal,
        reservation_id: str | None = None,
        valid_until: datetime | None = None,
        phase_request_id: str | None = None,
        phase_hash: str | None = None,
    ) -> None:
        async with asyncio.timeout(10):
            while True:
                try:
                    await self._mutate(
                        operation_id,
                        counter_keys,
                        kind=kind,
                        amount=amount,
                        reservation_id=reservation_id,
                        valid_until=valid_until,
                        phase_request_id=phase_request_id,
                        phase_hash=phase_hash,
                    )
                    return
                except BudgetBusy:
                    await asyncio.sleep(0.02)

    async def _mutate(
        self,
        operation_id: str,
        counter_keys: tuple[str, ...],
        *,
        kind: Literal["reserve", "resize", "release", "debit"],
        amount: Decimal,
        reservation_id: str | None = None,
        valid_until: datetime | None = None,
        phase_request_id: str | None = None,
        phase_hash: str | None = None,
    ) -> None:
        if not operation_id or not amount.is_finite() or amount < 0:
            raise ValueError("stable operation identity and finite nonnegative amount required")
        previous = await self.operation(operation_id)
        if previous is not None:
            if (
                previous.payload.kind,
                previous.payload.amount,
                previous.payload.reservation_id,
                tuple(c.before.counter_key for c in previous.payload.changes),
            ) != (kind, amount, reservation_id, tuple(sorted(counter_keys))) or (
                previous.payload.phase_request_id,
                previous.payload.phase_hash,
            ) != (phase_request_id, phase_hash):
                raise BudgetPending("operation replay conflict")
            if kind == "reserve" and any(
                c.reservation is None or c.reservation.valid_until != valid_until for c in previous.payload.changes
            ):
                raise BudgetPending("reservation deadline replay conflict")
            await self.recover(operation_id)
            return
        await self.assert_admission(counter_keys)
        async with self.transactions() as tx:
            states = await self.states(tx, counter_keys, lock=True)
            if len(states) != len(set(counter_keys)) or any(s.pending_operation or s.status != "ready" for s in states):
                raise BudgetBusy("all operation counters must be ready")
            changes = tuple(
                [await self._change(tx, state, kind, amount, reservation_id, valid_until) for state in states]
            )
            await self._persist_operation(
                tx,
                operation_id,
                OperationPayload(
                    kind=kind,
                    changes=changes,
                    amount=amount,
                    reservation_id=reservation_id,
                    phase_request_id=phase_request_id,
                    phase_hash=phase_hash,
                ),
            )
        await self.recover(operation_id)

    async def _change(
        self,
        tx: Database,
        state: CounterState,
        kind: str,
        amount: Decimal,
        reservation_id: str | None,
        valid_until: datetime | None,
    ) -> CounterChange:
        await self._balance(tx, state.target)
        previous = TypeAdapter(tuple[ReservationState, ...]).validate_python(
            await tx.query_raw(
                'SELECT reservation_id,counter_key,generation,remaining,valid_until,status FROM "LiteLLM_BudgetReservation" '
                "WHERE reservation_id=$1 AND counter_key=$2 FOR UPDATE",
                reservation_id or "",
                state.counter_key,
            )
        )
        if kind == "reserve":
            if not reservation_id or valid_until is None or valid_until <= datetime.now(timezone.utc) or previous:
                raise BudgetPending("new valid reservation identity required")
            reservation = ReservationState(
                reservation_id=reservation_id,
                counter_key=state.counter_key,
                generation=state.generation,
                remaining=amount,
                valid_until=valid_until,
                status="active",
            )
            return CounterChange(before=state, generation=state.generation, delta=amount, reservation=reservation)
        if kind in ("resize", "release") and not previous:
            raise BudgetPending("reservation identity missing")
        if (
            kind == "resize"
            and previous
            and (previous[0].status != "active" or previous[0].valid_until <= datetime.now(timezone.utc))
        ):
            raise BudgetPending("reservation resize requires active unexpired identity")
        if previous and previous[0].status == "settled":
            if kind == "release":
                return CounterChange(before=state, generation=state.generation, delta=Decimal(0))
            raise BudgetPending("reservation already settled")
        remaining = previous[0].remaining if previous and previous[0].generation == state.generation else Decimal(0)
        new_remaining = amount if kind == "resize" else Decimal(0)
        reservation = (
            previous[0].model_copy(
                update={
                    "remaining": new_remaining,
                    "generation": state.generation,
                    "status": "active" if kind == "resize" else "released" if kind == "release" else "settled",
                }
            )
            if previous
            else None
        )
        return CounterChange(
            before=state,
            generation=state.generation,
            delta=amount - remaining if kind != "release" else -remaining,
            debit=amount if kind == "debit" else Decimal(0),
            reservation=reservation,
        )

    async def reset(
        self, operation_id: str, counter_key: str, *, boundary: datetime, reset_at: datetime | None
    ) -> None:
        previous = await self.operation(operation_id)
        if previous:
            change = previous.payload.changes[0]
            if (
                previous.payload.kind != "reset"
                or change.before.counter_key != counter_key
                or change.period_start != boundary
                or change.reset_at != reset_at
            ):
                raise BudgetPending("reset replay conflict")
            await self.recover(operation_id)
            return
        await self.assert_admission((counter_key,))
        async with self.transactions() as tx:
            states = await self.states(tx, (counter_key,), lock=True)
            if len(states) != 1 or states[0].pending_operation:
                raise BudgetPending("reset predecessor missing or pending")
            state = states[0]
            if boundary <= state.period_start:
                raise BudgetPending("obsolete reset boundary")
            await self._balance(tx, state.target)
            reservations = TypeAdapter(tuple[ReservationState, ...]).validate_python(
                await tx.query_raw(
                    'SELECT reservation_id,counter_key,generation,remaining,valid_until,status FROM "LiteLLM_BudgetReservation" '
                    "WHERE counter_key=$1 AND status='active' FOR UPDATE",
                    counter_key,
                )
            )
            carry = sum(
                (r.remaining for r in reservations if r.generation == state.generation and r.valid_until > boundary),
                Decimal(0),
            )
            change = CounterChange(
                before=state,
                generation=uuid4().hex,
                delta=Decimal(0),
                reset_value=carry,
                reset_at=reset_at,
                period_start=boundary,
            )
            await self._persist_operation(
                tx, operation_id, OperationPayload(kind="reset", changes=(change,), amount=Decimal(0))
            )
        await self.recover(operation_id)

    async def recover(self, operation_id: str) -> None:
        previous = await self.operation(operation_id)
        if previous is None:
            raise BudgetPending("operation missing")
        if previous.status == "committed":
            return
        claimed = TypeAdapter(tuple[OperationRow, ...]).validate_python(
            await self.db.query_raw(
                "UPDATE \"LiteLLM_BudgetOperation\" SET fence=fence+1,status='running',lease_until=NOW()+INTERVAL '15 seconds' "
                "WHERE operation_id=$1 AND (status='pending' OR (status='running' AND lease_until<NOW())) "
                "RETURNING operation_id,payload,payload_hash,status,fence,lease_until",
                operation_id,
            )
        )
        if not claimed:
            raise BudgetBusy("operation claim pending")
        row = claimed[0]
        if row.lease_until is None:
            raise BudgetPending("operation lease absent")
        counter_keys = tuple(c.before.counter_key for c in row.payload.changes)
        try:
            arguments = tuple(
                value
                for c in row.payload.changes
                for value in (
                    c.before.generation,
                    str(c.before.committed_seq),
                    c.generation,
                    str(c.delta),
                    str(c.reset_value) if c.reset_value is not None else "increment",
                )
            )
            redis_keys = tuple(k for key in counter_keys for k in keys(key, self.namespace))
            status = await self.redis.eval(
                APPLY_OPERATION,
                len(redis_keys),
                *redis_keys,
                operation_id,
                row.payload_hash,
                str(row.fence),
                str(int(row.lease_until.timestamp() * 1000)),
                *arguments,
            )
            if status in ("lease", "fence"):
                raise BudgetPending("operation lease lost")
            if status != "ok":
                await self.quarantine(counter_keys, str(status))
                raise BudgetReconciliation("operation evidence mismatch: " + str(status))
            await self._commit(row)
        except Exception:
            await self.db.execute_raw(
                "UPDATE \"LiteLLM_BudgetOperation\" SET status='pending' WHERE operation_id=$1 AND fence=$2 AND status='running'",
                operation_id,
                row.fence,
            )
            raise

    async def _commit(self, row: OperationRow) -> None:
        async with self.transactions() as tx:
            states = await self.states(tx, tuple(c.before.counter_key for c in row.payload.changes), lock=True)
            if any(s.pending_operation != row.operation_id or s.status != "ready" for s in states):
                raise BudgetPending("operation ownership lost")
            if row.payload.phase_request_id is not None:
                await self._verify_phase(tx, row.payload)
            for change in row.payload.changes:
                await self._balance(tx, change.before.target)
                await self._write_balance(tx, change)
                if change.reservation is not None:
                    reservation = change.reservation
                    await tx.execute_raw(
                        'INSERT INTO "LiteLLM_BudgetReservation" (reservation_id,counter_key,generation,remaining,valid_until,status) '
                        "VALUES ($1,$2,$3,$4::numeric,$5::timestamptz,$6) ON CONFLICT(reservation_id,counter_key) DO UPDATE "
                        "SET generation=EXCLUDED.generation,remaining=EXCLUDED.remaining,status=EXCLUDED.status",
                        reservation.reservation_id,
                        reservation.counter_key,
                        reservation.generation,
                        str(reservation.remaining),
                        reservation.valid_until,
                        reservation.status,
                    )
                if change.period_start is not None:
                    await tx.execute_raw(
                        'UPDATE "LiteLLM_BudgetReservation" SET generation=$2,remaining=CASE WHEN valid_until>$3::timestamptz THEN remaining ELSE 0 END '
                        "WHERE counter_key=$1 AND status='active' AND generation=$4",
                        change.before.counter_key,
                        change.generation,
                        change.period_start,
                        change.before.generation,
                    )
                await tx.execute_raw(
                    'UPDATE "LiteLLM_ModerationMeteringCounter" SET generation=$2,committed_seq=committed_seq+1,pending_operation=NULL,'
                    "period_start=COALESCE(NULLIF($3::text,'')::timestamptz,period_start) WHERE counter_key=$1",
                    change.before.counter_key,
                    change.generation,
                    change.period_start.isoformat() if change.period_start else "",
                )
            if row.payload.phase_request_id is not None:
                await tx.execute_raw(
                    "UPDATE \"LiteLLM_ModerationMeteringPhase\" SET status='settled',receipt=payload WHERE request_id=$1",
                    row.payload.phase_request_id,
                )
            if (
                await tx.execute_raw(
                    "UPDATE \"LiteLLM_BudgetOperation\" SET status='committed' WHERE operation_id=$1 AND fence=$2 AND status='running' AND lease_until>NOW()",
                    row.operation_id,
                    row.fence,
                )
                != 1
            ):
                raise BudgetPending("operation claim expired before SQL receipt")

    async def _verify_phase(self, tx: Database, payload: OperationPayload) -> None:
        from litellm.proxy.video_endpoints.moderation_metering import MeteringStore, PhaseRow

        rows = TypeAdapter(tuple[PhaseRow, ...]).validate_python(
            await tx.query_raw(
                'SELECT * FROM "LiteLLM_ModerationMeteringPhase" WHERE request_id=$1 FOR UPDATE',
                payload.phase_request_id,
            )
        )
        if len(rows) != 1 or rows[0].payload_hash != payload.phase_hash:
            raise BudgetPending("phase identity/hash missing or changed")
        phase = rows[0].payload
        if not phase.finalized or phase.amount != payload.amount:
            raise BudgetPending("authoritative phase amount missing")
        authority = MeteringStore(self.db, self.transactions, self.redis, namespace=self.namespace)
        balances = await authority._identity(tx, phase.binding)
        if set(balances) - {change.before.counter_key for change in payload.changes}:
            raise BudgetPending("phase debit omitted identity")

    async def _write_balance(self, tx: Database, change: CounterChange) -> None:
        target = change.before.target
        if target.window is not None:
            if change.reset_at is not None:
                if (
                    await tx.execute_raw(
                        f"UPDATE \"{target.table}\" SET budget_limits=(SELECT jsonb_agg(CASE WHEN w->>'budget_duration'=$2 "
                        "THEN jsonb_set(w,'{reset_at}',to_jsonb($3::text)) ELSE w END) FROM jsonb_array_elements(budget_limits::jsonb) w) "
                        f"WHERE {target.column}=$1",
                        target.identity,
                        target.window,
                        change.reset_at.isoformat(),
                    )
                    != 1
                ):
                    raise BudgetPending("window reset identity missing")
            return
        if change.reset_value is None and change.debit == 0:
            return
        reset = change.reset_value is not None
        arguments = (target.identity,) if reset else (target.identity, str(change.debit))
        assignment = "spend=0" if reset else "spend=COALESCE(spend,0)+$2::double precision"
        if target.kind == "team_member" and not reset:
            assignment += ",total_spend=total_spend+$2::double precision"
        identity_arguments = (*arguments, target.team_id) if target.team_id else arguments
        suffix = f" AND team_id=${len(identity_arguments)}" if target.team_id else ""
        all_arguments = (
            (*identity_arguments, change.reset_at) if reset and change.reset_at is not None else identity_arguments
        )
        reset_assignment = (
            f",budget_reset_at=${len(all_arguments)}::timestamptz" if reset and change.reset_at is not None else ""
        )
        if (
            await tx.execute_raw(
                f'UPDATE "{target.table}" SET {assignment}{reset_assignment} WHERE {target.column}=$1{suffix}',
                *all_arguments,
            )
            != 1
        ):
            raise BudgetPending("budget debit/reset identity missing")
