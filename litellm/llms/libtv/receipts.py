import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping, TypedDict
from urllib.parse import urlsplit

from redis.exceptions import RedisError, ResponseError

ReceiptState = Literal["not_submitted", "rejected", "unknown", "submitted", "submitting"]
TaskState = Literal["active", "succeeded", "failed", "resolved"]
ClaimOutcome = Literal["owner", "existing", "rejected", "missing", "mismatch"]


class TerminalResult(TypedDict):
    url: str
    provider_task_id: str


def _terminal_result(value: object) -> TerminalResult | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"url", "provider_task_id"}:
        raise ValueError("terminal result has invalid shape")
    url = value["url"]
    provider_task_id = value["provider_task_id"]
    parsed = urlsplit(url) if isinstance(url, str) else None
    if (
        not isinstance(provider_task_id, str)
        or not provider_task_id
        or parsed is None
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
    ):
        raise ValueError("terminal result is invalid")
    return {"url": url, "provider_task_id": provider_task_id}

_CLAIM_SCRIPT = """
local function merge_pool_attempt(pool_json, candidate_json)
  local candidate = cjson.decode(candidate_json)
  local attempts = {}
  if pool_json then
    local existing = cjson.decode(pool_json)
    attempts = existing['attempts'] or {}
    if #attempts == 0 and existing['deployment_id'] and existing['receipt_key'] then
      table.insert(attempts, {
        deployment_id = existing['deployment_id'],
        receipt_key = existing['receipt_key'],
        submission_state = existing['submission_state'],
      })
    end
  end
  local next_attempt = candidate['attempts'][1]
  local replaced = false
  for index, attempt in ipairs(attempts) do
    if attempt['deployment_id'] == next_attempt['deployment_id'] then
      attempts[index] = next_attempt
      replaced = true
      break
    end
  end
  if not replaced then table.insert(attempts, next_attempt) end
  candidate['attempts'] = attempts
  return cjson.encode(candidate)
end
local indexed = redis.call('GET', KEYS[1])
if indexed then
  local stored = redis.call('GET', indexed)
  if not stored then
    local pool = redis.call('GET', KEYS[2])
    if pool then
      local pool_value = cjson.decode(pool)
      if pool_value['fingerprint'] ~= ARGV[1] then
        return {'mismatch', indexed, pool}
      end
    end
    return {'missing', indexed}
  end
  local value = cjson.decode(stored)
  if value['fingerprint'] ~= ARGV[1] then
    return {'mismatch', indexed, stored}
  end
  if value['submission_state'] == 'submitting' or value['submission_state'] == 'unknown' or value['submission_state'] == 'submitted' then
    return {'existing', indexed, stored}
  end
  if value['deployment_id'] == ARGV[2] then
    return {value['submission_state'], indexed, stored}
  end
end
local pool = redis.call('GET', KEYS[2])
if pool then
  local pool_value = cjson.decode(pool)
  if pool_value['fingerprint'] ~= ARGV[1] then
    return {'mismatch', KEYS[3], pool}
  end
  if pool_value['submission_state'] == 'submitting' or pool_value['submission_state'] == 'unknown' or pool_value['submission_state'] == 'submitted' then
    local pool_receipt = pool_value['receipt_key']
    local pool_stored = redis.call('GET', pool_receipt)
    if pool_stored then
      return {'existing', pool_receipt, pool_stored}
    end
    return {'missing', pool_receipt}
  end
end
local merged_pool = merge_pool_attempt(pool, ARGV[4])
redis.call('SET', KEYS[1], KEYS[3])
redis.call('SET', KEYS[2], merged_pool)
redis.call('SET', KEYS[3], ARGV[3])
return {'owner', KEYS[3], ARGV[3]}
"""

_TRANSITION_SCRIPT = """
local function merge_pool_attempt(pool_json, candidate_json)
  local candidate = cjson.decode(candidate_json)
  local attempts = {}
  if pool_json then
    local existing = cjson.decode(pool_json)
    attempts = existing['attempts'] or {}
    if #attempts == 0 and existing['deployment_id'] and existing['receipt_key'] then
      table.insert(attempts, {
        deployment_id = existing['deployment_id'],
        receipt_key = existing['receipt_key'],
        submission_state = existing['submission_state'],
      })
    end
  end
  local next_attempt = candidate['attempts'][1]
  local replaced = false
  for index, attempt in ipairs(attempts) do
    if attempt['deployment_id'] == next_attempt['deployment_id'] then
      attempts[index] = next_attempt
      replaced = true
      break
    end
  end
  if not replaced then table.insert(attempts, next_attempt) end
  candidate['attempts'] = attempts
  return cjson.encode(candidate)
end
local stored = redis.call('GET', KEYS[3])
if not stored then
  return {'missing'}
end
local current = cjson.decode(stored)
-- cjson represents explicit JSON null with a truthy sentinel.
if current['resolution_tombstone'] and current['resolution_tombstone'] ~= cjson.null then
  return {'resolved', stored}
end
if current['task_state'] and current['task_state'] ~= cjson.null and current['task_state'] ~= 'active' then
  return {'terminal', stored}
end
if current['submission_state'] ~= ARGV[1] or current['deployment_id'] ~= ARGV[2] then
  return {'conflict', stored}
end
if ARGV[5] ~= '' and (not current['billing_event_id'] or current['billing_event_id'] == '' or current['billing_event_id'] == cjson.null) then
  if not redis.call('GET', KEYS[5]) then
    redis.call('XADD', KEYS[4], '*', 'payload', ARGV[5])
    redis.call('SET', KEYS[5], 'delivered')
  end
end
redis.call('SET', KEYS[3], ARGV[3])
redis.call('SET', KEYS[2], merge_pool_attempt(redis.call('GET', KEYS[2]), ARGV[4]))
return {'ok', ARGV[3]}
"""


@dataclass(frozen=True, slots=True)
class StoredReceipt:
    team_id: str
    model: str
    request_id: str
    fingerprint: str
    submission_state: ReceiptState
    deployment_id: str | None = None
    provider_task_id: str | None = None
    resume_token: str | None = None
    provider_code: str | None = None
    message: str | None = None
    response_cost: float | None = None
    billing_event_id: str | None = None
    api_key: str | None = None
    user_id: str | None = None
    organization_id: str | None = None
    scale: int | None = None
    project_id: str | None = None
    artifact_id: str | None = None
    attribution_user_id: str | None = None
    terminal_result: TerminalResult | None = None
    resolution_tombstone: dict | None = None
    task_state: TaskState = "active"

    @classmethod
    def from_json(cls, raw: str) -> "StoredReceipt":
        value = json.loads(raw)
        state = value["submission_state"]
        if state not in {"not_submitted", "rejected", "unknown", "submitted", "submitting"}:
            raise ValueError("invalid receipt state")
        task_state = value.get("task_state") or "active"
        if task_state not in {"active", "succeeded", "failed", "resolved"}:
            raise ValueError("invalid task state")
        scale = value.get("scale")
        if scale is not None and (isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0):
            raise ValueError("invalid receipt scale")
        terminal_result = _terminal_result(value.get("terminal_result"))
        if task_state == "succeeded" and terminal_result is None:
            raise ValueError("succeeded receipt has no terminal result")
        return cls(
            team_id=value["team_id"],
            model=value["model"],
            request_id=value["request_id"],
            fingerprint=value["fingerprint"],
            submission_state=state,
            deployment_id=value.get("deployment_id"),
            provider_task_id=value.get("provider_task_id"),
            resume_token=value.get("resume_token"),
            provider_code=value.get("provider_code"),
            message=value.get("message"),
            response_cost=float(value["response_cost"]) if value.get("response_cost") is not None else None,
            billing_event_id=value.get("billing_event_id"),
            api_key=value.get("api_key"),
            user_id=value.get("user_id"),
            organization_id=value.get("organization_id"),
            scale=scale,
            project_id=value.get("project_id"),
            artifact_id=value.get("artifact_id"),
            attribution_user_id=value.get("attribution_user_id"),
            terminal_result=terminal_result,
            resolution_tombstone=value.get("resolution_tombstone"),
            task_state=task_state,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "team_id": self.team_id,
                "model": self.model,
                "request_id": self.request_id,
                "fingerprint": self.fingerprint,
                "submission_state": self.submission_state,
                "deployment_id": self.deployment_id,
                "provider_task_id": self.provider_task_id,
                "resume_token": self.resume_token,
                "provider_code": self.provider_code,
                "message": self.message,
                "response_cost": self.response_cost,
                "billing_event_id": self.billing_event_id,
                "api_key": self.api_key,
                "user_id": self.user_id,
                "organization_id": self.organization_id,
                "scale": self.scale,
                "project_id": self.project_id,
                "artifact_id": self.artifact_id,
                "attribution_user_id": self.attribution_user_id,
                "terminal_result": self.terminal_result,
                "resolution_tombstone": self.resolution_tombstone,
                "task_state": self.task_state,
            },
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class ReceiptClaim:
    outcome: ClaimOutcome
    receipt_key: str
    receipt: StoredReceipt | None


def request_fingerprint(payload: Mapping[str, object], model: str) -> str:
    canonical = {
        "source_sha256": payload.get("source_sha256"),
        "model": model,
        "style": payload.get("style", "Standard V2"),
        "scale": payload.get("scale", 2),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class LibTVReceiptStore:
    def __init__(self, redis_client):
        self.redis = redis_client

    @staticmethod
    def _identity(team_id: str, model: str, request_id: str) -> str:
        return hashlib.sha256(f"{team_id}\0{model}\0{request_id}".encode()).hexdigest()

    @classmethod
    def _index_key(cls, team_id: str, model: str, request_id: str) -> str:
        return f"libtv:receipt:index:{cls._identity(team_id, model, request_id)}"

    @classmethod
    def _pool_key(cls, team_id: str, model: str, request_id: str) -> str:
        return f"libtv:receipt:pool:{cls._identity(team_id, model, request_id)}"

    @classmethod
    def receipt_key(cls, team_id: str, model: str, request_id: str, fingerprint: str, deployment_id: str) -> str:
        identity = "\0".join((team_id, model, request_id, fingerprint, deployment_id))
        return f"libtv:receipt:{hashlib.sha256(identity.encode()).hexdigest()}"

    @staticmethod
    def _pool_record(receipt: StoredReceipt, receipt_key: str) -> str:
        attempt = {
            "deployment_id": receipt.deployment_id,
            "receipt_key": receipt_key,
            "submission_state": receipt.submission_state,
        }
        return json.dumps(
            {
                "fingerprint": receipt.fingerprint,
                "submission_state": receipt.submission_state,
                "deployment_id": receipt.deployment_id,
                "receipt_key": receipt_key,
                "attempts": [attempt],
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    async def claim(
        self,
        team_id: str,
        model: str,
        request_id: str,
        fingerprint: str,
        deployment_id: str,
        *,
        response_cost: float | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        scale: int | None = None,
        project_id: str | None = None,
        artifact_id: str | None = None,
        attribution_user_id: str | None = None,
    ) -> ReceiptClaim:
        index_key = self._index_key(team_id, model, request_id)
        pool_key = self._pool_key(team_id, model, request_id)
        receipt_key = self.receipt_key(team_id, model, request_id, fingerprint, deployment_id)
        receipt = StoredReceipt(
            team_id=team_id,
            model=model,
            request_id=request_id,
            fingerprint=fingerprint,
            submission_state="submitting",
            deployment_id=deployment_id,
            response_cost=response_cost,
            api_key=api_key,
            user_id=user_id,
            organization_id=organization_id,
            scale=scale,
            project_id=project_id,
            artifact_id=artifact_id,
            attribution_user_id=attribution_user_id,
        )
        pool = self._pool_record(receipt, receipt_key)
        try:
            result = await self.redis.eval(
                _CLAIM_SCRIPT,
                3,
                index_key,
                pool_key,
                receipt_key,
                fingerprint,
                deployment_id,
                receipt.to_json(),
                pool,
            )
        except ResponseError as error:
            raise RedisError("receipt store Lua CAS is unavailable") from error
        outcome = self._text(result[0])
        stored = StoredReceipt.from_json(self._text(result[2])) if len(result) > 2 and outcome != "mismatch" else None
        return ReceiptClaim(outcome=outcome, receipt_key=self._text(result[1]), receipt=stored)

    async def transition(
        self,
        receipt: StoredReceipt,
        receipt_key: str,
        submission_state: ReceiptState,
        *,
        provider_task_id: str | None = None,
        resume_token: str | None = None,
        provider_code: str | None = None,
        message: str | None = None,
        response_cost: float | None = None,
        expected_state: ReceiptState = "submitting",
        billing_event: Mapping[str, object] | object | None = None,
        resolution_tombstone: dict | None = None,
        task_state: TaskState | None = None,
        terminal_result: TerminalResult | Mapping[str, object] | None = None,
    ) -> StoredReceipt:
        updated = StoredReceipt(
            team_id=receipt.team_id,
            model=receipt.model,
            request_id=receipt.request_id,
            fingerprint=receipt.fingerprint,
            submission_state=submission_state,
            deployment_id=receipt.deployment_id,
            provider_task_id=provider_task_id or receipt.provider_task_id,
            resume_token=resume_token or receipt.resume_token,
            provider_code=provider_code or receipt.provider_code,
            message=message or receipt.message,
            response_cost=response_cost if response_cost is not None else getattr(receipt, "response_cost", None),
            billing_event_id=(
                str(billing_event.get("event_id") or billing_event.get("billing_key"))
                if isinstance(billing_event, Mapping) and billing_event.get("event_id")
                else (
                    billing_event.event_id
                    if billing_event is not None and hasattr(billing_event, "event_id")
                    else getattr(receipt, "billing_event_id", None)
                )
            ),
            api_key=getattr(receipt, "api_key", None),
            user_id=getattr(receipt, "user_id", None),
            organization_id=getattr(receipt, "organization_id", None),
            scale=getattr(receipt, "scale", None),
            project_id=getattr(receipt, "project_id", None),
            artifact_id=getattr(receipt, "artifact_id", None),
            attribution_user_id=getattr(receipt, "attribution_user_id", None),
            terminal_result=_terminal_result(terminal_result) if terminal_result is not None else getattr(receipt, "terminal_result", None),
            resolution_tombstone=resolution_tombstone or getattr(receipt, "resolution_tombstone", None),
            task_state=task_state or getattr(receipt, "task_state", "active"),
        )
        if updated.task_state == "succeeded" and updated.terminal_result is None:
            raise ValueError("succeeded receipt requires a terminal result")
        if (
            updated.terminal_result is not None
            and updated.provider_task_id != updated.terminal_result["provider_task_id"]
        ):
            raise ValueError("terminal result task does not match receipt")
        index_key = self._index_key(receipt.team_id, receipt.model, receipt.request_id)
        pool_key = self._pool_key(receipt.team_id, receipt.model, receipt.request_id)
        event_json = ""
        if billing_event is not None:
            if hasattr(billing_event, "to_dict"):
                event_value = billing_event.to_dict()
            elif isinstance(billing_event, Mapping):
                event_value = dict(billing_event)
            else:
                raise TypeError("billing_event must be a mapping or provide to_dict()")
            event_json = json.dumps(event_value, separators=(",", ":"), sort_keys=True)
        eval_args = [index_key, pool_key, receipt_key]
        if event_json:
            event_id = updated.billing_event_id
            if not event_id:
                raise ValueError("billing event must have an event_id")
            eval_args.extend(("libtv:billing:outbox", f"libtv:billing:outbox:delivered:{event_id}"))
        eval_args.extend(
            [
                expected_state,
                receipt.deployment_id or "",
                updated.to_json(),
                self._pool_record(updated, receipt_key),
                event_json,
            ]
        )
        try:
            result = await self.redis.eval(
                _TRANSITION_SCRIPT,
                5 if event_json else 3,
                *eval_args,
            )
        except ResponseError as error:
            raise RedisError("receipt store Lua CAS is unavailable") from error
        outcome = self._text(result[0])
        if outcome == "missing":
            return StoredReceipt(
                team_id=receipt.team_id,
                model=receipt.model,
                request_id=receipt.request_id,
                fingerprint=receipt.fingerprint,
                submission_state="unknown",
                deployment_id=receipt.deployment_id,
                message="receipt missing after pending claim",
                api_key=getattr(receipt, "api_key", None),
                user_id=getattr(receipt, "user_id", None),
                organization_id=getattr(receipt, "organization_id", None),
                scale=getattr(receipt, "scale", None),
                project_id=getattr(receipt, "project_id", None),
                artifact_id=getattr(receipt, "artifact_id", None),
                attribution_user_id=getattr(receipt, "attribution_user_id", None),
            )
        if outcome == "conflict":
            raise RedisError("receipt deployment transition conflict")
        if outcome == "resolved":
            return StoredReceipt.from_json(self._text(result[1]))
        if outcome == "terminal":
            return StoredReceipt.from_json(self._text(result[1]))
        return updated

    async def get(self, team_id: str, model: str, request_id: str) -> StoredReceipt | None:
        index_key = self._index_key(team_id, model, request_id)
        receipt_key = await self.redis.get(index_key)
        if receipt_key is None:
            return None
        raw = await self.redis.get(receipt_key)
        if raw is None:
            return None
        return StoredReceipt.from_json(self._text(raw))

    async def get_receipt(self, team_id: str, model: str, request_id: str) -> StoredReceipt | None:
        return await self.get(team_id, model, request_id)

    async def get_attempts(self, team_id: str, model: str, request_id: str) -> tuple[StoredReceipt, ...]:
        raw = await self.redis.get(self._pool_key(team_id, model, request_id))
        if raw is None:
            return ()
        pool = json.loads(self._text(raw))
        attempts = pool.get("attempts") or (() if not pool.get("receipt_key") else (pool,))
        values = await asyncio.gather(
            *(self.redis.get(attempt.get("receipt_key")) for attempt in attempts if attempt.get("receipt_key"))
        )
        return tuple(StoredReceipt.from_json(self._text(value)) for value in values if value is not None)

    async def recover_pending(self, team_id: str, model: str, request_id: str) -> StoredReceipt | None:
        current = await self.get(team_id, model, request_id)
        if current is None or current.submission_state != "submitting":
            return current
        index_key = self._index_key(team_id, model, request_id)
        receipt_key_value = await self.redis.get(index_key)
        if receipt_key_value is None:
            return None
        receipt_key = self._text(receipt_key_value)
        return await self.transition(current, receipt_key, "unknown", message="recovered pending submission")

    async def wait(self, claim: ReceiptClaim, timeout: float = 2.0) -> StoredReceipt | None:
        if claim.receipt is None:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            current = await self.get(claim.receipt.team_id, claim.receipt.model, claim.receipt.request_id)
            if current is not None and current.submission_state != "submitting":
                return current
            await asyncio.sleep(0.01)
        return await self.get(claim.receipt.team_id, claim.receipt.model, claim.receipt.request_id)

    async def readiness(self) -> bool:
        await self.redis.ping()
        config = await self.redis.config_get("appendonly", "appendfsync")
        values = {str(key).lower(): str(value).lower() for key, value in config.items()}
        if values.get("appendonly") != "yes" or values.get("appendfsync") != "always":
            return False
        probe = f"libtv:receipt:readiness:{id(self)}"
        try:
            await self.redis.set(probe, "ok")
            return (await self.redis.get(probe)) == "ok"
        finally:
            await self.redis.delete(probe)

    async def readiness_check(self) -> bool:
        return await self.readiness()

    @staticmethod
    def _text(value) -> str:
        if isinstance(value, bytes):
            return value.decode()
        return str(value)
