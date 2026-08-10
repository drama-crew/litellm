import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping

from redis.exceptions import RedisError, ResponseError

ReceiptState = Literal["not_submitted", "rejected", "unknown", "submitted", "submitting"]
ClaimOutcome = Literal["owner", "existing", "rejected", "missing", "mismatch"]

_CLAIM_SCRIPT = """
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
redis.call('SET', KEYS[1], KEYS[3])
redis.call('SET', KEYS[2], ARGV[4])
redis.call('SET', KEYS[3], ARGV[3])
return {'owner', KEYS[3], ARGV[3]}
"""

_TRANSITION_SCRIPT = """
local stored = redis.call('GET', KEYS[3])
if not stored then
  return {'missing'}
end
local current = cjson.decode(stored)
if current['deployment_id'] ~= ARGV[1] then
  return {'conflict', stored}
end
redis.call('SET', KEYS[3], ARGV[2])
redis.call('SET', KEYS[2], ARGV[3])
return {'ok', ARGV[2]}
"""

_FALLBACK_LOCK = asyncio.Lock()


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

    @classmethod
    def from_json(cls, raw: str) -> "StoredReceipt":
        value = json.loads(raw)
        state = value["submission_state"]
        if state not in {"not_submitted", "rejected", "unknown", "submitted", "submitting"}:
            raise ValueError("invalid receipt state")
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
    def receipt_key(cls, team_id: str, model: str, request_id: str, fingerprint: str) -> str:
        identity = "\0".join((team_id, model, request_id, fingerprint))
        return f"libtv:receipt:{hashlib.sha256(identity.encode()).hexdigest()}"

    @staticmethod
    def _pool_record(receipt: StoredReceipt, receipt_key: str) -> str:
        return json.dumps(
            {
                "fingerprint": receipt.fingerprint,
                "submission_state": receipt.submission_state,
                "deployment_id": receipt.deployment_id,
                "receipt_key": receipt_key,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    async def claim(
        self, team_id: str, model: str, request_id: str, fingerprint: str, deployment_id: str
    ) -> ReceiptClaim:
        index_key = self._index_key(team_id, model, request_id)
        pool_key = self._pool_key(team_id, model, request_id)
        receipt_key = self.receipt_key(team_id, model, request_id, fingerprint)
        receipt = StoredReceipt(
            team_id=team_id,
            model=model,
            request_id=request_id,
            fingerprint=fingerprint,
            submission_state="submitting",
            deployment_id=deployment_id,
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
            if "unknown command 'eval'" not in str(error):
                raise
            result = await self._claim_without_lua(index_key, pool_key, receipt_key, receipt, pool)
        outcome = self._text(result[0])
        stored = StoredReceipt.from_json(self._text(result[2])) if len(result) > 2 and outcome != "mismatch" else None
        return ReceiptClaim(outcome=outcome, receipt_key=self._text(result[1]), receipt=stored)

    async def _claim_without_lua(
        self, index_key: str, pool_key: str, receipt_key: str, receipt: StoredReceipt, pool: str
    ) -> list[str]:
        async with _FALLBACK_LOCK:
            indexed = await self.redis.get(index_key)
            if indexed is not None:
                indexed = self._text(indexed)
                stored = await self.redis.get(indexed)
                if stored is None:
                    pool_raw = await self.redis.get(pool_key)
                    if pool_raw is not None:
                        pool_value = json.loads(self._text(pool_raw))
                        if pool_value["fingerprint"] != receipt.fingerprint:
                            return ["mismatch", indexed, self._text(pool_raw)]
                    return ["missing", indexed]
                stored_text = self._text(stored)
                current = StoredReceipt.from_json(stored_text)
                if current.fingerprint != receipt.fingerprint:
                    return ["mismatch", indexed, stored_text]
                if current.submission_state in {"submitting", "unknown", "submitted"}:
                    return ["existing", indexed, stored_text]
                if current.deployment_id == receipt.deployment_id:
                    return [current.submission_state, indexed, stored_text]
            pool_raw = await self.redis.get(pool_key)
            if pool_raw is not None:
                pool_value = json.loads(self._text(pool_raw))
                if pool_value["fingerprint"] != receipt.fingerprint:
                    return ["mismatch", receipt_key, self._text(pool_raw)]
                if pool_value["submission_state"] in {"submitting", "unknown", "submitted"}:
                    pool_receipt_key = pool_value["receipt_key"]
                    pool_stored = await self.redis.get(pool_receipt_key)
                    if pool_stored is None:
                        return ["missing", pool_receipt_key]
                    return ["existing", pool_receipt_key, self._text(pool_stored)]
            await self.redis.set(index_key, receipt_key)
            await self.redis.set(pool_key, pool)
            await self.redis.set(receipt_key, receipt.to_json())
            return ["owner", receipt_key, receipt.to_json()]

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
        )
        index_key = self._index_key(receipt.team_id, receipt.model, receipt.request_id)
        pool_key = self._pool_key(receipt.team_id, receipt.model, receipt.request_id)
        try:
            result = await self.redis.eval(
                _TRANSITION_SCRIPT,
                3,
                index_key,
                pool_key,
                receipt_key,
                receipt.deployment_id or "",
                updated.to_json(),
                self._pool_record(updated, receipt_key),
            )
        except ResponseError as error:
            if "unknown command 'eval'" not in str(error):
                raise
            result = await self._transition_without_lua(index_key, pool_key, receipt_key, receipt, updated)
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
            )
        if outcome == "conflict":
            raise RedisError("receipt deployment transition conflict")
        return updated

    async def _transition_without_lua(
        self,
        index_key: str,
        pool_key: str,
        receipt_key: str,
        receipt: StoredReceipt,
        updated: StoredReceipt,
    ) -> list[str]:
        async with _FALLBACK_LOCK:
            stored = await self.redis.get(receipt_key)
            if stored is None:
                return ["missing"]
            current = StoredReceipt.from_json(self._text(stored))
            if current.deployment_id != receipt.deployment_id:
                return ["conflict", self._text(stored)]
            await self.redis.set(receipt_key, updated.to_json())
            await self.redis.set(pool_key, self._pool_record(updated, receipt_key))
            return ["ok", updated.to_json()]

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
        await self.redis.set(probe, "ok")
        return (await self.redis.get(probe)) == "ok"

    async def readiness_check(self) -> bool:
        return await self.readiness()

    @staticmethod
    def _text(value) -> str:
        if isinstance(value, bytes):
            return value.decode()
        return str(value)
