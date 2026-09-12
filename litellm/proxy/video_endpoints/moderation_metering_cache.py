from collections.abc import Sequence
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter


class RedisCommands(Protocol):
    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object: ...


class CounterVersion(BaseModel):
    model_config = ConfigDict(frozen=True)
    counter_key: str
    generation: str
    committed_seq: int
    reserved: Decimal = Decimal(0)


REGISTER = """
if redis.call('EXISTS', KEYS[2]) == 1 then
  if redis.call('HGET', KEYS[2], 'generation') ~= ARGV[1] then return 'generation' end
  if redis.call('EXISTS', KEYS[1]) == 0 then return 'missing' end
  redis.call('PERSIST', KEYS[1])
  return 'ok'
end
if ARGV[3] ~= 'new' then return 'missing' end
if redis.call('EXISTS', KEYS[1]) == 0 then redis.call('SET', KEYS[1], ARGV[2]) end
redis.call('PERSIST', KEYS[1])
redis.call('HSET', KEYS[2], 'generation', ARGV[1], 'seq', '0')
return 'ok'
"""

APPLY = """
local now = redis.call('TIME')
local milliseconds = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
if milliseconds >= tonumber(ARGV[3]) then return 'lease' end
for i = 1, #KEYS, 2 do
  local offset = 4 + math.floor((i - 1) / 2) * 3
  local generation, seq = ARGV[offset], tonumber(ARGV[offset + 1])
  if redis.call('EXISTS', KEYS[i]) == 0 then return 'missing' end
  if redis.call('HGET', KEYS[i + 1], 'generation') ~= generation then return 'generation' end
  local actual_seq = tonumber(redis.call('HGET', KEYS[i + 1], 'seq'))
  if actual_seq == seq + 1 then
    if redis.call('HGET', KEYS[i + 1], 'event') ~= ARGV[1] then return 'ahead' end
    if redis.call('HGET', KEYS[i + 1], 'payload') ~= ARGV[2] then return 'payload' end
  elseif actual_seq ~= seq then
    return actual_seq and actual_seq > seq and 'ahead' or 'behind'
  end
end
for i = 1, #KEYS, 2 do
  local offset = 4 + math.floor((i - 1) / 2) * 3
  local seq = tonumber(ARGV[offset + 1])
  if tonumber(redis.call('HGET', KEYS[i + 1], 'seq')) == seq then
    redis.call('INCRBYFLOAT', KEYS[i], ARGV[offset + 2])
    redis.call('HSET', KEYS[i + 1], 'seq', seq + 1, 'event', ARGV[1], 'payload', ARGV[2])
  end
  redis.call('PERSIST', KEYS[i])
end
return 'ok'
"""

CHECK = """
if redis.call('EXISTS', KEYS[1]) == 0 then return 'missing' end
if redis.call('HGET', KEYS[2], 'generation') ~= ARGV[1] then return 'generation' end
local current = tonumber(redis.call('HGET', KEYS[2], 'seq'))
if current == tonumber(ARGV[2]) then return 'ok' end
if current and current > tonumber(ARGV[2]) then return 'ahead' end
return 'behind'
"""

INCREMENT = """
if redis.call('EXISTS', KEYS[2]) == 0 then return {'legacy'} end
if redis.call('EXISTS', KEYS[1]) == 0 then return {'missing'} end
local result = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
redis.call('PERSIST', KEYS[1])
return {'ok', result}
"""


def keys(counter_key: str, namespace: str) -> tuple[str, str]:
    return namespace + counter_key, namespace + 'moderation:counter:' + counter_key


async def register(
    redis: RedisCommands, version: CounterVersion, initial: Decimal, *, namespace: str, fresh: bool
) -> str:
    return TypeAdapter(str).validate_python(await redis.eval(
        REGISTER, 2, *keys(version.counter_key, namespace), version.generation, str(initial), 'new' if fresh else 'existing'
    ))


async def adjust(
    redis: RedisCommands, counters: Sequence[CounterVersion], *, event_id: str, payload_hash: str,
    amount: Decimal, lease_until_ms: int, namespace: str
) -> str:
    counter_keys = tuple(key for counter in counters for key in keys(counter.counter_key, namespace))
    arguments = tuple(value for counter in counters for value in (
        counter.generation, str(counter.committed_seq), str(amount - counter.reserved)
    ))
    return TypeAdapter(str).validate_python(await redis.eval(
        APPLY, len(counter_keys), *counter_keys, event_id, payload_hash, str(lease_until_ms), *arguments
    ))


async def check(redis: RedisCommands, counter: CounterVersion, *, namespace: str) -> str:
    return TypeAdapter(str).validate_python(await redis.eval(
        CHECK, 2, *keys(counter.counter_key, namespace), counter.generation, str(counter.committed_seq)
    ))
