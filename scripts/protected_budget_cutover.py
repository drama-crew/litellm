import argparse
import asyncio
import os
from pathlib import Path

from pydantic import TypeAdapter

from litellm.proxy.spend_tracking.protected_budget import ReceiptRow
from litellm.proxy.video_endpoints.moderation_metering_cache import keys

from litellm.proxy.spend_tracking.protected_budget import CutoverReceipt, ProtectedBudgetStore
from litellm.proxy.video_endpoints.moderation_metering import MeteringStore


INSPECT_COUNTER = """
if redis.call('PTTL', KEYS[3]) ~= -1 then return 'mode_ttl_or_missing' end
if redis.call('GET', KEYS[3]) ~= '\"protected-v1\"' then return 'mode' end
if redis.call('PTTL', KEYS[1]) ~= -1 or redis.call('PTTL', KEYS[2]) ~= -1 then return 'ttl_or_missing' end
if redis.call('HGET', KEYS[2], 'generation') ~= ARGV[1] then return 'generation' end
if redis.call('HGET', KEYS[2], 'seq') ~= ARGV[2] then return 'sequence' end
if redis.call('HGET', KEYS[2], 'cutover') ~= ARGV[3] then return 'cutover_digest' end
return 'ok'
"""


async def inspect_cutover(authority: ProtectedBudgetStore, receipt: CutoverReceipt) -> bool:
    rows = TypeAdapter(tuple[ReceiptRow, ...]).validate_python(
        await authority.db.query_raw(
            'SELECT payload,payload_hash,status FROM "LiteLLM_BudgetCutover" WHERE receipt_id=$1',
            receipt.receipt_id,
        )
    )
    if len(rows) != 1 or rows[0].payload_hash != receipt.digest() or rows[0].payload != receipt or rows[0].status != "ready":
        return False
    states = await authority.states(authority.db, tuple(b.target.counter_key for b in receipt.baselines))
    if len(states) != len(receipt.baselines):
        return False
    for state in states:
        if state.status != "ready" or state.pending_operation is not None:
            return False
        result = await authority.redis.eval(
            INSPECT_COUNTER, 3, *keys(state.counter_key, authority.namespace),
            authority.namespace + "protected:mode", state.generation, str(state.committed_seq), receipt.digest(),
        )
        if result != "ok":
            return False
    return True


async def execute(receipt_path: Path, namespace: str, database_env: str, redis_env: str, inspect_only: bool = False) -> None:
    from prisma import Prisma
    from redis.asyncio import Redis

    receipt = CutoverReceipt.model_validate_json(receipt_path.read_text())
    db = Prisma(datasource={"url": os.environ[database_env]})
    redis = Redis.from_url(os.environ[redis_env], decode_responses=True)
    try:
        await db.connect()
        meter = MeteringStore.from_client(db, redis, namespace=namespace)
        authority = ProtectedBudgetStore(meter.db, meter.transactions, meter.redis, namespace=namespace)
        if inspect_only:
            if not await inspect_cutover(authority, receipt):
                raise SystemExit("protected cutover pending or evidence unavailable; no state changed")
            print("protected cutover inspection ready:", receipt.digest())
            return
        await authority.register_cutover(receipt)
        await authority.assert_admission(tuple(b.target.counter_key for b in receipt.baselines))
        print("operator-attested cutover ready:", receipt.receipt_id, receipt.digest())
    finally:
        try:
            await redis.aclose()
        finally:
            if db.is_connected():
                await db.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply or recover an operator-attested protected budget cutover")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--database-env", default="DATABASE_URL")
    parser.add_argument("--redis-env", default="REDIS_URL")
    args = parser.parse_args()
    try:
        asyncio.run(execute(args.receipt, args.namespace, args.database_env, args.redis_env, args.inspect))
    except Exception:
        raise SystemExit("protected budget operation failed; preserve maintenance fence and inspect evidence") from None


if __name__ == "__main__":
    main()
