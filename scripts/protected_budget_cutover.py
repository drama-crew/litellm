import argparse
import asyncio
import os
from pathlib import Path

from litellm.proxy.spend_tracking.protected_budget import CutoverReceipt, ProtectedBudgetStore
from litellm.proxy.video_endpoints.moderation_metering import MeteringStore


async def execute(receipt_path: Path, namespace: str, database_env: str, redis_env: str) -> None:
    from prisma import Prisma
    from redis.asyncio import Redis

    receipt = CutoverReceipt.model_validate_json(receipt_path.read_text())
    db = Prisma(datasource={"url": os.environ[database_env]})
    redis = Redis.from_url(os.environ[redis_env], decode_responses=True)
    try:
        await db.connect()
        meter = MeteringStore.from_client(db, redis, namespace=namespace)
        authority = ProtectedBudgetStore(meter.db, meter.transactions, meter.redis, namespace=namespace)
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
    parser.add_argument("--namespace", default="")
    parser.add_argument("--database-env", default="DATABASE_URL")
    parser.add_argument("--redis-env", default="REDIS_URL")
    args = parser.parse_args()
    asyncio.run(execute(args.receipt, args.namespace, args.database_env, args.redis_env))


if __name__ == "__main__":
    main()
