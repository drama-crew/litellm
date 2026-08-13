"""Durable billing events for asynchronous libtv image tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

BILLING_STREAM_KEY = "libtv:billing:outbox"
BILLING_CONSUMER_GROUP = "libtv-billing-reconciler"
BILLING_EVENT_FIELD = "payload"


@dataclass(frozen=True, slots=True)
class ImageBillingEvent:
    deployment_id: str
    provider_task_id: str
    response_cost: float
    team_id: str | None = None
    user_id: str | None = None
    organization_id: str | None = None
    api_key: str | None = None
    scale: int | None = None
    project_id: str | None = None
    artifact_id: str | None = None
    attribution_user_id: str | None = None
    model: str = "topaz-image-upscaler"
    event_id: str = field(default="")
    occurred_at: str = field(default="")

    def __post_init__(self) -> None:
        if not self.event_id:
            object.__setattr__(self, "event_id", self.billing_key)
        if not self.occurred_at:
            object.__setattr__(self, "occurred_at", datetime.now(timezone.utc).isoformat())

    @property
    def billing_key(self) -> str:
        return f"libtv-image:{self.deployment_id}:{self.provider_task_id}"

    @property
    def request_id(self) -> str:
        return self.billing_key

    @property
    def org_id(self) -> str | None:
        return self.organization_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"request_id": self.request_id, "billing_key": self.billing_key}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageBillingEvent":
        return cls(
            deployment_id=str(value["deployment_id"]),
            provider_task_id=str(value.get("provider_task_id") or value["task_id"]),
            response_cost=float(value.get("response_cost", value.get("spend", 0.0))),
            team_id=_optional_str(value.get("team_id")),
            user_id=_optional_str(value.get("user_id")),
            organization_id=_optional_str(value.get("organization_id", value.get("org_id"))),
            api_key=_optional_str(value.get("api_key", value.get("key_id"))),
            scale=value.get("scale")
            if isinstance(value.get("scale"), int) and not isinstance(value.get("scale"), bool)
            else None,
            project_id=_optional_str(value.get("project_id")),
            artifact_id=_optional_str(value.get("artifact_id")),
            attribution_user_id=_optional_str(value.get("attribution_user_id")),
            model=str(value.get("model") or "topaz-image-upscaler"),
            event_id=str(value.get("event_id") or ""),
            occurred_at=str(value.get("occurred_at") or ""),
        )


BillingOutboxEvent = ImageBillingEvent


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _event_from_stream(fields: Mapping[Any, Any]) -> ImageBillingEvent:
    payload = fields.get(BILLING_EVENT_FIELD)
    if payload is None:
        payload = fields.get(BILLING_EVENT_FIELD.encode())
    if isinstance(payload, bytes):
        payload = payload.decode()
    if not isinstance(payload, str):
        raise ValueError("libtv billing event has no payload")
    return ImageBillingEvent.from_dict(json.loads(payload))


class LibTVBillingReconciler:
    def __init__(
        self,
        redis_client: Any,
        prisma_client: Any,
        *,
        stream_key: str = BILLING_STREAM_KEY,
        consumer_group: str = BILLING_CONSUMER_GROUP,
        consumer: str | None = None,
        poll_interval: float = 1.0,
        batch_size: int = 100,
    ) -> None:
        self.redis = redis_client
        self.prisma_client = prisma_client
        self.stream_key = stream_key
        self.consumer_group = consumer_group
        self.consumer = consumer or f"{uuid.uuid4()}"
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._group_ready = False

    async def ensure_group(self) -> None:
        if self._group_ready:
            return
        try:
            await self.redis.xgroup_create(
                name=self.stream_key,
                groupname=self.consumer_group,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise
        self._group_ready = True

    async def _read_events(self) -> list[tuple[str, Mapping[Any, Any]]]:
        events: list[tuple[str, Mapping[Any, Any]]] = []
        try:
            reclaimed = await self.redis.xautoclaim(
                self.stream_key,
                self.consumer_group,
                self.consumer,
                min_idle_time=0,
                start_id="0-0",
                count=self.batch_size,
            )
            entries = reclaimed[1] if isinstance(reclaimed, (tuple, list)) and len(reclaimed) > 1 else []
            events.extend((event_id, fields) for event_id, fields in entries)
        except (AttributeError, NotImplementedError):
            pending = await self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer,
                streams={self.stream_key: "0"},
                count=self.batch_size,
            )
            events.extend(_flatten_stream_entries(pending, self.stream_key))

        if len(events) < self.batch_size:
            fresh = await self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer,
                streams={self.stream_key: ">"},
                count=self.batch_size - len(events),
                block=1,
            )
            events.extend(_flatten_stream_entries(fresh, self.stream_key))
        return events

    async def reconcile_once(self) -> int:
        await self.ensure_group()
        events = await self._read_events()
        processed = 0
        for event_id, fields in events:
            event = _event_from_stream(fields)
            await self._reconcile_event(event)
            await self.redis.xack(self.stream_key, self.consumer_group, event_id)
            processed += 1
        return processed

    async def _reconcile_event(self, event: ImageBillingEvent) -> None:
        db = getattr(self.prisma_client, "db", self.prisma_client)
        async with db.tx() as transaction:
            inserted = await transaction.execute_raw(
                'INSERT INTO "LiteLLM_SpendLogs" '
                "(request_id, call_type, api_key, spend, total_tokens, prompt_tokens, "
                'completion_tokens, "startTime", "endTime", model, "user", metadata, '
                "team_id, organization_id) "
                "VALUES ($1, $2, $3, $4, 0, 0, 0, $5, $5, $6, $7, $8, $9, $10) "
                "ON CONFLICT (request_id) DO NOTHING",
                event.request_id,
                "image_upscale",
                event.api_key or "",
                event.response_cost,
                _event_time(event.occurred_at),
                event.model,
                event.user_id or "",
                json.dumps(
                    {
                        "libtv_billing_key": event.billing_key,
                        **({"scale": event.scale} if event.scale is not None else {}),
                        **({"project_id": event.project_id} if event.project_id else {}),
                        **({"artifact_id": event.artifact_id} if event.artifact_id else {}),
                        **({"user_id": event.attribution_user_id} if event.attribution_user_id else {}),
                    }
                ),
                event.team_id,
                event.organization_id,
            )
            if not inserted:
                return
            await _increment_spend(
                transaction, '"LiteLLM_VerificationToken"', "token", event.api_key, event.response_cost
            )
            await _increment_spend(transaction, '"LiteLLM_TeamTable"', "team_id", event.team_id, event.response_cost)
            await _increment_spend(transaction, '"LiteLLM_UserTable"', "user_id", event.user_id, event.response_cost)
            await _increment_spend(
                transaction,
                '"LiteLLM_OrganizationTable"',
                "organization_id",
                event.organization_id,
                event.response_cost,
            )

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("libtv billing outbox reconciliation failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        await self.ensure_group()
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name="libtv-billing-reconciler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None


def _flatten_stream_entries(result: Any, stream_key: str) -> list[tuple[str, Mapping[Any, Any]]]:
    if not result:
        return []
    for name, entries in result:
        if name == stream_key or (isinstance(name, bytes) and name.decode() == stream_key):
            return [
                (event_id.decode() if isinstance(event_id, bytes) else str(event_id), fields)
                for event_id, fields in entries
            ]
    return []


def _event_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        parsed = datetime.now(timezone.utc)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


async def _increment_spend(
    transaction: Any, table: str, identifier_column: str, identifier: str | None, amount: float
) -> None:
    if not identifier:
        return
    await transaction.execute_raw(
        f"UPDATE {table} SET spend = spend + $1 WHERE {identifier_column} = $2",
        amount,
        identifier,
    )


async def start_libtv_billing_reconciler(prisma_client: Any) -> LibTVBillingReconciler | None:
    if prisma_client is None:
        return None
    from litellm.llms.libtv.persistence import get_receipt_store

    receipt_store = get_receipt_store()
    if receipt_store is None:
        return None
    reconciler = LibTVBillingReconciler(receipt_store.redis, prisma_client)
    await reconciler.start()
    return reconciler


LibTVBillingOutbox = LibTVBillingReconciler
BillingOutboxReconciler = LibTVBillingReconciler
