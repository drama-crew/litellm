from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from litellm.llms.libtv.billing_outbox import BudgetAuthorityDependencyPending, CausynBillingEvent, ImageBillingEvent
from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
from litellm.proxy.video_endpoints.moderation_metering import PhaseEvent
from litellm.proxy.video_endpoints.moderation_metering_projection import (
    BillingFacts,
    FinancialProjection,
    LegacyMetadata,
)


async def settle_outbox(event: ImageBillingEvent | CausynBillingEvent) -> bool:
    if isinstance(event, CausynBillingEvent) and event.metering_event_json:
        phase = PhaseEvent.model_validate_json(event.metering_event_json)
        if (
            phase.binding.fingerprint,
            phase.binding.user_id,
            phase.binding.team_id,
            phase.binding.organization_id,
            phase.provider_task_id,
            phase.amount,
            phase.provider,
            phase.deployment_id,
        ) != (
            event.api_key,
            event.user_id,
            event.team_id,
            event.organization_id,
            event.provider_task_id,
            Decimal(str(event.response_cost)),
            event.provider,
            event.deployment_id,
        ):
            raise ValueError("outbox financial phase identity mismatch")
        authority = runtime.store()
        await authority.persist(phase)
        await authority.run_once()
        receipt = await authority.settlement(phase.binding)
        if phase not in receipt.receipts:
            raise BudgetAuthorityDependencyPending("actual phase receipt pending")
        return True
    if isinstance(event, CausynBillingEvent) and event.moderation_intent_id:
        raise BudgetAuthorityDependencyPending("moderated actual requires exact phase proof")
    first = next(
        (
            f"spend:{kind}:{identity}"
            for kind, identity in (
                ("key", event.api_key),
                ("team", event.team_id),
                ("user", event.user_id),
                ("org", event.organization_id),
            )
            if identity
        ),
        None,
    )
    if first is None or not await runtime.counter_mode(first):
        return False
    timestamp = datetime.fromisoformat(event.occurred_at)
    projection = FinancialProjection(
        request_id=event.request_id,
        fingerprint=event.api_key,
        user_id=event.user_id,
        team_id=event.team_id,
        organization_id=event.organization_id,
        provider=event.provider if isinstance(event, CausynBillingEvent) else "libtv",
        deployment_id=event.deployment_id,
        provider_task_id=event.provider_task_id,
        model=event.model,
        amount=Decimal(str(event.response_cost)),
        legacy_metadata=LegacyMetadata(causyn_billing_key=event.billing_key, provider=event.provider)
        if isinstance(event, CausynBillingEvent)
        else LegacyMetadata(
            libtv_billing_key=event.billing_key,
            scale=event.scale,
            project_id=event.project_id,
            artifact_id=event.artifact_id,
            user_id=event.attribution_user_id,
        ),
        facts=BillingFacts.model_validate_json(event.billing_facts_json)
        if isinstance(event, CausynBillingEvent) and event.billing_facts_json
        else BillingFacts(
            started_at=timestamp,
            ended_at=timestamp,
            route=event.task_type if isinstance(event, CausynBillingEvent) else "image_upscale",
        ),
    )
    result = await runtime.settle_legacy(
        event.request_id,
        token=event.api_key,
        user_id=event.user_id,
        team_id=event.team_id,
        org_id=event.organization_id,
        end_user_id=None,
        tags=(),
        amount=event.response_cost,
        reservation_id=event.reservation_id if isinstance(event, CausynBillingEvent) else None,
        financial=projection,
    )
    return bool(result.counter_keys)
