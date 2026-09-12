from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.causyn.context_ir import ContextIRService, get_context_ir_service
from litellm.llms.causyn.context_ir_budget import release_unaccepted_reservation, require_durable_budget
from litellm.llms.causyn.context_ir_callback import verify_callback
from litellm.llms.causyn.context_ir_store import BillingIdentity, ContextIRTask, new_task_id
from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.video_endpoints.minimax_h3_endpoints import MiniMaxH3Route, task_owner

router = APIRouter(route_class=MiniMaxH3Route)
Auth = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]


async def context_ir_service() -> ContextIRService:
    return get_context_ir_service()


Service = Annotated[ContextIRService, Depends(context_ir_service)]


class ListParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_num: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = Field(
        default=None, alias="filter.status"
    )
    task_ids: tuple[str, ...] = Field(default=(), alias="filter.task_ids")
    model: str | None = Field(default=None, alias="filter.model")
    task_type: Literal["generation", "h3_context_ir", "regeneration"] | None = Field(
        default=None, alias="filter.task_type"
    )


@router.post("/v2/h3_context_ir", tags=["MiniMax H3"])
async def create_context_ir(request: Request, auth: Auth, service: Service) -> dict[str, str]:
    spec = request.scope.get("causyn_context_ir_spec")
    if not isinstance(spec, ContextIRRequest):
        raise RewriteError("Invalid Context IR request", 400)
    from litellm.llms.causyn.h3_prompt import AUTH_MODEL
    from litellm.proxy.video_endpoints import moderation_bridge

    moderated = await moderation_bridge.submit(
        request, auth, spec.model_dump(mode="json"), "context_ir", policy_model=AUTH_MODEL
    )
    if moderated is not None:
        return {"task_id": moderated.id}
    reservation = TypeAdapter[dict[str, JsonValue] | None](dict[str, JsonValue] | None).validate_python(
        auth.budget_reservation
    )
    try:
        require_durable_budget(reservation)
        await verify_callback(spec.callback_url)
    except BaseException:
        await release_unaccepted_reservation(reservation)
        raise
    identity = auth.api_key or auth.token or ""
    key_hash = identity if re.fullmatch(r"[0-9a-f]{64}", identity) else hashlib.sha256(identity.encode()).hexdigest()
    from litellm.proxy.video_endpoints.moderation_metering_entry import execute as metered_execute

    task = await metered_execute(
        request,
        auth,
        "context_ir",
        accept_context_ir(
            service,
            spec,
            task_owner(auth),
            reservation,
            BillingIdentity(api_key=key_hash, team_id=auth.team_id, user_id=auth.user_id, organization_id=auth.org_id),
        ),
    )
    if task.metering_binding_json:
        from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent, request_id
        from litellm.proxy.video_endpoints.moderation_metering_runtime import store

        binding = BillingBinding.model_validate_json(task.metering_binding_json)
        event = PhaseEvent(
            binding=binding,
            request_id=request_id(binding, "completion"),
            phase="completion",
            provider="causyn",
            deployment_id=AUTH_MODEL,
            native_id=task.id,
            provider_task_id=task.id,
        )
        request.scope["moderation_context_ir_event"] = event

        async def persist() -> None:
            await store().persist(event)

        await asyncio.gather(persist(), return_exceptions=True)
    return {"task_id": task.id}


async def accept_context_ir(
    service: ContextIRService,
    spec: ContextIRRequest,
    owner: str,
    reservation: dict[str, JsonValue] | None,
    billing: BillingIdentity,
) -> ContextIRTask:
    task_id = new_task_id(owner)
    try:
        return await service.create(spec, owner=owner, reservation=reservation, billing=billing, task_id=task_id)
    except Exception:
        persisted = await service.store.get(task_id)
        if persisted is not None:
            return persisted
        await release_unaccepted_reservation(reservation)
        raise


@router.get("/v2/query/video_generation", tags=["MiniMax H3"])
async def list_context_ir(request: Request, auth: Auth, service: Service) -> dict[str, JsonValue]:
    from litellm.proxy.video_endpoints import moderation_bridge

    moderated = await moderation_bridge.list_tasks(request, auth, v2=True)
    if moderated is not None:
        return moderated
    params = ListParams.model_validate(
        {
            key: request.query_params.getlist(key) if key == "filter.task_ids" else value
            for key, value in request.query_params.items()
        }
    )
    rows = tuple(
        task
        for task in await service.store.list_tasks(task_owner(auth))
        if (params.status is None or task.status == params.status)
        and (params.model is None or task.request.model == params.model)
        and (not params.task_ids or task.id in params.task_ids)
        and params.task_type in {None, "h3_context_ir"}
    )
    start = (params.page_num - 1) * params.page_size
    return {"items": [task.public() for task in rows[start : start + params.page_size]], "total": len(rows)}


@router.delete("/v2/video_generation/{video_id}", tags=["MiniMax H3"])
async def delete_context_ir(video_id: str, request: Request, auth: Auth, service: Service) -> dict[str, str]:
    from litellm.proxy.video_endpoints import moderation_bridge as bridge

    if video_id.startswith(bridge.PREFIX):
        import os

        from litellm.proxy.video_endpoints.moderation_execution import Ticket, cancel

        outcome = await bridge.platform(
            request, "POST", f"/intents/{video_id}/cancel", {"principal": bridge.principal(auth)}
        )
        if isinstance(outcome.get("ticket"), str):
            outcome = await cancel(
                Ticket(ticket=outcome["ticket"]), request, "Bearer " + os.environ["DRAMA_MODERATION_SERVICE_TOKEN"]
            )
        action = TypeAdapter(str).validate_python(outcome["action"])
    else:
        action = await service.cancel_or_delete(video_id, task_owner(auth))
    return {"task_id": video_id, "action": action, "status": action}
