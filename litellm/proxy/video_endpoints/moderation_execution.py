from __future__ import annotations

import hmac
import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import uuid4

import httpx
import jwt
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.video_endpoints import moderation_bridge as bridge
from litellm.types.videos.main import CharacterObject, VideoObject

if TYPE_CHECKING:
    from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, MeteringStore, PhaseEvent


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ticket: str


class SettlementTicket(Ticket):
    close_completion: Literal["undelivered", "delivered"] | None = None


class Claims(BaseModel):
    intent_id: str
    request_digest: str
    input_digest: str
    model: str
    policy_version: str
    policy_digest: str
    purpose: Literal["submit", "collect", "cancel", "settlement"]


class BillingContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    intent_id: str
    model: str
    principal: dict[str, JsonValue]
    billing: dict[str, JsonValue]


BILLING_CONTEXT: ContextVar[BillingContext | None] = ContextVar("moderation_billing_context", default=None)


def authorize(ticket: str, authorization: str | None, purpose: str) -> Claims:
    secret = os.getenv("DRAMA_MODERATION_SERVICE_TOKEN", "")
    if not secret or not hmac.compare_digest(authorization or "", "Bearer " + secret):
        raise HTTPException(401, "Trusted moderation service authentication required")
    try:
        claims = Claims.model_validate(
            jwt.decode(
                ticket,
                secret,
                algorithms=["HS256"],
                audience="moderation-fork",
                options={"require": ["exp", "iat", "aud"]},
            )
        )
    except (jwt.InvalidTokenError, ValueError):
        raise HTTPException(403, "Invalid moderation continuation ticket") from None
    if claims.purpose != purpose:
        raise HTTPException(403, "Moderation ticket purpose mismatch")
    return claims


def execution_request(request: Request, payload: dict[str, JsonValue], path: str, method: str = "POST") -> Request:
    scope = {
        **{
            key: value
            for key, value in request.scope.items()
            if key not in {"parsed_body", "state", "route", "endpoint"}
        },
        "state": dict(request.scope.get("state", {})),
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            *(
                [(b"x-litellm-call-id", request.headers["x-litellm-call-id"].encode())]
                if "x-litellm-call-id" in request.headers
                else []
            ),
        ],
        "path_params": {},
        "moderation_admission": bridge.ADMITTED,
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    return Request(scope, receive)


class PreflightFailure(Exception):
    pass


async def invoke(request: Request, auth: UserAPIKeyAuth, payload: dict[str, JsonValue], route: str) -> object:
    from litellm.proxy.video_endpoints import endpoints

    prepared = execution_request(request, {**payload, "num_retries": 0}, request.url.path)
    match route:
        case "avideo_generation":
            return await endpoints.video_generation(prepared, Response(), None, auth)
        case "avideo_remix":
            return await endpoints.video_remix(
                TypeAdapter(str).validate_python(payload["source_video_id"]), prepared, Response(), auth
            )
        case "avideo_edit":
            return await endpoints.video_edit(prepared, Response(), auth)
        case "avideo_extension":
            return await endpoints.video_extension(prepared, Response(), auth)
        case "avideo_create_character":
            from tempfile import SpooledTemporaryFile

            from fastapi import UploadFile

            media = SpooledTemporaryFile(max_size=1024 * 1024)
            upload = UploadFile(file=media, filename="moderated-input.mp4")
            try:
                try:
                    async with httpx.AsyncClient(timeout=120, follow_redirects=False) as client:
                        async with client.stream("GET", TypeAdapter(str).validate_python(payload["video"])) as response:
                            response.raise_for_status()
                            async for chunk in response.aiter_bytes(65536):
                                if media.tell() + len(chunk) > 256 * 1024 * 1024:
                                    raise ValueError("Approved character input exceeds media limit")
                                await upload.write(chunk)
                    await upload.seek(0)
                except Exception as exc:
                    raise PreflightFailure("Private character input preparation failed") from exc
                return await endpoints.video_create_character(
                    prepared, Response(), upload, TypeAdapter(str).validate_python(payload["name"]), auth
                )
            finally:
                await upload.close()
        case _:
            raise ValueError("Unsupported moderation continuation route")


async def authenticate(request: Request, credential: str) -> UserAPIKeyAuth:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    return await user_api_key_auth(
        request,
        api_key="Bearer " + credential,
        azure_api_key_header=None,
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=None,
    )


def transport_outcome(error: BaseException, depth: int = 0) -> str | None:
    if isinstance(error, (PreflightFailure, httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return "not_sent"
    rejected = {400, 401, 402, 403, 404, 422, 429}
    if isinstance(error, httpx.HTTPStatusError):
        return "rejected" if error.response.status_code in rejected else "ambiguous"
    if isinstance(error, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.WriteError)):
        return "ambiguous"
    original = error.__cause__ or error.__context__
    if original is not None and original is not error and depth < 16:
        return transport_outcome(original, depth + 1)
    return None


def provider_proved_not_sent(error: BaseException) -> bool:
    """Contract §9: the attempt ledger proved every attempt was not-sent.

    Only the explicit verdict the metered entry attached to ``error`` (or a still
    open all-not-sent ledger) counts; exception chains are never consulted.
    """
    from litellm.router_utils import attempt_outcomes

    return attempt_outcomes.verdict_of(error)


def failure_outcome(error: BaseException) -> str:
    evidence = transport_outcome(error)
    if evidence is not None:
        return evidence
    if provider_proved_not_sent(error):
        return "not_sent"
    rejected = {400, 401, 402, 403, 404, 422, 429}
    if isinstance(error, HTTPException) and error.status_code in rejected:
        return "rejected"
    if isinstance(error, ProxyException) and error.code in {str(code) for code in rejected}:
        return "rejected"
    return "ambiguous"


REFUSED_BEFORE_PROCESSING = frozenset({405, 406, 410, 411, 413, 414, 415, 431, 451})


def _provider_response_status(error: BaseException, depth: int = 0) -> int | None:
    import openai

    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code
    if isinstance(error, openai.APIStatusError) and not type(error).__module__.startswith("litellm"):
        return error.status_code
    original = error.__cause__
    if original is None or original is error or depth >= 16:
        return None
    return _provider_response_status(original, depth + 1)


def _reported_status(error: BaseException, depth: int = 0) -> int | None:
    from litellm.exceptions import APIError

    status = _provider_response_status(error)
    if status is not None:
        return status
    if isinstance(error, APIError) and isinstance(error.status_code, int):
        return error.status_code
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    original = error.__cause__ or error.__context__
    if original is None or original is error or depth >= 16:
        return None
    return _reported_status(original, depth + 1)


def provider_status_proof(error: BaseException) -> int | None:
    status = _provider_response_status(error)
    return status if status in REFUSED_BEFORE_PROCESSING else None


async def clear_submission_failure(intent_id: str) -> None:
    from litellm.proxy.video_endpoints import moderation_metering_runtime as metering

    try:
        await metering.store().clear_submission_failure(intent_id)
    except Exception:  # noqa: BLE001  # a new attempt must proceed even when the store is unavailable
        logging.getLogger(__name__).warning("stale submission failure marker not cleared", exc_info=True)


async def record_submission_failure(intent_id: str, attempt: str, error: BaseException) -> None:
    proof = provider_status_proof(error)
    code = "provider_not_submitted" if proof is not None else "provider_submission_ambiguous"
    from litellm.proxy.video_endpoints import moderation_metering_runtime as metering

    try:
        await metering.store().record_submission_failure(
            intent_id, code, proof if proof is not None else _reported_status(error), "provider request failed", attempt
        )
    except Exception:  # noqa: BLE001  # the marker is best effort and must never mask the provider failure
        logging.getLogger(__name__).warning("provider submission failure marker not recorded", exc_info=True)


router = APIRouter(prefix="/internal/moderation")


@router.post("/submit", include_in_schema=False)
async def submit(body: Ticket, request: Request, authorization: Annotated[str | None, Header()] = None):
    claims = authorize(body.ticket, authorization, "submit")
    begin = await bridge.platform(request, "POST", f"/intents/{claims.intent_id}/begin", {"ticket": body.ticket})
    if begin.get("acquired") is not True:
        return {"accepted": True, "state": begin.get("state")}
    token = TypeAdapter(str).validate_python(begin["token"])
    await clear_submission_failure(claims.intent_id)
    try:
        payload = bridge.JSON_OBJECT.validate_python(begin["request"])
        route = TypeAdapter(str).validate_python(begin["route"])
        execution = execution_request(
            request,
            payload,
            {
                "context_ir": "/video/minimax-h3/v2/h3_context_ir",
                "avideo_remix": "/v1/videos/" + str(payload.get("source_video_id", "")) + "/remix",
                "avideo_edit": "/v1/videos/edits",
                "avideo_extension": "/v1/videos/extensions",
                "avideo_create_character": "/v1/videos/characters",
            }.get(route, "/v1/videos"),
        )
        execution.scope["headers"] = [
            *execution.scope["headers"],
            (b"x-litellm-call-id", ("public-video:" + claims.intent_id + ":submit").encode()),
        ]
        if route == "context_ir":
            from litellm.llms.causyn.h3_prompt import AUTH_MODEL, ContextIRRequest
            from litellm.proxy.common_utils.http_parsing_utils import _safe_set_request_parsed_body

            execution.scope["causyn_context_ir_spec"] = ContextIRRequest.model_validate(payload)
            _safe_set_request_parsed_body(execution, {"model": AUTH_MODEL})
        auth = await authenticate(execution, TypeAdapter(str).validate_python(begin["credential"]))
        from litellm.proxy.video_endpoints.moderation_metering_entry import attest

        attest(execution, begin["metering"])
        execution.scope["moderation_public_receipt"] = (claims.intent_id, token)
    except Exception as exc:
        await bridge.platform(
            request,
            "POST",
            f"/intents/{claims.intent_id}/not-sent",
            {"token": token, "outcome": "rejected" if failure_outcome(exc) == "rejected" else "not_sent"},
        )
        raise
    try:
        if route == "context_ir":
            from litellm.llms.causyn.context_ir import get_context_ir_service
            from litellm.llms.causyn.h3_prompt import ContextIRRequest
            from litellm.proxy.video_endpoints.context_ir_endpoints import create_context_ir

            execution.scope["causyn_context_ir_spec"] = ContextIRRequest.model_validate(payload)
            result = await create_context_ir(execution, auth, get_context_ir_service())
            native_id = result["task_id"]
        else:
            result = await invoke(execution, auth, payload, route)
            if not isinstance(result, (VideoObject, CharacterObject)) or not result.id:
                raise RuntimeError("Provider returned no durable video task ID")
            native_id = result.id
    except Exception as exc:
        outcome = failure_outcome(exc)
        if outcome != "ambiguous":
            await bridge.platform(
                request, "POST", f"/intents/{claims.intent_id}/not-sent", {"token": token, "outcome": outcome}
            )
        else:
            await record_submission_failure(claims.intent_id, token, exc)
        raise
    if route != "context_ir":
        await bridge.capture(execution, result)
    else:
        await bridge.platform(
            request,
            "POST",
            f"/intents/{claims.intent_id}/receipt",
            {
                "token": token,
                "native_id": native_id,
                "billing": {
                    "request_id": "causyn-context-ir:" + native_id,
                    "metering_event": execution.scope["moderation_context_ir_event"].model_dump(mode="json"),
                    "provider": "causyn",
                    "provider_task_id": native_id,
                    "status": "pending",
                },
            },
        )
    if isinstance(result, CharacterObject):
        await bridge.platform(
            request,
            "POST",
            f"/intents/{claims.intent_id}/output",
            {
                "native_id": native_id,
                "facts": {
                    "status": "completed",
                    "media_type": "character",
                    "character": bridge.JSON_OBJECT.validate_python(result.model_dump(mode="json")),
                },
            },
        )
    return {"accepted": True, "state": "submitted"}


async def completion_status_read(
    request: Request,
    auth: UserAPIKeyAuth,
    *,
    native_id: str,
    intent_id: str,
    metering: object,
    read_ticket: str | None,
    context: BillingContext | None,
) -> object:
    """One provider status read under the completion metering scope (shared by collect and settlement)."""
    from litellm.proxy.video_endpoints.endpoints import video_status
    from litellm.proxy.video_endpoints.moderation_metering_entry import attest

    execution = execution_request(request, {}, "/v1/videos/" + native_id, "GET")
    attest(execution, metering, phase="completion")
    if read_ticket is not None:
        execution.scope["moderation_financial_read_ticket"] = read_ticket
    execution.scope["headers"] = [
        *execution.scope["headers"],
        # Never the phase request id itself: that id belongs to the phase's financial projection row, and a
        # regular spend-log row written for a status poll (even a failed one) would collide with it forever.
        (b"x-litellm-call-id", ("public-video:" + intent_id + ":completion:poll:" + uuid4().hex).encode()),
    ]
    context_token = BILLING_CONTEXT.set(context)
    try:
        return await video_status(native_id, execution, Response(), auth)
    finally:
        BILLING_CONTEXT.reset(context_token)


def _is_provider_task_not_found(exc: BaseException) -> bool:
    """True only for the typed adapter answer "this task id does not exist upstream", anywhere in the chain."""
    from litellm.llms.custom_llm import ProviderTaskNotFound

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, ProviderTaskNotFound):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


DEAD_UPSTREAM_TTL_MARGIN_S = 3600


def _dead_upstream_min_age_s() -> float:
    try:
        configured = float(os.environ.get("LITELLM_DEAD_UPSTREAM_MIN_AGE_S", "21600"))
    except ValueError:
        configured = 21600.0
    from litellm.llms.libtv.transfer import STATUS_TTL_SECONDS

    # A missing status key only proves "gone" once the key would have expired on its own (Redis flush,
    # failover or a wrong URL also make it vanish), so never trust it before TTL + margin.
    return max(configured, STATUS_TTL_SECONDS + DEAD_UPSTREAM_TTL_MARGIN_S)


async def _close_dead_upstream(store: MeteringStore, intent_id: str, native_id: str, exc: BaseException) -> str | None:
    """Close the completion as undelivered when the provider authoritatively lost the task.

    Returns "closed", "finalized" (completion already settled, nothing to do) or None (not a dead upstream:
    caller re-raises). Requires: typed not-found, submit settled with this exact native id and at least the
    minimum age, so a just-created task is never judged gone.
    """
    if not _is_provider_task_not_found(exc):
        return None
    submit = await store.phase(intent_id, "submit")
    if submit is None or not submit.finalized or not submit.native_id or submit.native_id != native_id:
        return None
    facts = submit.facts
    since = None if facts is None else (facts.ended_at or facts.started_at)
    if since is None:
        return None
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - since).total_seconds() < _dead_upstream_min_age_s():
        return None
    completion = await store.phase(intent_id, "completion")
    if completion is not None and completion.finalized:
        return "finalized"
    binding = await store.binding(intent_id)
    if "completion" not in binding.expected_phases:
        return None
    await _persist_undelivered(store, binding, submit)
    logging.getLogger(__name__).error(
        "dead upstream: provider no longer knows task; completion closed undelivered at amount 0 "
        "intent_id=%s native_id=%s age_s=%.0f",
        intent_id,
        native_id,
        (datetime.now(timezone.utc) - since).total_seconds(),
    )
    return "closed"


@router.post("/collect", include_in_schema=False)
async def collect(body: Ticket, request: Request, authorization: Annotated[str | None, Header()] = None):
    claims = authorize(body.ticket, authorization, "collect")
    task = await bridge.platform(request, "POST", f"/intents/{claims.intent_id}/execution", {"ticket": body.ticket})
    if task.get("state") != "submitted":
        return {"accepted": True}
    native_id = TypeAdapter(str).validate_python(task["native_id"])
    principal = bridge.JSON_OBJECT.validate_python(task["principal"])
    auth = UserAPIKeyAuth(
        api_key=TypeAdapter(str).validate_python(principal["fingerprint"]),
        user_id=TypeAdapter(str).validate_python(principal["user_id"]),
        team_id=TypeAdapter(str).validate_python(principal["team_id"]),
    )
    if task["route"] == "context_ir":
        from litellm.llms.causyn.context_ir import get_context_ir_service

        ir = await get_context_ir_service().store.get(native_id)
        if ir is None or ir.status not in {"succeeded", "failed", "cancelled"}:
            return {"accepted": False}
        from litellm.llms.causyn.context_ir import financial_event

        phase = financial_event(ir)
        if phase is not None:
            await bridge.platform(
                request,
                "POST",
                f"/intents/{claims.intent_id}/financial-event",
                {"ticket": body.ticket, "native_id": native_id, "metering_event": phase.model_dump(mode="json")},
            )
        facts = {"status": ir.status, "content": ir.public().get("content"), "media_type": "text"}
    elif task["route"] == "avideo_create_character":
        from litellm.proxy.video_endpoints.endpoints import video_get_character

        execution = execution_request(request, {}, "/v1/videos/characters/" + native_id, "GET")
        character = await video_get_character(native_id, execution, Response(), auth)
        facts = {
            "status": "completed",
            "media_type": "character",
            "character": bridge.JSON_OBJECT.validate_python(
                CharacterObject.model_validate(character).model_dump(mode="json")
            ),
        }
    else:
        try:
            result = await completion_status_read(
                request,
                auth,
                native_id=native_id,
                intent_id=claims.intent_id,
                metering=task["metering"],
                read_ticket=body.ticket,
                context=BillingContext(
                    intent_id=claims.intent_id,
                    model=claims.model,
                    principal=principal,
                    billing=bridge.JSON_OBJECT.validate_python(task["billing"]),
                ),
            )
        except Exception as exc:
            from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime

            closed = await _close_dead_upstream(runtime.store(), claims.intent_id, native_id, exc)
            if closed is None:
                raise
            if closed == "finalized":
                return {"accepted": False}
            # Same shape as a provider-reported `failed` status, so the app terminalizes the task.
            result = VideoObject(id=native_id, object="video", status="failed")
        if not isinstance(result, VideoObject) or result.status not in {"completed", "failed", "cancelled"}:
            return {"accepted": False}
        stored = bridge.JSON_OBJECT.validate_python(result.object_store_result or {})
        facts = {
            "status": result.status,
            "url": result._hidden_params.get("url"),
            "staging_key": stored.get("staging_key"),
            "media_type": "video",
            **({"error": bridge.JSON_OBJECT.validate_python(result.error)} if result.error else {}),
        }
        if result.status == "completed" and not facts["url"] and not facts["staging_key"]:
            from litellm.proxy.video_endpoints.moderation_content import materialize_content

            facts = {
                **facts,
                **await materialize_content(
                    request,
                    auth,
                    native_id,
                    intent_id=claims.intent_id,
                    ticket=TypeAdapter(str).validate_python(task["upload_ticket"]),
                ),
            }
    await bridge.platform(
        request,
        "POST",
        f"/intents/{claims.intent_id}/output",
        {
            "native_id": native_id,
            "facts": bridge.JSON_OBJECT.validate_python(facts),
        },
    )
    return {"accepted": True}


@router.post("/cancel", include_in_schema=False)
async def cancel(body: Ticket, request: Request, authorization: Annotated[str | None, Header()] = None):
    claims = authorize(body.ticket, authorization, "cancel")
    task = await bridge.platform(request, "POST", f"/intents/{claims.intent_id}/cancellation", {"ticket": body.ticket})
    if task.get("state") == "cancelled":
        return {"action": "cancelled"}
    if task.get("state") != "cancel_requested":
        raise HTTPException(409, "Cancellation has not been requested")
    from litellm.llms.causyn.context_ir import get_context_ir_service
    from litellm.proxy.video_endpoints.minimax_h3_endpoints import task_owner

    principal = bridge.JSON_OBJECT.validate_python(task["principal"])
    auth = UserAPIKeyAuth(api_key=TypeAdapter(str).validate_python(principal["fingerprint"]))
    native_id = TypeAdapter(str).validate_python(task["native_id"])
    action = await get_context_ir_service().cancel_or_delete(native_id, task_owner(auth))
    await bridge.platform(
        request,
        "POST",
        f"/intents/{claims.intent_id}/cancellation-receipt",
        {
            "native_id": native_id,
            "facts": {"action": action},
        },
    )
    return {"action": action}


@router.post("/settlement", include_in_schema=False)
async def settlement(body: SettlementTicket, request: Request, authorization: Annotated[str | None, Header()] = None):
    from litellm.proxy.video_endpoints import moderation_metering_runtime as metering
    from litellm.proxy.video_endpoints.moderation_metering import AdmissionMissing, BillingBinding, PhaseEvent

    claims = authorize(body.ticket, authorization, "settlement")
    authority = await bridge.platform(
        request, "POST", f"/intents/{claims.intent_id}/settlement-authority", {"ticket": body.ticket}
    )
    store = metering.store()
    try:
        binding = await store.binding(claims.intent_id)
    except AdmissionMissing as error:
        return JSONResponse(status_code=404, content={"error": {"code": "admission_missing", "message": str(error)}})
    if binding.intent_id != claims.intent_id or binding.request_digest != claims.request_digest:
        raise HTTPException(403, "Settlement intent identity mismatch")
    if authority.get("binding") is not None and BillingBinding.model_validate(authority["binding"]) != binding:
        raise HTTPException(403, "Settlement actor identity mismatch")
    proof = bridge.JSON_OBJECT.validate_python(authority.get("recovery_binding") or {})
    if proof and any(binding.model_dump(mode="json").get(key) != value for key, value in proof.items()):
        raise HTTPException(403, "Settlement recovery admission mismatch")
    if not proof and authority.get("binding") is None:
        raise HTTPException(403, "Settlement admission required")
    for value in bridge.JSON_OBJECT.validate_python(authority.get("events") or {}).values():
        event = PhaseEvent.model_validate(value)
        if event.binding != binding or event.native_id != authority["native_id"]:
            raise HTTPException(403, "Settlement provider binding mismatch")
        await store.persist(event)
    manual = await store.manual_settlement_reason(binding.intent_id)
    if manual is not None:
        return JSONResponse(status_code=409, content={"error": {"code": "key_identity_mismatch", "message": manual}})
    if body.close_completion is not None:
        await close_completion(request, store, binding, body.close_completion)
    events: list[PhaseEvent] = []
    for phase in binding.expected_phases:
        event = await store.phase(binding.intent_id, phase)
        if event is not None and event.native_id:
            events.append(event)
    failure = None if events else await store.submission_failure(binding.intent_id)
    if failure is not None:
        if failure[0] == "provider_not_submitted":
            try:
                await store.void_unsent(binding.intent_id)
            except Exception:  # noqa: BLE001  # voiding is best effort; the 409 answer stands regardless
                logging.getLogger(__name__).warning("unsent metering placeholders not voided", exc_info=True)
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": failure[0],
                    "provider_status": failure[1],
                    "message": failure[2],
                    "attempt": failure[3],
                }
            },
        )
    return (await store.settlement(binding)).model_copy(update={"events": tuple(events)}).model_dump(mode="json")


async def _persist_undelivered(store: MeteringStore, binding: BillingBinding, submit: PhaseEvent) -> None:
    """Persist a finalized amount-0 completion under the submit identity (idempotent, deterministic payload)."""
    from litellm.proxy.video_endpoints.moderation_metering import PhaseEvent, request_id
    from litellm.proxy.video_endpoints.moderation_metering_projection import BillingFacts, actual_debit

    # BillingFacts forbids extra keys, so the audit marker is logged rather than stored.
    logging.getLogger(__name__).info("closed_by=app_undelivered intent_id=%s", binding.intent_id)
    current = await store.phase(binding.intent_id, "completion")
    if current is not None and current.native_id and not current.finalized:
        # A running poll already persisted this row. persist() only accepts amount, finalized and the
        # raw_cost_credit fill on top of it, so derive the close from the stored row (deterministic).
        facts = current.facts
        if facts is not None and facts.raw_cost_credit is None:
            facts = facts.model_copy(update={"raw_cost_credit": Decimal(0)})
        if facts is None:
            stamp = submit.facts.ended_at if submit.facts is not None else datetime.fromtimestamp(0, timezone.utc)
            facts = BillingFacts(started_at=stamp, ended_at=stamp, route="avideo_status", raw_cost_credit=Decimal(0))
        event = current.model_copy(update={"amount": actual_debit(Decimal(0)), "finalized": True, "facts": facts})
    else:
        # Deterministic: duplicate or concurrent closes must yield byte-identical payloads.
        stamp = submit.facts.ended_at if submit.facts is not None else datetime.fromtimestamp(0, timezone.utc)
        started = submit.facts.started_at if submit.facts is not None else stamp
        raw = Decimal(0)
        event = PhaseEvent(
            binding=binding,
            request_id=request_id(binding, "completion"),
            phase="completion",
            provider=submit.provider,
            deployment_id=submit.deployment_id,
            native_id=submit.native_id,
            provider_task_id=submit.provider_task_id,
            amount=actual_debit(raw),
            finalized=True,
            facts=BillingFacts(started_at=started, ended_at=stamp, route="avideo_status", raw_cost_credit=raw),
        )
    try:
        await store.persist(event)
    except ValueError:
        current = await store.phase(binding.intent_id, "completion")
        if current is None or not current.finalized:
            raise


async def close_completion(
    request: Request,
    store: MeteringStore,
    binding: BillingBinding,
    mode: Literal["undelivered", "delivered"],
) -> None:
    """Contract section 8: close a completion phase nobody polls. Ignored unless provably closable."""
    submit = await store.phase(binding.intent_id, "submit")
    completion = await store.phase(binding.intent_id, "completion")
    if (
        "completion" not in binding.expected_phases
        or "submit" not in binding.expected_phases
        or submit is None
        or not submit.native_id
        or (completion is not None and completion.finalized)
    ):
        return
    if mode == "undelivered":
        await _persist_undelivered(store, binding, submit)
        return
    if binding.actor_user_id is None:
        return
    auth = UserAPIKeyAuth(api_key=binding.fingerprint, user_id=binding.user_id, team_id=binding.team_id)
    try:
        await completion_status_read(
            request,
            auth,
            native_id=submit.native_id,
            intent_id=binding.intent_id,
            metering={
                "intent_id": binding.intent_id,
                "request_digest": binding.request_digest,
                "actor_user_id": binding.actor_user_id,
                "model": binding.model,
                "generation_id": binding.generation_id,
            },
            read_ticket=None,
            context=None,
        )
    except Exception as exc:
        if await _close_dead_upstream(store, binding.intent_id, submit.native_id, exc) is not None:
            return
        logging.getLogger(__name__).warning(
            "completion close status read failed intent_id=%s", binding.intent_id, exc_info=True
        )
