from __future__ import annotations

import hmac
import json
import os
from contextvars import ContextVar
from typing import Annotated, Literal

import httpx
import jwt
from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.video_endpoints import moderation_bridge as bridge
from litellm.types.videos.main import CharacterObject, VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ticket: str


class Claims(BaseModel):
    intent_id: str
    request_digest: str
    input_digest: str
    model: str
    policy_version: str
    policy_digest: str
    purpose: Literal["submit", "collect", "cancel"]


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


def failure_outcome(error: BaseException) -> str:
    evidence = transport_outcome(error)
    if evidence is not None:
        return evidence
    rejected = {400, 401, 402, 403, 404, 422, 429}
    if isinstance(error, HTTPException) and error.status_code in rejected:
        return "rejected"
    if isinstance(error, ProxyException) and error.code in {str(code) for code in rejected}:
        return "rejected"
    return "ambiguous"


router = APIRouter(prefix="/internal/moderation")


@router.post("/submit", include_in_schema=False)
async def submit(body: Ticket, request: Request, authorization: Annotated[str | None, Header()] = None):
    claims = authorize(body.ticket, authorization, "submit")
    begin = await bridge.platform(request, "POST", f"/intents/{claims.intent_id}/begin", {"ticket": body.ticket})
    if begin.get("acquired") is not True:
        return {"accepted": True, "state": begin.get("state")}
    token = TypeAdapter(str).validate_python(begin["token"])
    try:
        payload = bridge.JSON_OBJECT.validate_python(begin["request"])
        route = TypeAdapter(str).validate_python(begin["route"])
        execution = execution_request(
            request,
            payload,
            {
                "context_ir": "/v2/h3_context_ir",
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
        raise
    decoded = decode_video_id_with_provider(native_id)
    provider = decoded.get("custom_llm_provider") or ("context_ir" if route == "context_ir" else "unknown")
    provider_id = decoded.get("video_id") or native_id
    request_id = (
        "causyn-context-ir:" + native_id
        if route == "context_ir"
        else "causyn:" + provider_id
        if provider == "causyn"
        else "public-video:" + claims.intent_id
    )
    await bridge.platform(
        request,
        "POST",
        f"/intents/{claims.intent_id}/receipt",
        {
            "token": token,
            "native_id": native_id,
            "billing": {
                "request_id": request_id,
                "provider": provider,
                "provider_task_id": provider_id,
                "deployment_id": decoded.get("model_id"),
                "status": "pending",
                "request_ids": [
                    request_id,
                    "public-video:" + claims.intent_id + ":submit",
                    "public-video:" + claims.intent_id + ":completion",
                ],
                "pricing_snapshot": bridge.JSON_OBJECT.validate_python(
                    result._hidden_params.get("billing_pricing_snapshot") or {}
                )
                if isinstance(result, VideoObject)
                else {},
                "usage_snapshot": bridge.JSON_OBJECT.validate_python(result.usage or {})
                if isinstance(result, VideoObject)
                else {},
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
        from litellm.proxy.video_endpoints.endpoints import video_status

        execution = execution_request(request, {}, "/v1/videos/" + native_id, "GET")
        execution.scope["headers"] = [
            *execution.scope["headers"],
            (b"x-litellm-call-id", ("public-video:" + claims.intent_id + ":completion").encode()),
        ]
        context_token = BILLING_CONTEXT.set(
            BillingContext(
                intent_id=claims.intent_id,
                model=claims.model,
                principal=principal,
                billing=bridge.JSON_OBJECT.validate_python(task["billing"]),
            )
        )
        try:
            result = await video_status(native_id, execution, Response(), auth)
        finally:
            BILLING_CONTEXT.reset(context_token)
        if not isinstance(result, VideoObject) or result.status not in {"completed", "failed", "cancelled"}:
            return {"accepted": False}
        stored = bridge.JSON_OBJECT.validate_python(result.object_store_result or {})
        facts = {
            "status": result.status,
            "url": result._hidden_params.get("url"),
            "staging_key": stored.get("staging_key"),
            "media_type": "video",
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
