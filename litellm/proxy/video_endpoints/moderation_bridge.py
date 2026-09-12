from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import uuid

import httpx
from fastapi import HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from starlette.datastructures import UploadFile as StarletteUploadFile

from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.videos.main import CharacterObject, VideoObject

PREFIX = "mod_video_"
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
ADMITTED = object()
INTAKE_ROUTES = frozenset(
    {
        "/videos",
        "/v1/videos",
        "/v2/video_generation",
        "/v2/h3_context_ir",
        "/videos/{video_id}/remix",
        "/v1/videos/{video_id}/remix",
        "/videos/edits",
        "/v1/videos/edits",
        "/videos/extensions",
        "/v1/videos/extensions",
        "/videos/characters",
        "/v1/videos/characters",
    }
)


class View(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    state: str
    model: str
    moderation_status: str
    policy_source: str
    created_at: int
    output: dict[str, JsonValue] | None = None
    parameters: dict[str, JsonValue] = {}


def configured(auth: UserAPIKeyAuth) -> bool:
    metadata = JSON_OBJECT.validate_python(auth.metadata or {})
    return bool(
        os.getenv("DRAMA_MODERATION_PLATFORM_URL") or metadata.get("openapi_key_id") or metadata.get("project_id")
    )


def defer_budget(request: Request, auth: UserAPIKeyAuth, route: str) -> bool:
    return (
        request.method == "POST"
        and route in INTAKE_ROUTES
        and configured(auth)
        and request.scope.get("moderation_admission") is not ADMITTED
    )


def principal(auth: UserAPIKeyAuth) -> dict[str, JsonValue]:
    identity = auth.api_key or auth.token
    metadata = JSON_OBJECT.validate_python(auth.metadata or {})
    if not identity or not auth.user_id or not auth.team_id:
        raise HTTPException(403, "No verified platform credential mapping")
    if not metadata.get("openapi_key_id") and not metadata.get("project_id"):
        raise HTTPException(403, "No verified platform credential mapping")
    return {
        "user_id": auth.user_id,
        "team_id": auth.team_id,
        "fingerprint": identity
        if re.fullmatch(r"[a-f0-9]{64}", identity)
        else hashlib.sha256(identity.encode()).hexdigest(),
        "openapi_key_id": metadata.get("openapi_key_id"),
        "project_id": metadata.get("project_id"),
    }


async def platform(request: Request, method: str, path: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    url = os.getenv("DRAMA_MODERATION_PLATFORM_URL")
    secret = os.getenv("DRAMA_MODERATION_SERVICE_TOKEN")
    if not url or not secret:
        raise HTTPException(503, "Moderation control service is not configured")
    transport = getattr(request.app.state, "moderation_transport", None)
    async with httpx.AsyncClient(base_url=url, transport=transport, timeout=15, follow_redirects=False) as client:
        response = await client.request(
            method, "/internal/moderation" + path, json=payload, headers={"Authorization": "Bearer " + secret}
        )
    if response.is_error:
        raise HTTPException(
            response.status_code if response.status_code < 500 else 503, "Moderation control request was rejected"
        )
    return JSON_OBJECT.validate_python(response.json())


def video(view: View) -> VideoObject:
    output = view.output or {}
    result = VideoObject(
        id=view.id,
        object="video",
        model=view.model,
        created_at=view.created_at,
        status="cancelled"
        if view.state == "cancelled"
        else "completed"
        if view.state == "completed"
        else "failed"
        if view.state in {"failed", "moderation_rejected"}
        else "in_progress"
        if view.state in {"submitted", "output_moderation"}
        else "queued",
        moderation_status=view.moderation_status,
        generation_status=view.state,
    )
    result._hidden_params = {"moderation_public_parameters": view.parameters}
    if view.state == "completed" and isinstance(output.get("url"), str):
        result._hidden_params = {**result._hidden_params, "url": output["url"]}
    if view.state == "completed" and isinstance(output.get("content"), dict):
        result._hidden_params = {**result._hidden_params, "content": output["content"]}
    return result


def character(result: VideoObject) -> CharacterObject:
    params = JSON_OBJECT.validate_python(result._hidden_params.get("moderation_public_parameters") or {})
    return CharacterObject(
        id=result.id,
        created_at=result.created_at,
        name=str(params.get("name") or ""),
        status=result.status,
        moderation_status=result.moderation_status,
    )


def v2_task(result: VideoObject) -> dict[str, JsonValue]:
    params = JSON_OBJECT.validate_python(result._hidden_params.get("moderation_public_parameters") or {})
    public = {
        "id": result.id,
        "model": {"hailuo-h3": "MiniMax-H3", "hailuo-h3-max": "MiniMax-H3-Max"}.get(result.model or "", result.model),
        "status": {"completed": "succeeded", "in_progress": "running"}.get(result.status, result.status),
        "created_at": result.created_at,
        "moderation_status": result.moderation_status,
        "generation_status": result.generation_status,
        "modality": "video",
        "task_type": "generation",
        **params,
    }
    return JSON_OBJECT.validate_python(
        {
            **public,
            **(
                {"content": result._hidden_params.get("content") or {"url": result._hidden_params.get("url")}}
                if result.status == "completed"
                else {}
            ),
        }
    )


async def list_tasks(request: Request, auth: UserAPIKeyAuth, *, v2: bool = False) -> dict[str, JsonValue] | None:
    if not configured(auth) or request.scope.get("moderation_admission") is ADMITTED:
        return None
    if v2:
        from litellm.proxy.video_endpoints.context_ir_endpoints import ListParams

        params = ListParams.model_validate(
            {
                key: request.query_params.getlist(key) if key == "filter.task_ids" else value
                for key, value in request.query_params.items()
            }
        )
        filters = {
            "limit": params.page_size,
            "offset": (params.page_num - 1) * params.page_size,
            "status": params.status,
            "model": params.model,
            "task_type": params.task_type,
            "task_ids": list(params.task_ids),
            "surface": "v2",
        }
    else:
        filters = {
            "limit": int(request.query_params.get("limit", "20")),
            "after": request.query_params.get("after"),
            "order": request.query_params.get("order", "desc"),
            "surface": "v1",
        }
    result = await platform(
        request, "POST", "/intents/list", {"principal": principal(auth), **JSON_OBJECT.validate_python(filters)}
    )
    views = TypeAdapter(list[View]).validate_python(result["items"])
    for view in views:
        await model_access(auth, view.model)
    items = [
        v2_task(video(view)) if v2 else JSON_OBJECT.validate_python(video(view).model_dump(mode="json"))
        for view in views
    ]
    return (
        {"items": items, "total": result.get("total", len(items))}
        if v2
        else {"object": "list", "data": items, "has_more": result.get("has_more", False)}
    )


async def submit(
    request: Request,
    auth: UserAPIKeyAuth,
    data: object,
    route: str,
    upload: UploadFile | None = None,
    *,
    policy_model: str | None = None,
) -> VideoObject | None:
    if not configured(auth) or request.scope.get("moderation_admission") is ADMITTED:
        return None
    raw = TypeAdapter(dict[str, object]).validate_python(data)
    payload = JSON_OBJECT.validate_python(
        {key: value for key, value in raw.items() if not isinstance(value, StarletteUploadFile)}
    )
    owner = principal(auth)
    ticket = request.headers.get("x-drama-moderation-admission")
    if ticket:
        admission = await platform(
            request,
            "POST",
            "/generation-admission",
            {
                "ticket": ticket,
                "payload": payload,
                "fingerprint": owner["fingerprint"],
            },
        )
        if admission.get("acquired") is not True:
            return VideoObject(
                id=TypeAdapter(str).validate_python(admission["native_id"]), object="video", status="queued"
            )
        request.scope["moderation_admission"] = ADMITTED
        request.scope["moderation_producer_ticket"] = ticket
        from litellm.proxy.auth.user_api_key_auth import _run_centralized_common_checks

        await _run_centralized_common_checks(auth, request, payload, "/v1/videos")
        return None
    if (
        any(
            request.headers.get(name)
            for name in (
                "x-litellm-model",
                "custom-llm-provider",
                "x-litellm-custom-llm-provider",
                "x-litellm-api-base",
            )
        )
        or request.query_params
    ):
        raise HTTPException(422, "Provider routing overrides are not supported")
    model = policy_model or await resolved_model(auth, payload.get("model"))
    if model and not policy_model:
        payload = {**payload, "model": model}
    source = await source_descriptor(request, auth, payload)
    normalized = await inline_media(request, owner, payload)
    prepared = {
        **JSON_OBJECT.validate_python(normalized),
        **(
            {
                "video" if route == "avideo_create_character" else "input_reference": await upload_media(
                    request, owner, upload
                )
            }
            if upload is not None
            else {}
        ),
    }
    result = await platform(
        request,
        "POST",
        "/intents",
        {
            "principal": owner,
            "model": model or "",
            "route": route,
            "payload": prepared,
            "source": source,
            "idempotency_key": request.headers.get("idempotency-key") or str(uuid.uuid4()),
        },
    )
    projected = video(View.model_validate(result))
    from litellm.proxy.video_endpoints import openapi_log_capture

    history_id = await openapi_log_capture.start(request, auth, prepared)
    await openapi_log_capture.submitted(history_id, projected)
    return projected


async def capture(request: Request, result: object) -> None:
    ticket = request.scope.get("moderation_producer_ticket")
    if isinstance(ticket, str) and isinstance(result, VideoObject):
        await platform(request, "POST", "/generation-receipt", {"ticket": ticket, "native_id": result.id})


async def upload_media(request: Request, owner: dict[str, JsonValue], upload: StarletteUploadFile) -> str:
    digest = hashlib.sha256()
    size = 0
    await upload.seek(0)
    while chunk := await upload.read(65536):
        size += len(chunk)
        if size > 256 * 1024 * 1024:
            raise HTTPException(413, "Media exceeds 256 MiB")
        digest.update(chunk)
    target = await platform(
        request, "POST", "/uploads", {"principal": owner, "digest": digest.hexdigest(), "size": size}
    )
    await upload.seek(0)

    async def chunks():
        while data := await upload.read(65536):
            yield data

    async with httpx.AsyncClient(timeout=120, follow_redirects=False) as client:
        response = await client.put(
            TypeAdapter(str).validate_python(target["url"]),
            content=chunks(),
            headers=TypeAdapter(dict[str, str]).validate_python(target["headers"]),
        )
        if response.status_code != 409:
            response.raise_for_status()
    return TypeAdapter(str).validate_python(target["reference"])


async def inline_media(request: Request, owner: dict[str, JsonValue], value: JsonValue) -> JsonValue:
    if isinstance(value, str) and value.startswith("data:"):
        if len(value) > 360 * 1024 * 1024 or ";base64," not in value:
            raise HTTPException(413, "Invalid or oversized inline media")
        try:
            raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        except ValueError:
            raise HTTPException(422, "Invalid inline media") from None
        upload = StarletteUploadFile(file=io.BytesIO(raw))
        try:
            return await upload_media(request, owner, upload)
        finally:
            await upload.close()
    if isinstance(value, dict):
        return {key: await inline_media(request, owner, child) for key, child in value.items()}
    if isinstance(value, list):
        return [await inline_media(request, owner, child) for child in value]
    return value


async def query(request: Request, auth: UserAPIKeyAuth, task_id: str, *, purpose: str = "query") -> VideoObject | None:
    if not configured(auth) or request.scope.get("moderation_admission") is ADMITTED:
        return None
    read_ticket = request.headers.get("x-drama-moderation-read")
    if read_ticket:
        await platform(
            request,
            "POST",
            "/generation-read",
            {
                "ticket": read_ticket,
                "native_id": task_id,
                "fingerprint": principal(auth)["fingerprint"],
                "purpose": purpose,
            },
        )
        request.scope["moderation_admission"] = ADMITTED
        return None
    if not task_id.startswith(PREFIX):
        lookup = await platform(request, "POST", "/lookup", {"principal": principal(auth), "native_id": task_id})
        if lookup.get("bound") is True:
            if not isinstance(lookup.get("view"), dict):
                raise HTTPException(404, "Private generation requires bound service authorization")
            view = View.model_validate(lookup["view"])
            await model_access(auth, view.model)
            return video(view)
        if await legacy_owner(request, auth, task_id):
            return None
        raise HTTPException(404, "Task has no verified owner registration")
    result = await platform(request, "POST", f"/intents/{task_id}/view", {"principal": principal(auth)})
    view = View.model_validate(result)
    await model_access(auth, view.model)
    projected = video(view)
    from litellm.proxy.video_endpoints import openapi_log_capture

    await openapi_log_capture.observe(auth, projected)
    return projected


async def legacy_owner(request: Request, auth: UserAPIKeyAuth, task_id: str) -> bool:
    import hmac

    from litellm.proxy.video_endpoints.minimax_h3_endpoints import task_owner
    from litellm.proxy.video_endpoints.minimax_h3_models import MiniMaxTask
    from litellm.proxy.video_endpoints.openapi_log_capture import database

    decoded = request.scope.get("minimax_h3_task")
    if (
        isinstance(decoded, MiniMaxTask)
        and decoded.native_id == task_id
        and hmac.compare_digest(decoded.owner, task_owner(auth))
    ):
        return True
    db = database()
    if db is None:
        return False
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
        await db.query_raw(
            'SELECT id FROM "LiteLLM_OpenApiLog" WHERE owner=$1 AND task_id=$2 LIMIT 1',
            principal(auth)["fingerprint"],
            task_id,
        )
    )
    return bool(rows)


async def download(request: Request, auth: UserAPIKeyAuth, task_id: str):
    from fastapi.responses import RedirectResponse

    result = await query(request, auth, task_id, purpose="download")
    if result is None:
        return None
    if result.status != "completed":
        raise HTTPException(409, "Video is not approved for download")
    url = result._hidden_params.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise HTTPException(503, "Approved media URL is unavailable")
    return RedirectResponse(url, status_code=302)


async def source_descriptor(
    request: Request, auth: UserAPIKeyAuth, payload: dict[str, JsonValue]
) -> dict[str, JsonValue] | None:
    import hmac

    from fastapi import Response

    from litellm.proxy.video_endpoints.endpoints import video_status
    from litellm.proxy.video_endpoints.minimax_h3_endpoints import task_owner
    from litellm.proxy.video_endpoints.minimax_h3_models import TASK_PREFIX, decode_task
    from litellm.proxy.video_endpoints.moderation_execution import execution_request
    from litellm.proxy.video_endpoints.openapi_log_capture import database

    reference = payload.get("video")
    requested = (
        payload.get("source_video_id")
        or payload.get("video_id")
        or (reference.get("id") if isinstance(reference, dict) else None)
    )
    if not isinstance(requested, str) or requested.startswith(PREFIX):
        return None
    if requested.startswith(TASK_PREFIX):
        try:
            h3 = decode_task(requested)
        except ValueError:
            raise HTTPException(404, "Source task not found") from None
        if not hmac.compare_digest(h3.owner, task_owner(auth)):
            raise HTTPException(404, "Source task not found")
        source_id, model = h3.native_id, h3.model
    else:
        source_id, model = requested, None
    bound = await platform(request, "POST", "/lookup", {"principal": principal(auth), "native_id": source_id})
    if bound.get("bound") is True:
        if not isinstance(bound.get("view"), dict):
            raise HTTPException(404, "Private generation cannot be used as a public source")
        view = View.model_validate(bound["view"])
        return {"public_id": view.id}
    if model is None:
        if not await legacy_owner(request, auth, source_id):
            raise HTTPException(404, "Source task has no verified owner registration")
        db = database()
        if db is not None:
            rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
                await db.query_raw(
                    'SELECT model FROM "LiteLLM_OpenApiLog" WHERE owner=$1 AND task_id=$2 LIMIT 1',
                    principal(auth)["fingerprint"],
                    source_id,
                )
            )
            model = TypeAdapter(str | None).validate_python(rows[0].get("model")) if rows else None
    execution = execution_request(request, {}, "/v1/videos/" + source_id, "GET")
    result = await video_status(source_id, execution, Response(), auth)
    if not isinstance(result, VideoObject) or result.status != "completed":
        raise HTTPException(409, "Source video is not complete")
    url = result._hidden_params.get("url")
    if not isinstance(url, str):
        from litellm.proxy.video_endpoints.moderation_content import materialize_content

        materialized = await materialize_content(request, auth, source_id)
        url = materialized["private_reference"]
    return {"requested_id": requested, "native_id": source_id, "model": model or result.model, "url": url}


async def model_access(auth: UserAPIKeyAuth, model: str) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.auth_checks import can_key_call_resolved_model

    await can_key_call_resolved_model(model, None, auth, proxy_server.llm_router)


async def resolved_model(auth: UserAPIKeyAuth, value: JsonValue) -> str | None:
    if value is None:
        return None
    from litellm.proxy.auth.auth_utils import _resolve_managed_resource_alias_chain

    original = TypeAdapter(str).validate_python(value)
    if any(part.startswith("_fallback") for part in original.split("/")):
        raise HTTPException(422, "Internal generation model is not public")
    resolved = TypeAdapter(str).validate_python(
        _resolve_managed_resource_alias_chain(
            original, team_model_aliases=auth.team_model_aliases, key_aliases=auth.aliases
        )
    )
    if any(
        part.startswith("_fallback") for part in resolved.split("/")
    ) or resolved != _resolve_managed_resource_alias_chain(
        resolved, team_model_aliases=auth.team_model_aliases, key_aliases=auth.aliases
    ):
        raise HTTPException(422, "Unstable or internal model alias is not public")
    await model_access(auth, resolved)
    return resolved
