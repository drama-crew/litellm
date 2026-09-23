from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.causyn.context_ir import ContextIRService, get_context_ir_service
from litellm.llms.causyn.h3_prompt import AUTH_MODEL, ContextIRRequest, RewriteError
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _safe_set_request_parsed_body
from litellm.proxy.video_endpoints import endpoints, moderation_bridge
from litellm.proxy.video_endpoints.minimax_h3_models import (
    MiniMaxH3Create,
)
from litellm.proxy.video_endpoints.minimax_h3_paths import DIRECT_PREFIX, INTERNAL_MODEL, IR_PREFIX, namespace
from litellm.types.videos.main import VideoObject

BODY_LIMIT = 64 * 1024 * 1024
ERROR_TYPES = {
    400: "bad_request_error",
    401: "authorized_error",
    402: "insufficient_balance_error",
    403: "authorized_error",
    404: "not_found_error",
    422: "unprocessable_entity_error",
    429: "rate_limit_error",
    500: "server_error",
}
STATUS_NAMES = {
    "queued": "queued",
    "in_progress": "running",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


class H3Error(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(message)


def error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "type": "error",
            "error": {"type": ERROR_TYPES.get(code, "server_error"), "message": message, "http_code": str(code)},
            "request_id": uuid.uuid4().hex,
        },
    )


def task_owner(auth: UserAPIKeyAuth) -> str:
    identity = auth.api_key or auth.token
    if not identity:
        raise H3Error(401, "API key authentication is required")
    return hashlib.sha256(("minimax-h3-owner\0" + identity).encode()).hexdigest()


async def read_json_body(request: Request) -> bytearray:
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > BODY_LIMIT:
            raise H3Error(400, "Request body exceeds 64 MB")
        raw.extend(chunk)
    return raw


async def prepare_request(request: Request) -> None:
    is_ir_create = request.url.path == IR_PREFIX + "/v2/h3_context_ir"
    is_ir_list = request.url.path in {
        IR_PREFIX + "/v2/query/video_generation",
        DIRECT_PREFIX + "/v2/query/video_generation",
    }
    public_id = TypeAdapter(str).validate_python(request.path_params.get("video_id", ""))
    is_ir = is_ir_create
    request.scope["h3_namespace"] = namespace(request.url.path)
    request.scope["causyn_context_ir"] = is_ir
    if (request.query_params and not is_ir_list) or any(
        request.headers.get(name)
        for name in (
            "x-litellm-model",
            "custom-llm-provider",
            "x-litellm-custom-llm-provider",
            "x-drama-moderation-admission",
            "x-drama-moderation-read",
        )
    ):
        raise H3Error(400, "Provider routing overrides are not supported on this endpoint")
    if request.method == "POST":
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise H3Error(400, "Content-Type must be application/json")
        raw = await read_json_body(request)
        if is_ir_create:
            public_ir = json.loads(raw)
            if not isinstance(public_ir, dict) or public_ir.get("model") != "minimax-h3":
                raise H3Error(400, "model must be minimax-h3")
            spec_ir = ContextIRRequest.model_validate({**public_ir, "model": "MiniMax-H3"})
            spec_ir.require_supported()
            request.scope["causyn_context_ir_spec"] = spec_ir
            _safe_set_request_parsed_body(request, {"model": AUTH_MODEL})
        else:
            spec = MiniMaxH3Create.model_validate_json(raw)
            request.scope["minimax_h3_spec"] = spec
            body = spec.internal_body()
            if request.url.path == DIRECT_PREFIX + "/v2/video_generation":
                request.scope["causyn_direct_prompt"] = True
                body["prompt_processing"] = "direct"
            _safe_set_request_parsed_body(request, body)
    elif is_ir_list:
        _safe_set_request_parsed_body(request, {"model": INTERNAL_MODEL})
    elif is_ir:
        _safe_set_request_parsed_body(request, {"model": AUTH_MODEL})
    elif public_id.startswith(moderation_bridge.PREFIX):
        _safe_set_request_parsed_body(request, {"model": INTERNAL_MODEL})
    else:
        raise H3Error(404, "Task not found")


class MiniMaxH3Route(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                await prepare_request(request)
                return await original(request)
            except RewriteError as exc:
                return error_response(exc.status_code, str(exc))
            except H3Error as exc:
                return error_response(exc.code, str(exc))
            except json.JSONDecodeError:
                return error_response(400, "Invalid JSON body")
            except ValidationError as exc:
                return error_response(
                    400, "; ".join(item["msg"] for item in exc.errors(include_input=False, include_url=False))
                )
            except ProxyException as exc:
                try:
                    code = int(exc.code)
                except (TypeError, ValueError):
                    code = 500
                if not 400 <= code <= 599:
                    code = 500
                return error_response(code, exc.message if code < 500 else "Video provider request failed")
            except HTTPException as exc:
                return error_response(exc.status_code, str(exc.detail))
            except RuntimeError:
                verbose_proxy_logger.exception("MiniMax H3 request failed")
                return error_response(500, "Video request failed")

        return handle


async def context_ir_service_for_request(request: Request) -> ContextIRService | None:
    return get_context_ir_service() if request.scope.get("causyn_context_ir") else None


router = APIRouter(route_class=MiniMaxH3Route)


@router.post(DIRECT_PREFIX + "/v2/video_generation", tags=["MiniMax H3"])
@router.post(IR_PREFIX + "/v2/video_generation", tags=["MiniMax H3"])
async def create_video(
    request: Request,
    response: Response,
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> dict[str, str]:
    spec = request.scope.get("minimax_h3_spec")
    if not isinstance(spec, MiniMaxH3Create):
        raise H3Error(400, "Invalid video generation request")
    if spec.callback_url is not None:
        raise H3Error(422, "The current backend does not support callback_url; poll the query endpoint")
    if spec.keyframes and all(item.role == "last_frame" for item in spec.keyframes):
        raise H3Error(422, "The current backend requires a first_frame for keyframe generation")
    if not moderation_bridge.configured(auth):
        raise H3Error(503, "Moderation control service is required")
    video = await endpoints.video_generation(request, response, input_reference=None, user_api_key_dict=auth)
    if not isinstance(video, VideoObject) or not video.id:
        raise H3Error(500, "Video provider returned an invalid task")
    if video.id.startswith(moderation_bridge.PREFIX):
        return {"task_id": video.id}
    raise H3Error(503, "Durable moderation admission is required")


@router.get(DIRECT_PREFIX + "/v2/query/video_generation/{video_id}", tags=["MiniMax H3"])
@router.get(IR_PREFIX + "/v2/query/video_generation/{video_id}", tags=["MiniMax H3"])
async def query_video(
    video_id: str,
    request: Request,
    response: Response,
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    ir_service: Annotated[ContextIRService | None, Depends(context_ir_service_for_request)],
) -> dict[str, object]:
    if video_id.startswith(moderation_bridge.PREFIX):
        moderated = await moderation_bridge.query(request, auth, video_id)
        if moderated is None:
            raise H3Error(404, "Task not found")
        return {"task": moderation_bridge.v2_task(moderated, h3=True)}
    if ir_service is not None:
        return await query_context_ir_task(video_id, task_owner(auth), ir_service)
    raise H3Error(404, "Task not found")


async def query_context_ir_task(video_id: str, owner: str, service: ContextIRService) -> dict[str, object]:
    task = await service.store.get(video_id)
    if task is None or not task.listed or not hmac.compare_digest(task.owner, owner):
        raise H3Error(404, "Task not found")
    return {"task": task.public()}
