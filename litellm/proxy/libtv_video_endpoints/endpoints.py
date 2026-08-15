"""Two-stage (POST enqueue / GET poll) endpoints for the internal
video-generation worker.

Structurally mirrors ``litellm/proxy/image_endpoints/endpoints.py``'s
``libtv_validated_media_transfer`` route: module-level ``APIRouter``,
``Depends(user_api_key_auth)``, ``ORJSONResponse``, proxy-admin-only guard
checked first, manual ``orjson.loads(await request.body())`` parsing.

Note on package name: the plan's Task 13 names this package
``litellm/proxy/video_endpoints/``, but that path already exists in this
repo as an unrelated, fully-committed upstream feature (image/video
generation endpoints, ``litellm/proxy/video_endpoints/{__init__,endpoints,
utils}.py``). To avoid colliding with and shadowing that existing feature,
this module lives at ``litellm/proxy/libtv_video_endpoints/`` instead,
following the same ``libtv_``-prefix convention already used for the sibling
``libtv_validated_media_transfer`` route living inside ``image_endpoints``.
"""

from __future__ import annotations

import os

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import ORJSONResponse

from litellm.llms.libtv.transfer import get_transfer_redis
from litellm.llms.libtv.video_generate import (
    VideoGenerateError,
    VideoGenerateSettings,
    enqueue_video_generate,
    fetch_video_generate_status,
)
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth

router = APIRouter()

# Codes that represent a rejected request (bad input / no admission
# capacity) rather than a server-side misconfiguration. Anything else
# (including VideoGenerateSettings.from_environment()'s "misconfigured"
# code for an empty/malformed allowlist) defaults to 503 -- fail-closed
# configuration errors are a server problem, not a user-input problem
# (spec §1.1, mirroring validated_transfer._parse_hosts's existing
# semantics).
_REJECTION_STATUS_CODES = {
    "invalid_url": 422,
    "no_worker_available": 422,
    "no_capacity_available": 422,
    "invalid_params": 422,
}


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN.value)


def _get_redis_client():
    return get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL"))


@router.post(
    "/v1/libtv/video-generate",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
async def libtv_video_generate(request: Request, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
    if not _is_proxy_admin(user_api_key_dict):
        return ORJSONResponse(status_code=403, content={"error": "proxy-admin authorization is required"})
    try:
        payload = orjson.loads(await request.body())
        settings = VideoGenerateSettings.from_environment()
        task_id = await enqueue_video_generate(payload, redis=_get_redis_client(), settings=settings)
        return ORJSONResponse(status_code=202, content={"ok": True, "task_id": task_id})
    except orjson.JSONDecodeError as exc:
        return ORJSONResponse(status_code=422, content={"error": {"code": "invalid_params", "message": str(exc)}})
    except VideoGenerateError as exc:
        status_code = _REJECTION_STATUS_CODES.get(exc.code, 503)
        return ORJSONResponse(status_code=status_code, content={"error": {"code": exc.code, "message": exc.message}})


@router.get(
    "/v1/libtv/video-generate/{task_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
async def libtv_video_generate_status(
    task_id: str, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)
):
    if not _is_proxy_admin(user_api_key_dict):
        return ORJSONResponse(status_code=403, content={"error": "proxy-admin authorization is required"})
    body = await fetch_video_generate_status(task_id, redis=_get_redis_client())
    status_code = 200 if "status" in body else 404
    return ORJSONResponse(status_code=status_code, content=body)
