"""Public direct-upload file routes: authenticated thin proxies over the platform's ``/internal/moderation/files``.

The platform owns signing, quota, validation and storage; this layer only authenticates the caller (through the same
H3 route class as the generation routes), forwards the verified principal and adds the H3 ``base_resp`` envelope.
Nothing here ever returns a URL a client could fetch the file from.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.video_endpoints import moderation_bridge
from litellm.proxy.video_endpoints.minimax_h3_endpoints import H3Error, MiniMaxH3Route
from litellm.proxy.video_endpoints.minimax_h3_paths import DIRECT_PREFIX, IR_PREFIX

FILE_ID = re.compile(r"cf_[0-9a-z]{26}")
BASE_RESP: dict[str, JsonValue] = {"status_code": 0, "status_msg": "success"}
Auth = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]

router = APIRouter(route_class=MiniMaxH3Route)


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["image", "video", "audio"]
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, strict=True)
    content_type: str = Field(min_length=1, max_length=64)


class ListQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=20, ge=1, le=100)
    after: str | None = Field(default=None, min_length=1, max_length=64)


def checked_id(file_id: str) -> str:
    """A malformed id can never exist: answer 404 without a platform round trip (and never splice it into a path)."""
    if FILE_ID.fullmatch(file_id) is None:
        raise moderation_bridge.file_not_found()
    return file_id


async def forward(
    request: Request, auth: UserAPIKeyAuth, path: str, extra: dict[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    result = await moderation_bridge.platform(
        request, "POST", path, {"principal": moderation_bridge.principal(auth), **(extra or {})}
    )
    return {**result, "base_resp": BASE_RESP}


@router.post(DIRECT_PREFIX + "/v1/causyn_files/uploads", tags=["MiniMax H3"])
@router.post(IR_PREFIX + "/v1/causyn_files/uploads", tags=["MiniMax H3"])
async def create_upload(request: Request, auth: Auth) -> dict[str, JsonValue]:
    raw = request.scope.get("causyn_files_body")
    if not isinstance(raw, (bytes, bytearray)):
        raise H3Error(400, "Invalid upload request")
    try:
        spec = UploadRequest.model_validate_json(bytes(raw))
    except ValidationError as exc:
        reason = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_input=False, include_url=False)
        )
        raise HTTPException(400, detail={"code": "invalid_upload", "message": reason}) from None
    return await forward(request, auth, "/files/uploads", spec.model_dump(mode="json"))


@router.post(DIRECT_PREFIX + "/v1/causyn_files/{file_id}/complete", tags=["MiniMax H3"])
@router.post(IR_PREFIX + "/v1/causyn_files/{file_id}/complete", tags=["MiniMax H3"])
async def complete_upload(file_id: str, request: Request, auth: Auth) -> dict[str, JsonValue]:
    return await forward(request, auth, f"/files/{checked_id(file_id)}/complete")


@router.get(DIRECT_PREFIX + "/v1/causyn_files/{file_id}", tags=["MiniMax H3"])
@router.get(IR_PREFIX + "/v1/causyn_files/{file_id}", tags=["MiniMax H3"])
async def get_file(file_id: str, request: Request, auth: Auth) -> dict[str, JsonValue]:
    return await forward(request, auth, f"/files/{checked_id(file_id)}/view")


@router.get(DIRECT_PREFIX + "/v1/causyn_files", tags=["MiniMax H3"])
@router.get(IR_PREFIX + "/v1/causyn_files", tags=["MiniMax H3"])
async def list_files(request: Request, auth: Auth) -> dict[str, JsonValue]:
    query = ListQuery.model_validate(dict(request.query_params))
    return await forward(request, auth, "/files/list", {"limit": query.limit, "after": query.after})


@router.delete(DIRECT_PREFIX + "/v1/causyn_files/{file_id}", tags=["MiniMax H3"])
@router.delete(IR_PREFIX + "/v1/causyn_files/{file_id}", tags=["MiniMax H3"])
async def delete_file(file_id: str, request: Request, auth: Auth) -> dict[str, JsonValue]:
    return await forward(request, auth, f"/files/{checked_id(file_id)}/delete")
