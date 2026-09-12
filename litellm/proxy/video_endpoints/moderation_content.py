from __future__ import annotations

import asyncio
import hashlib
from tempfile import TemporaryFile

import httpx
from fastapi import Request, Response
from pydantic import JsonValue, TypeAdapter

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import moderation_bridge as bridge
from litellm.videos.content_sink import VIDEO_CONTENT_SINK, VideoContentSink


async def materialize_content(
    request: Request,
    auth: UserAPIKeyAuth,
    native_id: str,
    *,
    intent_id: str | None = None,
    ticket: str | None = None,
) -> dict[str, str]:
    from litellm.proxy.video_endpoints.endpoints import video_content
    from litellm.proxy.video_endpoints.moderation_execution import execution_request

    async with asyncio.timeout(120):
        with TemporaryFile() as media:
            sink = VideoContentSink(media)
            context = VIDEO_CONTENT_SINK.set(sink)
            try:
                await video_content(
                    native_id,
                    execution_request(request, {}, "/v1/videos/" + native_id + "/content", "GET"),
                    Response(),
                    auth,
                )
            finally:
                VIDEO_CONTENT_SINK.reset(context)
            size = media.tell()
            if not 0 < size <= sink.max_bytes:
                raise ValueError("Provider did not supply bounded video content")
            media.seek(0)
            digest = hashlib.file_digest(media, "sha256").hexdigest()
            identity: dict[str, JsonValue] = (
                {"ticket": ticket, "native_id": native_id}
                if intent_id is not None and ticket is not None
                else {"principal": bridge.principal(auth)}
            )
            target = await bridge.platform(
                request,
                "POST",
                f"/intents/{intent_id}/output-upload" if intent_id is not None else "/uploads",
                {**identity, "digest": digest, "size": size},
            )
            media.seek(0)

            async def chunks():
                while chunk := media.read(65536):
                    yield chunk

            async with httpx.AsyncClient(
                timeout=120,
                follow_redirects=False,
                transport=getattr(request.app.state, "moderation_media_transport", None),
            ) as client:
                response = await client.put(
                    TypeAdapter(str).validate_python(target["url"]),
                    content=chunks(),
                    headers=TypeAdapter(dict[str, str]).validate_python(target["headers"]),
                )
                if response.status_code != 409:
                    response.raise_for_status()
            return {"private_reference": TypeAdapter(str).validate_python(target["reference"]), "digest": digest}
