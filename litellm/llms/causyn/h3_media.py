from __future__ import annotations

import asyncio
import base64
import io
import math

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.url_utils import validate_url
from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError
from litellm.proxy.video_endpoints.minimax_h3_models import ImageItem, MediaURL, VideoItem


async def fetch_media(client: httpx.AsyncClient, url: str, limit: int, redirects: int = 0) -> bytes:
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
        if len(raw) > limit:
            raise RewriteError("Reference file exceeds the permitted size", 400)
        return raw
    if redirects > 5:
        raise RewriteError("Reference URL has too many redirects", 400)
    checked, host = await asyncio.to_thread(validate_url, url)
    async with client.stream("GET", checked, headers={"Host": host}, timeout=15, follow_redirects=False) as response:
        if response.is_redirect:
            target = TypeAdapter[str | None](str | None).validate_python(response.headers.get("location"))
            if not target:
                raise RewriteError("Invalid reference redirect", 400)
            return await fetch_media(client, str(httpx.URL(url).join(target)), limit, redirects + 1)
        if response.status_code != 200:
            raise RewriteError("Reference file is not accessible", 400)
        with io.BytesIO() as output:
            async for chunk in response.aiter_bytes():
                if output.tell() + len(chunk) > limit:
                    raise RewriteError("Reference file exceeds the permitted size", 400)
                output.write(chunk)
            return output.getvalue()


def validate_dimensions(width: int, height: int) -> None:
    if not (256 <= width <= 5760 and 256 <= height <= 5760 and 0.4 <= width / height <= 2.5):
        raise RewriteError("Reference dimensions must be 256-5760 pixels and aspect ratio 0.4-2.5", 400)


def prepare_image(raw: bytes) -> str:
    from PIL import Image, ImageOps
    from pillow_heif import (  # pyright: ignore[reportMissingTypeStubs]  # pillow-heif ships no stubs
        register_heif_opener,  # pyright: ignore[reportUnknownVariableType]  # the zero-argument API is supported
    )

    register_heif_opener()
    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in {"JPEG", "PNG", "WEBP", "HEIF"}:
            raise RewriteError("Reference image must be JPG, PNG, WEBP, HEIC or HEIF", 400)
        validate_dimensions(*source.size)
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=88, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def inspect_video(raw: bytes) -> float:
    import av

    with av.open(io.BytesIO(raw), mode="r", format="mov", options={"enable_drefs": "0"}) as container:
        if len(container.streams.video) != 1:
            raise RewriteError("Reference video must contain exactly one video stream", 400)
        video = container.streams.video[0]
        if video.codec_context.name not in {"h264", "hevc"}:
            raise RewriteError("Reference video must use H.264 or H.265", 400)
        if any(audio.codec_context.name not in {"aac", "mp3", "mp3float"} for audio in container.streams.audio):
            raise RewriteError("Reference video audio must use AAC or MP3", 400)
        validate_dimensions(video.codec_context.width, video.codec_context.height)
        fps = float(video.average_rate or 0)
        if not 23.976 <= fps <= 60:
            raise RewriteError("Reference video frame rate must be 23.976-60 fps", 400)
        duration = float(container.duration or 0) / av.time_base
        if not math.isfinite(duration) or not 2 <= duration <= 15:
            raise RewriteError("Each reference video must last 2-15 seconds", 400)
        return duration


async def prepare_reference(
    client: httpx.AsyncClient, item: ImageItem | VideoItem
) -> tuple[ImageItem | VideoItem, float]:
    if isinstance(item, ImageItem):
        raw = await fetch_media(client, item.image_url.url, 30 * 1024 * 1024)
        url = await asyncio.to_thread(prepare_image, raw)
        return item.model_copy(update={"image_url": MediaURL(url=url)}), 0.0
    raw = await fetch_media(client, item.video_url.url, 50 * 1024 * 1024)
    duration = await asyncio.to_thread(inspect_video, raw)
    url = "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")
    return item.model_copy(update={"video_url": MediaURL(url=url)}), duration


async def prepare_media(client: httpx.AsyncClient, spec: ContextIRRequest) -> ContextIRRequest:
    try:
        async with asyncio.timeout(60):
            media = tuple(
                [
                    await prepare_reference(client, item)
                    for item in spec.ordered_media
                    if isinstance(item, (ImageItem, VideoItem))
                ]
            )
        if sum(duration for _, duration in media) > 15.000001:
            raise RewriteError("Combined reference video duration exceeds 15 seconds", 400)
        return spec.model_copy(
            update={
                "content": (
                    *(item for item in spec.content if not isinstance(item, (ImageItem, VideoItem))),
                    *(item for item, _ in media),
                )
            }
        )
    except RewriteError:
        raise
    except Exception as exc:
        raise RewriteError("Reference media could not be validated", 400) from exc
