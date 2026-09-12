from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import BinaryIO, Mapping

import httpx


@dataclass(frozen=True, slots=True)
class VideoContentSink:
    file: BinaryIO
    max_bytes: int = 256 * 1024 * 1024
    timeout_seconds: float = 120


VIDEO_CONTENT_SINK: ContextVar[VideoContentSink | None] = ContextVar("video_content_sink", default=None)


async def stream_content_to_sink(
    client: httpx.AsyncClient, url: str, headers: Mapping[str, str], sink: VideoContentSink
) -> None:
    sink.file.seek(0)
    sink.file.truncate(0)
    async with asyncio.timeout(sink.timeout_seconds):
        async with client.stream(
            "GET", url, headers=headers, follow_redirects=False, timeout=sink.timeout_seconds
        ) as response:
            response.raise_for_status()
            if response.status_code != 200 or int(response.headers.get("content-length", 0)) > sink.max_bytes:
                raise ValueError("Video content unavailable or exceeds byte limit")
            async for chunk in response.aiter_bytes(65536):
                if sink.file.tell() + len(chunk) > sink.max_bytes:
                    raise ValueError("Video content exceeds byte limit")
                sink.file.write(chunk)
            if sink.file.tell() == 0:
                raise ValueError("Video content is empty")
