from __future__ import annotations

import asyncio
import secrets

import httpx
from pydantic import JsonValue

from litellm.litellm_core_utils.url_utils import validate_url
from litellm.llms.causyn.h3_prompt import RewriteError


async def post_callback(url: str, body: dict[str, JsonValue]) -> httpx.Response:
    async with asyncio.timeout(3):
        checked, host = await asyncio.to_thread(validate_url, url)
        async with httpx.AsyncClient(trust_env=False) as client:
            return await client.post(checked, headers={"Host": host}, json=body, timeout=3, follow_redirects=False)


async def verify_callback(url: str | None) -> None:
    if url is None:
        return
    challenge = secrets.token_urlsafe(24)
    try:
        response = await post_callback(url, {"challenge": challenge})
        if response.is_success and response.json() == {"challenge": challenge}:
            return
    except (httpx.HTTPError, TimeoutError, ValueError):
        pass
    raise RewriteError("callback_url must return the unchanged challenge within 3 seconds", 400)


async def notify_callback(url: str, body: dict[str, JsonValue]) -> bool:
    try:
        return (await post_callback(url, body)).is_success
    except (httpx.HTTPError, TimeoutError, ValueError):
        return False
