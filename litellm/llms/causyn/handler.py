"""causyn — queue-backed video provider for the intranet GPU worker fleet.

Every other video provider in this repo makes an outbound HTTP call to a
vendor API. This one cannot: the worker sits on an intranet machine that
accepts no inbound connections and only ever polls the platform. litellm
supports exactly this shape through ``litellm.custom_provider_map`` --
``litellm/videos/main.py``'s ``_custom_video_generation`` / ``_custom_video_status``
/ ``_custom_video_content`` dispatch to a handler without going anywhere near
``base_llm_http_handler``, and none of the three signatures implies a round
trip. So "submit" here means XADD onto a Redis stream, and "poll" means
reading that task's status back.

Design: docs/superpowers/specs/2026-08-23-causyn-litellm-provider-design.md

Three deliberate non-responsibilities, all resolved by keeping object-store
work on the platform side (which owns the credentials and already has a
correct, recently-debugged signing implementation):

* the staging upload URL is NOT signed here -- the platform's worker-runner
  injects it when the worker claims the task;
* the finished object is NOT read from OSS here -- the platform presigns a GET
  when it records the worker's result, and ``avideo_content`` merely fetches
  that URL over plain HTTPS;
* consequently this module needs no object-store SDK and no OSS credentials.

The enqueue/poll engine itself is reused from ``litellm.llms.libtv.video_generate``
rather than reimplemented: that module already carries the fail-closed URL
allowlists, capacity admission, dedupe and status translation, all hardened
over a review pass. This handler is a standard-interface face on it, not a
second copy.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional, Union

import httpx

from litellm.llms.custom_llm import CustomLLM
from litellm.llms.libtv.transfer import get_transfer_redis
from litellm.llms.libtv.video_generate import (
    VideoGenerateError,
    VideoGenerateSettings,
    enqueue_video_generate,
    fetch_video_generate_status,
)
from litellm.types.videos.main import VideoObject

VIDEO_ID_PREFIX = "causyn_"

# Deadline handed to the worker fleet. Generous relative to a single render
# (the model tops out at 8 seconds of footage) because it also has to cover
# queueing behind other tasks on a single-GPU worker.
DEFAULT_DEADLINE_SECONDS = 1800

# Status translation. The engine already reduces the worker's raw states to
# {queued, running, succeeded, failed}; this maps that onto the OpenAI video
# vocabulary the proxy speaks.
_STATUS_TO_OPENAI = {
    "queued": "queued",
    "running": "in_progress",
    "succeeded": "completed",
    "failed": "failed",
}


def _redis_factory() -> Any:
    """Zero-arg factory, deliberately not an already-resolved client.

    ``enqueue_video_generate`` calls this only after its redis-free shape and
    URL checks pass, so a misconfigured Redis cannot preempt a deterministic
    rejection. Handing it a resolved client would defeat that: Python
    evaluates the argument expression before the callee's body runs.
    """
    return get_transfer_redis(os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL"))


def _gray_rollout_enabled() -> bool:
    """Mirror of the platform's DRAMA_INTERNAL_VIDEO_ENABLED admission gate.

    Registering the model in ``model_list`` makes it visible on ``/v1/models``
    to every key allowed to call it, which would otherwise let a sandbox agent
    reach it directly while the platform still refuses to admit it. Reading
    the same flag here keeps one meaning of "not open yet" across both paths.
    """
    raw = os.getenv("DRAMA_INTERNAL_VIDEO_ENABLED") or os.getenv(
        "OH_DRAMA_INTERNAL_VIDEO_ENABLED"
    )
    return bool(raw) and raw.strip().lower() in {"1", "true", "yes", "on"}


def _task_id_from_video_id(video_id: str) -> str:
    """Strip the public prefix.

    The id is deliberately just ``causyn_<task_id>`` over an id we generated
    ourselves. It is NOT the sibling libtv encoding, which base64s
    ``custom_llm_provider`` and the internal deployment id into the value --
    plain base64, so any caller can decode the vendor name and account-pool
    index straight out of a video id.
    """
    if not video_id.startswith(VIDEO_ID_PREFIX):
        raise VideoGenerateError("invalid_params", f"not a causyn video id: {video_id!r}")
    task_id = video_id[len(VIDEO_ID_PREFIX) :]
    if not task_id:
        raise VideoGenerateError("invalid_params", "causyn video id carries no task id")
    return task_id


def _references_from_params(optional_params: dict) -> list:
    """Normalise the OpenAI-shaped reference images into task-envelope items.

    Accepts both the platform's ``reference_images`` (list of URLs) and an
    already-shaped ``references`` list, so drama-cli and the platform can send
    whichever they already build without a per-caller branch here.
    """
    shaped = optional_params.get("references")
    if isinstance(shaped, list) and shaped:
        return shaped
    urls = optional_params.get("reference_images") or []
    if isinstance(urls, str):
        urls = [urls]
    return [
        {"role": "reference", "media_type": "image", "url": url}
        for url in urls
        if isinstance(url, str) and url
    ]


class CausynVideoHandler(CustomLLM):
    """Queue-backed video provider. Async only.

    The synchronous half of ``CustomLLM``'s video surface is intentionally left
    unimplemented: both callers (the platform's video_provider and drama-cli)
    use the async path, so a sync implementation would be an untested branch
    standing between a caller and a GPU.
    """

    async def avideo_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        optional_params: Optional[dict] = None,
        logging_obj: Any = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Any] = None,
    ) -> VideoObject:
        params: Dict[str, Any] = dict(optional_params or {})
        if not _gray_rollout_enabled():
            raise VideoGenerateError(
                "forbidden",
                "causyn video generation is not enabled on this deployment",
            )

        task_id = str(uuid.uuid4())
        request: Dict[str, Any] = {"prompt": prompt}

        seconds = params.get("seconds", params.get("duration_seconds"))
        if seconds is not None:
            request["duration_seconds"] = int(round(float(seconds)))
        size = params.get("size", params.get("resolution"))
        if size is not None:
            request["resolution"] = str(size)
        ratio = params.get("aspect_ratio", params.get("ratio"))
        if ratio is not None:
            request["ratio"] = str(ratio)
        seed = params.get("seed")
        if seed is not None:
            request["seed"] = seed
        request["references"] = _references_from_params(params)

        payload = {
            "task_id": task_id,
            "model": model,
            "deadline_ts": time.time() + DEFAULT_DEADLINE_SECONDS,
            "request": request,
        }

        settings = VideoGenerateSettings.from_environment()
        await enqueue_video_generate(
            payload, redis_factory=_redis_factory, settings=settings
        )
        return VideoObject(
            id=f"{VIDEO_ID_PREFIX}{task_id}",
            object="video",
            status="queued",
            model=model,
            created_at=int(time.time()),
            seconds=str(request.get("duration_seconds"))
            if request.get("duration_seconds") is not None
            else None,
            size=request.get("resolution"),
        )

    async def avideo_status(
        self,
        video_id: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        optional_params: Optional[dict] = None,
        logging_obj: Any = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Any] = None,
    ) -> VideoObject:
        task_id = _task_id_from_video_id(video_id)
        body = await fetch_video_generate_status(task_id, redis=_redis_factory())

        if "status" not in body:
            raise VideoGenerateError("unknown_task", "unknown task id")

        status = _STATUS_TO_OPENAI.get(body["status"], "failed")
        result = body.get("result") if isinstance(body.get("result"), dict) else None

        return VideoObject(
            id=video_id,
            object="video",
            status=status,
            error=body.get("error"),
            completed_at=int(time.time()) if status in ("completed", "failed") else None,
            seconds=str(result["duration_seconds"])
            if result and result.get("duration_seconds") is not None
            else None,
            # The whole point of carrying this: the finished object already
            # lives in OUR object store, so the platform finalises it with a
            # server-side copy instead of pulling the bytes back through here
            # (see design §3.2b -- routing them through /content would move the
            # same bytes OSS -> litellm -> platform -> OSS).
            object_store_result=result,
        )

    async def avideo_content(
        self,
        video_id: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        optional_params: Optional[dict] = None,
        logging_obj: Any = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Any] = None,
    ) -> bytes:
        """Fetch the finished render's bytes.

        Only the sandbox path needs this (an agent has to land the file inside
        its own filesystem); the platform takes the server-side copy instead.
        The URL is presigned by the platform when it records the worker's
        result -- this module holds no object-store credentials and no SDK, it
        just fetches a URL somebody else signed, exactly as
        validated_transfer.py consumes presigned parts.
        """
        task_id = _task_id_from_video_id(video_id)
        body = await fetch_video_generate_status(task_id, redis=_redis_factory())
        result = body.get("result") if isinstance(body.get("result"), dict) else None
        url = (result or {}).get("staging_url")
        if not isinstance(url, str) or not url:
            raise VideoGenerateError(
                "result_unavailable",
                "no downloadable url on this task's result",
            )
        async with httpx.AsyncClient(timeout=timeout or 120.0) as http:
            response = await http.get(url)
            response.raise_for_status()
            return response.content


causyn_video_handler = CausynVideoHandler()
