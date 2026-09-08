"""Durable record of what a libtv video submission actually asked for.

Neither side of this integration keeps the create request: the vendor exposes no
task-detail endpoint, our status poll carries only status/urls, and the usage
row we persist holds duration/resolution for billing and nothing else. So when a
caller reports "the result ignored my first frame", the only way to answer is a
record written at submission time, keyed by the vendor task id.

Two records, same payload: a log line (grep-able in the pod) and an OTel span
(queryable in the collector). Reference URLs are deliberately excluded -- callers
routinely pass presigned object-store URLs whose query string is a credential.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

AUDIT_LOGGER_NAME = "litellm.libtv.audit"
TRACER_NAME = "litellm.libtv"
SPAN_NAME = "libtv.video.submit"

# Production runs a WARNING root, so a logger that leaves its own level unset
# never even creates the record. Pin INFO here: the handler's root already owns
# a stdout handler, and this line is an audit trail, not a warning.
audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
audit_logger.setLevel(logging.INFO)

# Keep in sync with handler._REFERENCE_KEYS: which key a caller reached for is
# exactly the thing under dispute when a mode looks wrong.
REFERENCE_KEYS = (
    "input_reference",
    "image_references",
    "reference_images",
    "image",
    "last_image",
    "video_references",
    "reference_videos",
    "audio_references",
    "reference_audios",
)


def build_video_submission_record(
    *,
    model: str,
    mode: Optional[str],
    images: list,
    videos: list,
    audios: list,
    optional_params: dict,
    task_id: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    op = optional_params or {}
    return {
        "model": model,
        "mode": mode,
        "task_id": task_id,
        "reference_images": len(images or []),
        "reference_videos": len(videos or []),
        "reference_audios": len(audios or []),
        "reference_keys": [key for key in REFERENCE_KEYS if key in op],
        "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
        "resolution": op.get("resolution") or op.get("size"),
        "quality": op.get("quality"),
        "seconds": op.get("seconds") or op.get("duration"),
        "aspect_ratio": op.get("aspect_ratio") or op.get("ratio"),
    }


def _span_attributes(record: dict) -> dict:
    attrs: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        attrs[f"libtv.{key}"] = ",".join(value) if isinstance(value, list) else value
    return attrs


def record_video_submission(
    *,
    model: str,
    mode: Optional[str],
    images: list,
    videos: list,
    audios: list,
    optional_params: dict,
    task_id: Optional[str] = None,
    prompt: Optional[str] = None,
    tracer: Any = None,
) -> None:
    """Best-effort: an audit record must never fail a paid submission."""
    try:
        record = build_video_submission_record(
            model=model,
            mode=mode,
            images=images,
            videos=videos,
            audios=audios,
            optional_params=optional_params,
            task_id=task_id,
            prompt=prompt,
        )
    except Exception:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning("libtv audit record build failed", exc_info=True)
        return

    audit_logger.info("libtv video submission %s", json.dumps(record, sort_keys=True, ensure_ascii=False))

    try:
        if tracer is None:
            from opentelemetry import trace

            tracer = trace.get_tracer(TRACER_NAME)
        with tracer.start_as_current_span(SPAN_NAME) as span:
            span.set_attributes(_span_attributes(record))
    except Exception:  # pragma: no cover - tracing must never break generation
        logging.getLogger(__name__).debug("libtv audit span failed", exc_info=True)
