"""Authenticated opaque video IDs for the LibTV provider.

The public video API must not put provider or deployment routing metadata in a
client-visible identifier.  This codec therefore encrypts the routing tuple
and authenticates it with AES-GCM.  The key is deliberately sourced without
importing the proxy package so SDK/provider tests can use it independently.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any

from litellm.types.videos.main import DecodedVideoId
from litellm.types.videos.utils import register_video_id_codec

OPAQUE_VIDEO_ID_PREFIX = "video_v2_"
VIDEO_ID_SECRET_ENV = "LITELLM_VIDEO_ID_SECRET"
VIDEO_ID_KEY_DOMAIN = b"litellm-video-id-v2:libtv"


class VideoIdKeyUnavailable(RuntimeError):
    """Raised when a new opaque ID cannot be safely authenticated."""


def _secret() -> str:
    # Existing production workers already share LITELLM_MASTER_KEY.  Dedicated
    # and salt values remain useful for isolated deployments, but every source
    # is domain-separated below and no generated/plaintext fallback is allowed.
    return (
        os.getenv(VIDEO_ID_SECRET_ENV)
        or os.getenv("LITELLM_SALT_KEY")
        or os.getenv("LITELLM_MASTER_KEY")
        or ""
    )


def _key() -> bytes:
    secret = _secret()
    if not secret:
        raise VideoIdKeyUnavailable(f"{VIDEO_ID_SECRET_ENV} is required for new video IDs")
    return hashlib.sha256(VIDEO_ID_KEY_DOMAIN + b"\0" + secret.encode("utf-8")).digest()


def ensure_libtv_video_id_key() -> None:
    """Preflight the key before any paid provider operation is attempted."""
    _key()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def encode_libtv_video_id(video_id: str, model_id: str | None = None) -> str:
    """Return a URL-safe, authenticated opaque ID for a LibTV task."""
    if not video_id:
        return video_id
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    payload = json.dumps(
        {"v": 2, "p": "libtv", "m": model_id or "", "i": video_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    nonce = os.urandom(12)
    encrypted = AESGCM(_key()).encrypt(nonce, payload, OPAQUE_VIDEO_ID_PREFIX.encode("ascii"))
    return OPAQUE_VIDEO_ID_PREFIX + _b64(nonce + encrypted)


def decode_libtv_video_id(value: str) -> DecodedVideoId | None:
    """Decode an opaque ID, returning ``None`` for invalid/tampered values."""
    if not isinstance(value, str) or not value.startswith(OPAQUE_VIDEO_ID_PREFIX):
        return None
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        blob = _unb64(value[len(OPAQUE_VIDEO_ID_PREFIX) :])
        if len(blob) <= 12 + 16:
            return None
        payload = AESGCM(_key()).decrypt(
            blob[:12], blob[12:], OPAQUE_VIDEO_ID_PREFIX.encode("ascii")
        )
        data: Any = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict) or data.get("v") != 2 or data.get("p") != "libtv":
            return None
        task_id = data.get("i")
        model_id = data.get("m")
        if not isinstance(task_id, str) or not task_id or not isinstance(model_id, str):
            return None
        return DecodedVideoId(custom_llm_provider="libtv", model_id=model_id or None, video_id=task_id)
    except Exception:
        # Do not expose whether key, ciphertext, or payload validation failed.
        return None


register_video_id_codec("libtv", OPAQUE_VIDEO_ID_PREFIX, decode_libtv_video_id)
