import os
import uuid
from typing import Mapping, Optional

LIBTV_API_BASE = "https://api.liblib.tv"
LIBTV_PASSPORT_BASE = "https://passport.liblib.art"
LIBTV_BRIDGE_BASE = "https://bridge.liblib.art"

BRIDGE_BIZ_CODE = "4"
BRIDGE_PART_SIZE = 5 * 1024 * 1024
BRIDGE_UPLOAD_PREFIX = "upload-images"

NODE_TYPE_BACKEND = {"image": 2, "video": 3}
NODE_ACTION = {"image": "image_generate", "video": "video_generate"}
NODE_DEFAULT_NAME = {"image": "图片节点", "video": "视频节点"}


class LibTVError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Mapping[str, str]] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = dict(headers or {})
        super().__init__(message)


class LibTVContentPolicyError(LibTVError):
    """A generation rejected by libtv content moderation (real public figure,
    copyright, other restricted content). Distinct from transient/capacity/billing
    failures so the caller can surface it as a content-policy rejection that must
    NOT be retried or routed to another provider."""

    def __init__(self, message: str):
        super().__init__(status_code=400, message=message)


_COMPLIANCE_REASON_TOKENS = (
    "版权",
    "侵权",
    "违规",
    "违禁",
    "敏感",
    "审核",
    "涉黄",
    "涉政",
    "色情",
    "血腥",
    "未成年",
    "请调整描述或素材",
    "copyright",
    "infring",
    "nsfw",
    "moderation",
    "content policy",
)


def is_compliance_failure(reason: Optional[str]) -> bool:
    """Whether a libtv failedReason denotes a content-moderation rejection.

    Poll-time failures only carry a free-text ``failedReason`` (no structured
    code), so classification is a conservative positive whitelist: only clear
    compliance wording matches. Capacity ("算力不足"), billing ("积分不足"),
    network faults and unknown reasons return False and stay retryable/fallback-able."""
    if not reason:
        return False
    text = reason.lower()
    return any(token in text for token in _COMPLIANCE_REASON_TOKENS)


def resolve_libtv_credentials(token: Optional[str] = None, webid: Optional[str] = None) -> tuple:
    # An explicitly configured empty value is a broken deployment and must fail
    # closed. Falling back to the process-wide account here can silently make two
    # pool slots use the same account.
    resolved_token = token if token is not None else os.getenv("LIBTV_TOKEN") or os.getenv("LIBTV_CLI_USERTOKEN")
    resolved_webid = webid if webid is not None else os.getenv("LIBTV_WEBID") or os.getenv("LIBTV_CLI_WEBID")
    if not resolved_token:
        raise LibTVError(
            status_code=401,
            message="LibTV usertoken missing. Set LIBTV_TOKEN (or pass api_key).",
        )
    if not resolved_webid:
        raise LibTVError(
            status_code=401,
            message="LibTV webid missing. Set LIBTV_WEBID.",
        )
    return resolved_token, resolved_webid


def build_libtv_headers(token: str, webid: str) -> dict:
    return {
        "token": token,
        "webid": webid,
        "x-language": "zh",
        "X-Log-ID": str(uuid.uuid4()),
        "X-from-client": "cli",
        "Content-Type": "application/json",
    }


def build_bridge_headers(token: str) -> dict:
    return {"Token": token, "Content-Type": "application/json", "X-from-client": "cli"}


def build_upload_path(user_uuid: str, sha1_hex: str, filename: str) -> str:
    ext = filename[filename.rfind(".") :] if "." in filename else ""
    return f"{BRIDGE_UPLOAD_PREFIX}/{user_uuid}/{sha1_hex}{ext}"
