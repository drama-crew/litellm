import os
from typing import Mapping, Optional

XIAOYUNQUE_API_BASE = "https://xyq.jianying.com"
UPLOAD_FILE_PATH = "/api/biz/v1/skill/upload_file"
SUBMIT_RUN_PATH = "/api/biz/v1/skill/submit_run"
QUERY_RESULT_PATH = "/api/biz/v1/agent/query_generate_video_result"

AGENT_NAME = "pippit_video_part_agent"


class XiaoyunqueError(Exception):
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


class XiaoyunqueContentPolicyError(XiaoyunqueError):
    """A generation rejected by xiaoyunque content moderation (ret 12004/12005/12006/12015).
    Distinct from transient/param failures so the caller can surface it as a content-policy
    rejection that must NOT be retried or routed to another provider."""

    def __init__(self, message: str):
        super().__init__(status_code=400, message=message)


_AK_ERROR_TOKENS = ("Ak无效", "Ak为空", "Ak明细", "该Ak未启用", "Ak已过期", "未查询到有效的Ak")
_COMPLIANCE_RETS = frozenset({"12004", "12005", "12006", "12015"})
_RATE_LIMIT_RETS = frozenset({"10", "15", "16010"})


def is_ak_error(errmsg: Optional[str]) -> bool:
    """Whether a ret=2 errmsg denotes an AK (Access Key) authentication failure.

    ret=2 also covers plain parameter errors unrelated to credentials, so this is a
    conservative positive whitelist: only clear AK wording is treated as an
    authentication failure eligible for router failover to another account; any other
    ret=2 errmsg (unknown included) stays a terminal BadRequestError."""
    if not errmsg:
        return False
    return any(token in errmsg for token in _AK_ERROR_TOKENS)


def is_compliance_ret(ret: str) -> bool:
    return ret in _COMPLIANCE_RETS


def is_rate_limit_ret(ret: str) -> bool:
    return ret in _RATE_LIMIT_RETS


def resolve_xiaoyunque_credentials(token: Optional[str] = None) -> str:
    # An explicitly configured empty token is a broken deployment and must fail closed:
    # falling back to the process-wide XIAOYUNQUE_TOKEN here could silently make a second
    # pool slot (meant to use XIAOYUNQUE_TOKEN_2) reuse the first account.
    resolved_token = token if token is not None else os.getenv("XIAOYUNQUE_TOKEN")
    if not resolved_token:
        raise XiaoyunqueError(
            status_code=401,
            message="fb3 access key missing. Configure the provider credentials, or pass api_key.",
        )
    return resolved_token


def build_xiaoyunque_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def build_xiaoyunque_upload_headers(token: str) -> dict:
    # No Content-Type here: httpx computes the multipart boundary itself from `files=`.
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
