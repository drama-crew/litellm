import os
import uuid
from typing import Optional

LIBTV_API_BASE = "https://api.liblib.tv"

NODE_TYPE_BACKEND = {"image": 2, "video": 3}
NODE_ACTION = {"image": "image_generate", "video": "video_generate"}
NODE_DEFAULT_NAME = {"image": "图片节点", "video": "视频节点"}


class LibTVError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def resolve_libtv_credentials(token: Optional[str] = None, webid: Optional[str] = None) -> tuple:
    resolved_token = token or os.getenv("LIBTV_TOKEN") or os.getenv("LIBTV_CLI_USERTOKEN")
    resolved_webid = webid or os.getenv("LIBTV_WEBID") or os.getenv("LIBTV_CLI_WEBID")
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
