import os
import uuid
from typing import Optional

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


def build_bridge_headers(token: str) -> dict:
    return {"Token": token, "Content-Type": "application/json", "X-from-client": "cli"}


def build_upload_path(user_uuid: str, sha1_hex: str, filename: str) -> str:
    ext = filename[filename.rfind(".") :] if "." in filename else ""
    return f"{BRIDGE_UPLOAD_PREFIX}/{user_uuid}/{sha1_hex}{ext}"
