import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.libtv.persistence import LibTVPersistence, account_key, get_persistence, normalize_source_key

from .common import (
    QUERY_RESULT_PATH,
    SUBMIT_RUN_PATH,
    UPLOAD_FILE_PATH,
    XIAOYUNQUE_API_BASE,
    XiaoyunqueContentPolicyError,
    XiaoyunqueError,
    build_xiaoyunque_headers,
    build_xiaoyunque_upload_headers,
    is_compliance_ret,
    is_rate_limit_ret,
)
from .common import is_ak_error as _is_ak_error

logger = logging.getLogger(__name__)

_ACCOUNT_KEY_NAMESPACE = "xiaoyunque"


def parse_upload_asset_id(payload: Dict[str, Any]) -> str:
    asset_id = (payload.get("data") or {}).get("pippit_asset_id")
    if not asset_id:
        raise XiaoyunqueError(status_code=502, message=f"xiaoyunque upload_file returned no pippit_asset_id: {payload}")
    return str(asset_id)


def parse_submit_run(payload: Dict[str, Any]) -> Dict[str, str]:
    run = (payload.get("data") or {}).get("run") or {}
    thread_id, run_id = run.get("thread_id"), run.get("run_id")
    if not thread_id or not run_id:
        raise XiaoyunqueError(status_code=502, message=f"xiaoyunque submit_run returned no thread_id/run_id: {payload}")
    return {"thread_id": str(thread_id), "run_id": str(run_id)}


def _coerce_run_state(raw: Any) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _string_urls(values: Any) -> List[str]:
    return [u for u in (values or []) if isinstance(u, str) and u]


def parse_query_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data") or {}
    return {
        "run_state": _coerce_run_state(data.get("run_state")),
        "video_urls": _string_urls(data.get("video_urls")),
        "image_urls": _string_urls(data.get("image_urls")),
        "fail_reason": data.get("fail_reason"),
    }


def encode_composite_task_id(thread_id: str, run_id: str) -> str:
    return f"{thread_id}~{run_id}"


def decode_composite_task_id(task_id: str) -> Tuple[str, str]:
    if "~" not in task_id:
        raise XiaoyunqueError(
            status_code=400, message=f"xiaoyunque video id does not carry thread_id~run_id: {task_id!r}"
        )
    thread_id, run_id = task_id.split("~", 1)
    if not thread_id or not run_id:
        raise XiaoyunqueError(status_code=400, message=f"xiaoyunque video id has empty thread_id/run_id: {task_id!r}")
    return thread_id, run_id


def _status_code_for_ret(ret: str, errmsg: str) -> int:
    if ret == "2":
        return 401 if _is_ak_error(errmsg) else 400
    if is_rate_limit_ret(ret):
        return 429
    return 502


def _filename_from_url(url: str, default_name: str) -> str:
    base = url.split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]
    return base if base and "." in base else default_name


class XiaoyunqueClient:
    def __init__(
        self,
        token: str,
        sync_client: Optional[HTTPHandler] = None,
        async_client: Optional[AsyncHTTPHandler] = None,
        api_base: str = XIAOYUNQUE_API_BASE,
        request_timeout: float = 600.0,
        http_get=None,
        persistence: Optional["LibTVPersistence"] = None,
    ):
        self.token = token
        self.sync_client = sync_client
        self.async_client = async_client
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        # Presigned object-store GET seam; default to bare httpx so the verbatim url
        # reaches the store. Injectable for tests.
        self._http_get = http_get
        self._persistence = persistence
        # Namespaced so the shared LibTV*-named cache/billing tables (reused rather than
        # duplicated -- see persistence.py) can never collide a xiaoyunque row with a
        # libtv row that happens to hash to the same bare account_key.
        self._account_key = f"{_ACCOUNT_KEY_NAMESPACE}:{account_key(token)}"

    def _get_persistence(self) -> Optional["LibTVPersistence"]:
        return self._persistence if self._persistence is not None else get_persistence()

    @property
    def headers(self) -> Dict[str, str]:
        return build_xiaoyunque_headers(self.token)

    @property
    def upload_headers(self) -> Dict[str, str]:
        return build_xiaoyunque_upload_headers(self.token)

    def _check(self, response: Any, step: str) -> Dict[str, Any]:
        headers = dict(getattr(response, "headers", {}) or {})
        if response.status_code != 200:
            raise XiaoyunqueError(
                status_code=response.status_code,
                message=f"xiaoyunque {step} HTTP {response.status_code}: {response.text[:300]}",
                headers=headers,
            )
        payload = response.json()
        ret = str(payload.get("ret", "0"))
        if ret == "0":
            return payload
        errmsg = str(payload.get("errmsg") or "")
        message = f"xiaoyunque {step} ret={ret} errmsg={errmsg}"
        if is_compliance_ret(ret):
            raise XiaoyunqueContentPolicyError(message=message)
        raise XiaoyunqueError(status_code=_status_code_for_ret(ret, errmsg), message=message, headers=headers)

    # ---------- sync ----------
    def _post(self, path: str, body: Dict[str, Any], step: str) -> Dict[str, Any]:
        assert self.sync_client is not None, "sync_client required for sync calls"
        resp = self.sync_client.post(
            url=f"{self.api_base}{path}", json=body, headers=self.headers, timeout=self.request_timeout
        )
        return self._check(resp, step)

    def upload_file(self, data: bytes, filename: str) -> str:
        assert self.sync_client is not None, "sync_client required for sync calls"
        resp = self.sync_client.post(
            url=f"{self.api_base}{UPLOAD_FILE_PATH}",
            files={"file": (filename, data)},
            headers=self.upload_headers,
            timeout=self.request_timeout,
        )
        return parse_upload_asset_id(self._check(resp, "upload_file"))

    def submit_run(
        self, message: str, asset_ids: List[str], video_part_tool_param: Dict[str, Any], thread_id: Optional[str] = None
    ) -> Dict[str, str]:
        body: Dict[str, Any] = {
            "message": message,
            "agent_name": "pippit_video_part_agent",
            "video_part_tool_param": video_part_tool_param,
        }
        if asset_ids:
            body["asset_ids"] = asset_ids
        if thread_id:
            body["thread_id"] = thread_id
        return parse_submit_run(self._post(SUBMIT_RUN_PATH, body, "submit_run"))

    def query_result(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        body = {"thread_id": thread_id, "run_id": run_id}
        return parse_query_result(self._post(QUERY_RESULT_PATH, body, "query_result"))

    def _fetch_bytes(self, url: str) -> bytes:
        if self._http_get is not None:
            return self._http_get(url)
        if not url.startswith(("http://", "https://")):
            raise XiaoyunqueError(status_code=400, message="xiaoyunque reference url must be http(s)")
        resp = httpx.get(url, follow_redirects=True, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise XiaoyunqueError(
                status_code=resp.status_code, message=f"xiaoyunque reference fetch HTTP {resp.status_code}"
            )
        return resp.content

    def _resolve_cache_target(self, source_key: Optional[str]) -> Optional[Tuple["LibTVPersistence", str]]:
        if os.getenv("XIAOYUNQUE_UPLOAD_CACHE_DISABLED") == "1" or source_key is None:
            return None
        persistence = self._get_persistence()
        return None if persistence is None else (persistence, source_key)

    def ensure_uploaded(self, kind: str, url: str, data: Optional[bytes], default_name: str) -> str:
        # No cache here: LibTVPersistence is async-only, and this sync path (like libtv's
        # own sync ensure_libtv_url) is unused by the production proxy, which routes
        # exclusively through the async avideo_generation path below.
        return self._upload_uncached(kind, url, data, default_name)

    def _upload_uncached(self, kind: str, url: str, data: Optional[bytes], default_name: str) -> str:
        if kind == "url":
            return self.upload_file(self._fetch_bytes(url), _filename_from_url(url, default_name))
        return self.upload_file(data or b"", url or default_name)

    # ---------- async ----------
    async def _apost(self, path: str, body: Dict[str, Any], step: str) -> Dict[str, Any]:
        assert self.async_client is not None, "async_client required for async calls"
        resp = await self.async_client.post(
            url=f"{self.api_base}{path}", json=body, headers=self.headers, timeout=self.request_timeout
        )
        return self._check(resp, step)

    async def aupload_file(self, data: bytes, filename: str) -> str:
        assert self.async_client is not None, "async_client required for async calls"
        resp = await self.async_client.post(
            url=f"{self.api_base}{UPLOAD_FILE_PATH}",
            files={"file": (filename, data)},
            headers=self.upload_headers,
            timeout=self.request_timeout,
        )
        return parse_upload_asset_id(self._check(resp, "upload_file"))

    async def asubmit_run(
        self, message: str, asset_ids: List[str], video_part_tool_param: Dict[str, Any], thread_id: Optional[str] = None
    ) -> Dict[str, str]:
        body: Dict[str, Any] = {
            "message": message,
            "agent_name": "pippit_video_part_agent",
            "video_part_tool_param": video_part_tool_param,
        }
        if asset_ids:
            body["asset_ids"] = asset_ids
        if thread_id:
            body["thread_id"] = thread_id
        return parse_submit_run(await self._apost(SUBMIT_RUN_PATH, body, "submit_run"))

    async def aquery_result(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        body = {"thread_id": thread_id, "run_id": run_id}
        return parse_query_result(await self._apost(QUERY_RESULT_PATH, body, "query_result"))

    async def _afetch_bytes(self, url: str) -> bytes:
        return await asyncio.to_thread(self._fetch_bytes, url)

    async def aensure_uploaded(self, kind: str, url: str, data: Optional[bytes], default_name: str) -> str:
        cache_target = self._resolve_cache_target(normalize_source_key(kind, url, data))
        if cache_target is not None:
            cached = await self._acache_lookup(*cache_target)
            if cached is not None:
                return cached
        asset_id = await self._aupload_uncached(kind, url, data, default_name)
        if cache_target is not None:
            await self._acache_store(*cache_target, asset_id)
        return asset_id

    async def _aupload_uncached(self, kind: str, url: str, data: Optional[bytes], default_name: str) -> str:
        if kind == "url":
            return await self.aupload_file(await self._afetch_bytes(url), _filename_from_url(url, default_name))
        return await self.aupload_file(data or b"", url or default_name)

    async def _acache_lookup(self, persistence: "LibTVPersistence", source_key: str) -> Optional[str]:
        try:
            return await persistence.cached_upload(self._account_key, source_key)
        except Exception:  # noqa: BLE001  # any read failure degrades to a cache miss, never breaks the upload path
            logger.warning("xiaoyunque upload cache: cached_upload failed", exc_info=True)
            return None

    async def _acache_store(self, persistence: "LibTVPersistence", source_key: str, asset_id: str) -> None:
        try:
            await persistence.store_upload(self._account_key, source_key, asset_id, 0)
        except Exception:  # noqa: BLE001  # cache write is best-effort; a failed insert just means no reuse next time
            logger.warning("xiaoyunque upload cache: store_upload failed", exc_info=True)
