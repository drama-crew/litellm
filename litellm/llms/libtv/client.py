import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from .common import (
    BRIDGE_BIZ_CODE,
    BRIDGE_PART_SIZE,
    LIBTV_API_BASE,
    LIBTV_BRIDGE_BASE,
    LIBTV_PASSPORT_BASE,
    NODE_ACTION,
    NODE_DEFAULT_NAME,
    NODE_TYPE_BACKEND,
    LibTVError,
    build_bridge_headers,
    build_libtv_headers,
    build_upload_path,
)


THIRD_ASSET_POLL_ATTEMPTS = 30


def parse_upload_url(complete_payload: Dict[str, Any]) -> str:
    data = complete_payload.get("data") or {}
    url = data.get("cdnUrl") or data.get("ossUrl") or data.get("path")
    if not url or not isinstance(url, str):
        raise LibTVError(status_code=502, message=f"libtv upload complete returned no url: {complete_payload}")
    return url


def build_node_batch_body(
    project_uuid: str,
    node_kind: str,
    node_key: str,
    name: str,
    model_key: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    node_data: Dict[str, Any] = {
        "type": node_kind,
        "name": name,
        "url": [],
        "action": NODE_ACTION[node_kind],
        "generatorType": "default",
        "params": {**params, "model": model_key},
    }
    if node_kind == "video":
        node_data["poster"] = ""
    canvas_node = {
        "nodeKey": node_key,
        "projectUuid": project_uuid,
        "type": NODE_TYPE_BACKEND[node_kind],
        "name": name,
        "position": {"positionX": "0", "positionY": "0"},
        "parentKey": "",
        "data": json.dumps(node_data, ensure_ascii=False),
    }
    return {
        "projectUuid": project_uuid,
        "nodes": {"create": [canvas_node]},
        "connections": {},
    }


def build_generation_body(
    model_key: str,
    vendor: str,
    task_type: str,
    params: Dict[str, Any],
    node_key: str,
    project_uuid: str,
    team_id: Optional[int] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "params": params,
        "metadata": {"node_id": node_key, "project_id": project_uuid},
        "provider": vendor,
        "model": model_key,
        "taskType": task_type,
        "requestId": str(uuid.uuid4()),
    }
    if isinstance(team_id, int) and team_id > 0:
        body["teamId"] = team_id
    return body


def parse_project(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = (payload.get("data") or {}).get("projectMeta") or {}
    uuid_val = meta.get("uuid")
    if not uuid_val:
        raise LibTVError(status_code=502, message=f"libtv project/create returned no uuid: {payload}")
    team_id = meta.get("teamId")
    return {"project_uuid": str(uuid_val), "team_id": team_id if isinstance(team_id, int) else None}


def parse_task_id(payload: Dict[str, Any]) -> str:
    data = payload.get("data") or {}
    task_id = data.get("taskId") or data.get("task_id") or ""
    if not task_id:
        raise LibTVError(status_code=502, message=f"libtv generation/create returned no taskId: {payload}")
    return str(task_id)


def parse_verify_passed(payload: Dict[str, Any]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for item in ((payload.get("data") or {}).get("list") or []):
        url = item.get("url")
        if not url:
            continue
        risk = item.get("riskLabels")
        if isinstance(risk, str) and risk.strip():
            try:
                risk = json.loads(risk)
            except json.JSONDecodeError:
                risk = {}
        out[str(url)] = bool(isinstance(risk, dict) and risk.get("passed"))
    return out


def parse_third_asset_uuid(payload: Dict[str, Any]) -> str:
    uuid_val = (payload.get("data") or {}).get("uuid")
    if not uuid_val:
        raise LibTVError(status_code=502, message=f"libtv third_asset/create returned no uuid: {payload}")
    return str(uuid_val)


def parse_third_asset_item(payload: Dict[str, Any], asset_uuid: str) -> Optional[Dict[str, Any]]:
    for item in ((payload.get("data") or {}).get("list") or []):
        if item.get("uuid") == asset_uuid:
            return item
    return None


def _asset_ref_from_item(item: Optional[Dict[str, Any]], cdn_url: str) -> Optional[str]:
    # asset://<id> once the backend issues a verified asset id (portrait); the libtv cdn
    # url once the asset reaches a terminal state without one (non-portrait, exempt); None
    # while the check is still pending so the caller keeps polling.
    if not item:
        return None
    if item.get("assetId"):
        return f"asset://{item['assetId']}"
    if item.get("status") == 1:
        return cdn_url
    return None


def _pick_item_url(item: Dict[str, Any]) -> Optional[str]:
    for key in ("videoUrl", "previewPath", "url", "imageUrl"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_urls(task_result: Dict[str, Any], kind: str) -> List[str]:
    def collect(items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        return [u for u in (_pick_item_url(i) for i in items if isinstance(i, dict)) if u]

    if kind == "video":
        return collect(task_result.get("videos")) or collect(task_result.get("images"))
    return collect(task_result.get("images"))


def parse_progress(payload: Dict[str, Any], kind: str) -> Dict[str, Any]:
    data = payload.get("data") or {}
    progresses = data.get("progresses") or []
    if not progresses:
        return {"status": None, "urls": [], "failed_reason": None}
    last = progresses[0] or {}
    raw_status = last.get("status")
    try:
        status = int(raw_status)
    except (TypeError, ValueError):
        status = None
    urls: List[str] = []
    if status == 2:
        raw = last.get("taskResult")
        parsed: Dict[str, Any] = {}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(raw, dict):
            parsed = raw
        urls = _extract_urls(parsed, kind)
    return {"status": status, "urls": urls, "failed_reason": last.get("failedReason")}


class LibTVClient:
    def __init__(
        self,
        token: str,
        webid: str,
        sync_client: Optional[HTTPHandler] = None,
        async_client: Optional[AsyncHTTPHandler] = None,
        api_base: str = LIBTV_API_BASE,
        poll_interval: float = 3.0,
        poll_max_attempts: int = 200,
        request_timeout: float = 60.0,
        http_get=None,
        http_put=None,
    ):
        self.token = token
        self.webid = webid
        self.sync_client = sync_client
        self.async_client = async_client
        self.api_base = api_base.rstrip("/")
        self.poll_interval = poll_interval
        self.poll_max_attempts = poll_max_attempts
        self.request_timeout = request_timeout
        # Presigned object-store GET/PUT seams; default to bare httpx so the verbatim url
        # reaches the store (the litellm http client breaks the signature on the async
        # path). Injectable for tests.
        self._http_get = http_get
        self._http_put = http_put
        self._tool_spec_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._user_uuid: Optional[str] = None

    @property
    def headers(self) -> Dict[str, str]:
        return build_libtv_headers(self.token, self.webid)

    def _parse_user_uuid(self, payload: Dict[str, Any]) -> str:
        user_uuid = (payload.get("data") or {}).get("uuid")
        if not user_uuid:
            raise LibTVError(status_code=502, message="libtv getUserInfo returned no uuid")
        return str(user_uuid)

    def _bridge_url(self, action: str) -> str:
        return f"{LIBTV_BRIDGE_BASE}/gateway/oss-server-api/oss-service/api/oss/pre-sign/multipart/{action}/{BRIDGE_BIZ_CODE}"

    def resolve_user_uuid(self) -> str:
        if self._user_uuid is None:
            assert self.sync_client is not None, "sync_client required"
            resp = self.sync_client.post(
                url=f"{LIBTV_PASSPORT_BASE}/api/www/user/getUserInfo",
                json={},
                headers=self.headers,
                timeout=self.request_timeout,
            )
            self._user_uuid = self._parse_user_uuid(self._check(resp, "getUserInfo"))
        return self._user_uuid

    def upload_media(self, buffer: bytes, filename: str) -> str:
        user_uuid = self.resolve_user_uuid()
        path = build_upload_path(user_uuid, hashlib.sha1(buffer).hexdigest(), filename)
        init = self._check(
            self.sync_client.post(
                url=self._bridge_url("init"),
                json={"path": path, "fileSize": len(buffer), "partSize": BRIDGE_PART_SIZE},
                headers=build_bridge_headers(self.token),
                timeout=self.request_timeout,
            ),
            "upload/init",
        )
        data = init.get("data") or {}
        for i, part in enumerate(data.get("parts") or []):
            chunk = buffer[i * BRIDGE_PART_SIZE : (i + 1) * BRIDGE_PART_SIZE]
            put_status = self._put_bytes(part["url"], chunk)
            if put_status not in (200, 204):
                raise LibTVError(
                    status_code=put_status, message=f"libtv upload part {part.get('partNumber')} failed"
                )
        complete = self._check(
            self.sync_client.post(
                url=self._bridge_url("complete"),
                json={"path": path, "uploadId": data.get("uploadId")},
                headers=build_bridge_headers(self.token),
                timeout=self.request_timeout,
            ),
            "upload/complete",
        )
        return parse_upload_url(complete)

    async def aresolve_user_uuid(self) -> str:
        if self._user_uuid is None:
            assert self.async_client is not None, "async_client required"
            resp = await self.async_client.post(
                url=f"{LIBTV_PASSPORT_BASE}/api/www/user/getUserInfo",
                json={},
                headers=self.headers,
                timeout=self.request_timeout,
            )
            self._user_uuid = self._parse_user_uuid(self._check(resp, "getUserInfo"))
        return self._user_uuid

    async def aupload_media(self, buffer: bytes, filename: str) -> str:
        user_uuid = await self.aresolve_user_uuid()
        path = build_upload_path(user_uuid, hashlib.sha1(buffer).hexdigest(), filename)
        init = self._check(
            await self.async_client.post(
                url=self._bridge_url("init"),
                json={"path": path, "fileSize": len(buffer), "partSize": BRIDGE_PART_SIZE},
                headers=build_bridge_headers(self.token),
                timeout=self.request_timeout,
            ),
            "upload/init",
        )
        data = init.get("data") or {}
        for i, part in enumerate(data.get("parts") or []):
            chunk = buffer[i * BRIDGE_PART_SIZE : (i + 1) * BRIDGE_PART_SIZE]
            put_status = await asyncio.to_thread(self._put_bytes, part["url"], chunk)
            if put_status not in (200, 204):
                raise LibTVError(
                    status_code=put_status, message=f"libtv upload part {part.get('partNumber')} failed"
                )
        complete = self._check(
            await self.async_client.post(
                url=self._bridge_url("complete"),
                json={"path": path, "uploadId": data.get("uploadId")},
                headers=build_bridge_headers(self.token),
                timeout=self.request_timeout,
            ),
            "upload/complete",
        )
        return parse_upload_url(complete)

    def _index_tool_spec(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for tool in (payload.get("data") or {}).get("tools") or []:
            meta = tool.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    continue
            if not isinstance(meta, dict):
                continue
            model_key = meta.get("modelKey")
            if model_key:
                index[str(model_key)] = {
                    "vendor": meta.get("modelVendor") or "",
                    "task_type": tool.get("type") or "",
                    "properties": meta.get("properties") or {},
                    "config": meta.get("config") or {},
                }
        return index

    def _lookup(self, index: Dict[str, Dict[str, Any]], model_key: str) -> Dict[str, Any]:
        spec = index.get(model_key)
        if spec is None:
            raise LibTVError(status_code=404, message=f"libtv model not found in tool_spec: {model_key}")
        return spec

    def resolve_model_spec(self, model_key: str) -> Dict[str, Any]:
        if self._tool_spec_cache is None:
            assert self.sync_client is not None, "sync_client required"
            resp = self.sync_client.get(url=f"{self.api_base}/api/tool_spec/list", headers=self.headers)
            self._tool_spec_cache = self._index_tool_spec(self._check(resp, "tool_spec/list"))
        return self._lookup(self._tool_spec_cache, model_key)

    async def aresolve_model_spec(self, model_key: str) -> Dict[str, Any]:
        if self._tool_spec_cache is None:
            assert self.async_client is not None, "async_client required"
            resp = await self.async_client.get(url=f"{self.api_base}/api/tool_spec/list", headers=self.headers)
            self._tool_spec_cache = self._index_tool_spec(self._check(resp, "tool_spec/list"))
        return self._lookup(self._tool_spec_cache, model_key)

    def _check(self, response: Any, step: str) -> Dict[str, Any]:
        if response.status_code != 200:
            raise LibTVError(
                status_code=response.status_code,
                message=f"libtv {step} HTTP {response.status_code}: {response.text[:300]}",
            )
        payload = response.json()
        if payload.get("code") not in (0, None):
            raise LibTVError(
                status_code=502, message=f"libtv {step} code={payload.get('code')} msg={payload.get('msg')}"
            )
        return payload

    # ---------- sync ----------
    def _post(self, path: str, body: Dict[str, Any], step: str) -> Dict[str, Any]:
        assert self.sync_client is not None, "sync_client required for sync calls"
        resp = self.sync_client.post(
            url=f"{self.api_base}{path}", json=body, headers=self.headers, timeout=self.request_timeout
        )
        return self._check(resp, step)

    def generate(
        self,
        model_key: str,
        vendor: str,
        task_type: str,
        params: Dict[str, Any],
        project_name: str,
    ) -> Dict[str, Any]:
        project = self._post("/api/canvas/project/create", {"name": project_name}, "project/create")
        meta = parse_project(project)
        project_uuid, team_id = meta["project_uuid"], meta["team_id"]
        node_key = str(uuid.uuid4())
        self._post(
            "/api/canvas/nodes/batch",
            build_node_batch_body(project_uuid, task_type, node_key, NODE_DEFAULT_NAME[task_type], model_key, params),
            "nodes/batch",
        )
        created = self._post(
            "/api/task/generation/create",
            build_generation_body(model_key, vendor, task_type, params, node_key, project_uuid, team_id),
            "generation/create",
        )
        task_id = parse_task_id(created)
        for _ in range(self.poll_max_attempts):
            progress = self._post("/api/task/generation/progress", {"taskIds": [task_id]}, "generation/progress")
            state = parse_progress(progress, task_type)
            if state["status"] == 2:
                return {"urls": state["urls"], "task_id": task_id, "project_uuid": project_uuid, "node_key": node_key}
            if state["status"] == 3:
                raise LibTVError(status_code=502, message=f"libtv generation failed: {state['failed_reason']}")
            time.sleep(self.poll_interval)
        raise LibTVError(status_code=504, message=f"libtv generation poll timeout (task {task_id})")

    def _fetch_bytes(self, url: str) -> bytes:
        # Bare httpx, not the litellm http client wrappers: the async wrapper injects
        # default headers that break the presigned object-store signature (it returns
        # 403 for a url the sync client fetches fine). Restrict to http(s) so a
        # reference url cannot read local files.
        if self._http_get is not None:
            return self._http_get(url)
        if not (url.startswith("http://") or url.startswith("https://")):
            raise LibTVError(status_code=400, message="libtv reference url must be http(s)")
        resp = httpx.get(url, follow_redirects=True, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise LibTVError(status_code=resp.status_code, message=f"libtv reference fetch HTTP {resp.status_code}")
        return resp.content

    def _libtv_cdn_url(self, kind: str, url: str, data: Optional[bytes]) -> str:
        if kind == "url":
            if "libtv-res.liblib.art" in url:
                return url
            return self.upload_media(self._fetch_bytes(url), "reference.png")
        return self.upload_media(data or b"", url or "reference.png")

    def resolve_compliant_image_refs(self, refs: List[tuple]) -> List[str]:
        """For an auto-compliance (portrait-capable) model: ensure each image reference
        clears libtv moderation before generation. Returns one libtv reference per input:
        ``asset://<assetId>`` for images the backend issues a verified asset id for
        (portraits), or the libtv cdn url for images it exempts (no real person). Raises
        if any reference fails moderation so the caller can fall back to another provider."""
        cdn_urls = [self._libtv_cdn_url(kind, url, data) for (kind, url, data) in refs]
        passed = parse_verify_passed(self._post("/api/community/image/verify", {"urlList": cdn_urls}, "image/verify"))
        resolved: List[str] = []
        for cdn_url in cdn_urls:
            if not passed.get(cdn_url):
                raise LibTVError(
                    status_code=400,
                    message="libtv portrait compliance check did not pass for a reference image",
                )
            asset_uuid = parse_third_asset_uuid(
                self._post(
                    "/api/third_asset/create",
                    {"assetUrl": cdn_url, "assetType": "image", "version": 1},
                    "third_asset/create",
                )
            )
            ref = None
            for _ in range(THIRD_ASSET_POLL_ATTEMPTS):
                ref = _asset_ref_from_item(
                    parse_third_asset_item(
                        self._post("/api/third_asset/check", {"uuids": [asset_uuid]}, "third_asset/check"),
                        asset_uuid,
                    ),
                    cdn_url,
                )
                if ref:
                    break
                time.sleep(self.poll_interval)
            resolved.append(ref or cdn_url)
        return resolved

    # ---------- async ----------
    async def _apost(self, path: str, body: Dict[str, Any], step: str) -> Dict[str, Any]:
        assert self.async_client is not None, "async_client required for async calls"
        resp = await self.async_client.post(
            url=f"{self.api_base}{path}", json=body, headers=self.headers, timeout=self.request_timeout
        )
        return self._check(resp, step)

    async def agenerate(
        self,
        model_key: str,
        vendor: str,
        task_type: str,
        params: Dict[str, Any],
        project_name: str,
    ) -> Dict[str, Any]:
        project = await self._apost("/api/canvas/project/create", {"name": project_name}, "project/create")
        meta = parse_project(project)
        project_uuid, team_id = meta["project_uuid"], meta["team_id"]
        node_key = str(uuid.uuid4())
        await self._apost(
            "/api/canvas/nodes/batch",
            build_node_batch_body(project_uuid, task_type, node_key, NODE_DEFAULT_NAME[task_type], model_key, params),
            "nodes/batch",
        )
        created = await self._apost(
            "/api/task/generation/create",
            build_generation_body(model_key, vendor, task_type, params, node_key, project_uuid, team_id),
            "generation/create",
        )
        task_id = parse_task_id(created)
        for _ in range(self.poll_max_attempts):
            progress = await self._apost("/api/task/generation/progress", {"taskIds": [task_id]}, "generation/progress")
            state = parse_progress(progress, task_type)
            if state["status"] == 2:
                return {"urls": state["urls"], "task_id": task_id, "project_uuid": project_uuid, "node_key": node_key}
            if state["status"] == 3:
                raise LibTVError(status_code=502, message=f"libtv generation failed: {state['failed_reason']}")
            await asyncio.sleep(self.poll_interval)
        raise LibTVError(status_code=504, message=f"libtv generation poll timeout (task {task_id})")

    async def _afetch_bytes(self, url: str) -> bytes:
        return await asyncio.to_thread(self._fetch_bytes, url)

    def _put_bytes(self, url: str, data: bytes) -> int:
        # See _fetch_bytes: the presigned PUT must carry no extra headers (raw body
        # only) or the object store rejects the signature, so use bare httpx with
        # content= (sends no Content-Type) rather than the litellm wrappers.
        if self._http_put is not None:
            return self._http_put(url, data)
        if not (url.startswith("http://") or url.startswith("https://")):
            raise LibTVError(status_code=400, message="libtv upload url must be http(s)")
        return httpx.put(url, content=data, timeout=self.request_timeout).status_code

    async def _alibtv_cdn_url(self, kind: str, url: str, data: Optional[bytes]) -> str:
        if kind == "url":
            if "libtv-res.liblib.art" in url:
                return url
            return await self.aupload_media(await self._afetch_bytes(url), "reference.png")
        return await self.aupload_media(data or b"", url or "reference.png")

    async def aresolve_compliant_image_refs(self, refs: List[tuple]) -> List[str]:
        cdn_urls = [await self._alibtv_cdn_url(kind, url, data) for (kind, url, data) in refs]
        passed = parse_verify_passed(
            await self._apost("/api/community/image/verify", {"urlList": cdn_urls}, "image/verify")
        )
        resolved: List[str] = []
        for cdn_url in cdn_urls:
            if not passed.get(cdn_url):
                raise LibTVError(
                    status_code=400,
                    message="libtv portrait compliance check did not pass for a reference image",
                )
            asset_uuid = parse_third_asset_uuid(
                await self._apost(
                    "/api/third_asset/create",
                    {"assetUrl": cdn_url, "assetType": "image", "version": 1},
                    "third_asset/create",
                )
            )
            ref = None
            for _ in range(THIRD_ASSET_POLL_ATTEMPTS):
                ref = _asset_ref_from_item(
                    parse_third_asset_item(
                        await self._apost("/api/third_asset/check", {"uuids": [asset_uuid]}, "third_asset/check"),
                        asset_uuid,
                    ),
                    cdn_url,
                )
                if ref:
                    break
                await asyncio.sleep(self.poll_interval)
            resolved.append(ref or cdn_url)
        return resolved
