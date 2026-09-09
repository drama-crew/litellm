"""LibTV canvas state is a graph, not the flattened generation HTTP body."""

import json
import uuid
from copy import deepcopy
from typing import Any

from .common import NODE_ACTION, NODE_TYPE_BACKEND, LibTVError
from .transform import _allowed_setting_keys, advanced_setting_keys


_SETTING_KEYS = {"ratio", "resolution", "duration", "quality", "enableSound", "smartStoryboard"}
_ADVANCED_KEYS = {"search_enabled", "searchEnabled", "autoCompliance"}


def canvas_params(model: str, params: dict, spec: dict | None) -> dict:
    result = deepcopy(params)
    result["model"] = model
    properties = (spec or {}).get("properties") or {}
    mode = params.get("modeType", "text2video")
    groups = (
        ("settings", _allowed_setting_keys(spec, mode) if spec else _SETTING_KEYS),
        ("advancedSettings", advanced_setting_keys(spec, mode) if spec else _ADVANCED_KEYS),
    )
    for group, fields in groups:
        nested = dict(result.pop(group, {}) or {})
        for field in fields:
            prop = properties.get(field) or {}
            key = prop.get("originalField") or field
            if key in result:
                nested[key] = result.pop(key)
            elif field in result:
                nested[key] = result.pop(field)
            elif key not in nested and prop.get("default") is not None:
                nested[key] = prop["default"]
        result[group] = nested
    return result


def build_canvas_batch(
    project_uuid: str,
    node_kind: str,
    node_key: str,
    name: str,
    model_key: str,
    params: dict,
    spec: dict | None = None,
    asset_refs: dict | None = None,
) -> dict:
    """Materialize each reference slot and its edge before generation is submitted.

    A separate node per slot intentionally preserves first/last ordering even when
    the same image occupies both slots. Compliance IDs never replace display URLs.
    """
    p = canvas_params(model_key, params, spec)
    nodes: list[dict[str, Any]] = []
    connections = []

    def node(key: str, kind: str, label: str, data: dict, x: int, y: int) -> dict:
        return {
            "nodeKey": key,
            "projectUuid": project_uuid,
            "type": ({**NODE_TYPE_BACKEND, "audio": 4})[kind],
            "name": label,
            "position": {"positionX": str(x), "positionY": str(y)},
            "parentKey": "",
            "data": json.dumps(data, ensure_ascii=False),
        }

    # The video handler keeps the proven upstream wire shapes. Recover each
    # display URL from the compliance result saved by this request's client.
    def resource(value: Any, kind: str) -> dict:
        ref = dict(value) if isinstance(value, dict) else {"url": value}
        wire_url = ref.get("url")
        remembered = (asset_refs or {}).get((kind, wire_url)) or {}
        ref = {**ref, **remembered}
        url = ref.get("url")
        if not isinstance(url, str) or not url or url.startswith("asset://"):
            raise LibTVError(status_code=400, message="libtv canvas reference requires its original display URL")
        key = str(uuid.uuid4())
        label = f"参考素材 {len(nodes) + 1}"
        data: dict[str, Any] = {
            "type": kind,
            "name": label,
            "url": [url],
            "action": f"{kind}_resource",
            "generatorType": "default",
        }
        item = {"nodeId": key, "url": url}
        asset_id = ref.get("assetId")
        if asset_id:
            data[{"image": "portraitAssetId", "video": "assetVideoAssetId", "audio": "assetAudioAssetId"}[kind]] = (
                asset_id
            )
            item["assetId"] = asset_id
        elif ref.get("compliantExempt") is True:
            data["portraitCompliantExempt"] = True
            item["compliantExempt"] = True
        if kind == "video":
            data["poster"] = ""
        nodes.append(node(key, kind, label, data, 0, len(nodes) * 400))
        connections.append(
            {
                "projectUuid": project_uuid,
                "connectionId": str(uuid.uuid4()),
                "source": key,
                "target": node_key,
                "sourceHandle": "source",
                "targetHandle": "target",
                "type": "default",
                "deletable": True,
                "selectable": True,
            }
        )
        return item

    mixed = params.get("mixedList") if params.get("modeType") == "mixed2video" else None
    if mixed:
        for kind in ("image", "video", "audio"):
            p[f"{kind}List"] = []
        p["mixedList"] = []
        for value in mixed:
            kind = value.get("type") or value.get("mediaType") or "image"
            item = resource(value, kind)
            p[f"{kind}List"].append(item)
            p["mixedList"].append({**item, "mediaType": kind})
        p["mixedListOrder"] = [item["nodeId"] for item in p["mixedList"]]
    else:
        for kind in ("image", "video", "audio"):
            p[f"{kind}List"] = [resource(value, kind) for value in params.get(f"{kind}List", [])]
        if nodes:
            p["mixedListOrder"] = [n["nodeKey"] for n in nodes]
    for kind in ("image", "video", "audio"):
        if p[f"{kind}List"]:
            p[f"{kind}ListOrder"] = [item["nodeId"] for item in p[f"{kind}List"]]
    data = {
        "type": node_kind,
        "name": name,
        "url": [],
        "action": NODE_ACTION[node_kind],
        "generatorType": "default",
        "params": p,
    }
    if node_kind == "video":
        data["poster"] = ""
    generation_node = node(node_key, node_kind, name, data, 600 if nodes else 0, 0)
    return {
        "projectUuid": project_uuid,
        "nodes": {"create": [generation_node, *nodes]},
        "connections": {"create": connections} if connections else {},
    }


def generation_params(model: str, params: dict) -> dict:
    """Flatten only configuration groups; retain existing model reference encoding."""
    result = {k: v for k, v in params.items() if k not in ("settings", "advancedSettings")}
    result.update(params.get("settings") or {})
    result.update(params.get("advancedSettings") or {})
    result["model"] = model
    return result


def node_with_task(node: dict, task_id: str, state: dict | None = None) -> dict:
    result = deepcopy(node)
    data = json.loads(result["data"])
    state = state or {"status": 1, "urls": []}
    status = state.get("status")
    data["taskInfo"] = {
        "taskId": task_id,
        "loading": status not in (2, 3),
        "status": status,
        "progressPercent": state.get("progress_percent", 100 if status == 2 else 0),
    }
    if state.get("failed_reason"):
        data["taskInfo"]["failedReason"] = state["failed_reason"]
    if status == 2:
        data["url"] = state.get("urls") or []
        if state.get("poster"):
            data["poster"] = state["poster"]
    result["data"] = json.dumps(data, ensure_ascii=False)
    return result
