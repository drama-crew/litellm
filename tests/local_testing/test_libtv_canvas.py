"""Regressions captured with the official LibTV CLI 1.1.3 and browser canvas."""

import copy
import json

import pytest

from litellm.llms.libtv.canvas import build_canvas_batch, node_with_task
from litellm.llms.libtv.client import LibTVClient, build_generation_body
from litellm.llms.libtv.transform import build_generation_params


SPEC = {
    "model_key": "star-video2.5",
    "properties": {
        "ratio": {"default": "adaptive"},
        "resolution": {"default": "720p"},
        "duration": {"default": 5},
        "enableSound": {"default": "on"},
        "searchEnabled": {"originalField": "search_enabled", "default": 1},
        "autoCompliance": {"default": 1},
    },
    "config": {
        "settings": {"frames2video": ["ratio", "resolution", "duration", "enableSound"]},
        "advancedSettings": ["searchEnabled", "autoCompliance"],
    },
}


def requested_params():
    params = build_generation_params(
        "original prompt",
        {
            "ratio": "16:9",
            "resolution": "480p",
            "seconds": 5,
            "advancedSettings": {"search_enabled": 0, "autoCompliance": 1},
        },
        SPEC,
        "frames2video",
    )
    params["imageList"] = ["https://cdn.example/first.png"]
    return params


def test_cli_canvas_contract_keeps_frame_and_settings_after_graph_reconstruction():
    params = requested_params()
    before = copy.deepcopy(params)
    batch = build_canvas_batch("p", "video", "v", "video", "star-video2.5", params, SPEC)
    nodes = {n["nodeKey"]: json.loads(n["data"]) for n in batch["nodes"]["create"]}
    video = nodes["v"]["params"]
    # The UI reconstructs references from incoming nodes, not the wire imageList.
    incoming = [nodes[e["source"]] for e in batch["connections"]["create"] if e["target"] == "v"]
    assert [n["url"][0] for n in incoming] == [ref["url"] for ref in video["imageList"]] == params["imageList"]
    assert incoming[0]["action"] == "image_resource"
    assert video["modeType"] == "frames2video"
    assert video["settings"] == {"ratio": "16:9", "resolution": "480p", "duration": 5, "enableSound": "on"}
    assert video["advancedSettings"] == {"search_enabled": 0, "autoCompliance": 1}
    assert video["imageListOrder"] == video["mixedListOrder"] == [video["imageList"][0]["nodeId"]]
    assert "ratio" not in video and "search_enabled" not in video
    assert params == before
    wire = build_generation_body("star-video2.5", "seedance2.5", "video", params, "v", "p")["params"]
    assert wire["model"] == "star-video2.5" and wire["resolution"] == "480p" and wire["search_enabled"] == 0
    assert wire["imageList"] == params["imageList"]
    assert not {"settings", "advancedSettings", "imageListOrder", "mixedListOrder"}.intersection(wire)


@pytest.mark.parametrize("same_image", [False, True])
def test_two_frame_slots_preserve_order_and_duplicates(same_image):
    params = requested_params()
    params["imageList"].append(params["imageList"][0] if same_image else "https://cdn.example/last.png")
    batch = build_canvas_batch("p", "video", "v", "video", "star-video2.5", params, SPEC)
    data = json.loads(batch["nodes"]["create"][0]["data"])["params"]
    assert [item["url"] for item in data["imageList"]] == params["imageList"]
    assert len(set(data["imageListOrder"])) == 2
    assert data["imageListOrder"] == [edge["source"] for edge in batch["connections"]["create"]]


def test_registered_reference_retains_display_url_and_asset_id():
    params = requested_params()
    params["imageList"] = ["asset://verified"]
    batch = build_canvas_batch(
        "p",
        "video",
        "v",
        "video",
        "star-video2.5",
        params,
        SPEC,
        {("image", "asset://verified"): {"url": "https://cdn.example/person.png", "assetId": "verified"}},
    )
    video, image = [json.loads(n["data"]) for n in batch["nodes"]["create"]]
    assert image["url"] == ["https://cdn.example/person.png"]
    assert image["portraitAssetId"] == "verified"
    assert video["params"]["imageList"][0]["assetId"] == "verified"
    assert build_generation_body("m", "v", "video", params, "n", "p")["params"]["imageList"] == ["asset://verified"]


def test_mixed_references_have_one_node_per_slot_in_wire_order():
    params = {
        **requested_params(),
        "modeType": "mixed2video",
        "mixedList": [
            {"url": "https://cdn.example/v.mp4", "type": "video"},
            {"url": "https://cdn.example/i.png", "type": "image"},
            {"url": "https://cdn.example/a.mp3", "type": "audio"},
        ],
    }
    batch = build_canvas_batch("p", "video", "v", "video", "star-video2.5", params)
    data = json.loads(batch["nodes"]["create"][0]["data"])["params"]
    assert [x["url"] for x in data["mixedList"]] == [x["url"] for x in params["mixedList"]]
    assert [x["mediaType"] for x in data["mixedList"]] == ["video", "image", "audio"]
    assert len(batch["nodes"]["create"]) == 4
    assert len(data["imageList"]) == len(data["videoList"]) == len(data["audioList"]) == 1


@pytest.mark.parametrize("settings", [["ratio", "resolution", "duration", "enableSound"], SPEC["config"]["settings"]])
def test_schema_defaults_and_original_fields_for_both_model_schema_shapes(settings):
    spec = {**SPEC, "config": {**SPEC["config"], "settings": settings}}
    assert build_generation_params("x", {}, spec, "frames2video")["search_enabled"] == 1
    for op in ({"search_enabled": 0}, {"advancedSettings": {"searchEnabled": 0}}):
        result = build_generation_params("x", op, spec, "frames2video")
        assert result["search_enabled"] == 0
        assert "searchEnabled" not in result


class Response:
    status_code = 200
    headers = {}

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


class CanvasServer:
    def __init__(self):
        self.nodes = {}
        self.creates = 0
        self.fail_sync = False

    async def post(self, url, json=None, **kwargs):
        if url.endswith("project/create"):
            return Response({"code": 0, "data": {"projectMeta": {"uuid": "p"}}})
        if url.endswith("nodes/batch"):
            if json["nodes"].get("update") and self.fail_sync:
                raise RuntimeError("canvas unavailable")
            for node in json["nodes"].get("create", []) + json["nodes"].get("update", []):
                self.nodes[node["nodeKey"]] = copy.deepcopy(node)
            return Response({"code": 0})
        if url.endswith("generation/create"):
            self.creates += 1
            return Response({"code": 0, "data": {"taskId": "task"}})
        if url.endswith("generation/progress"):
            return Response(
                {
                    "code": 0,
                    "data": {
                        "progresses": [
                            {
                                "taskId": "task",
                                "status": 2,
                                "taskResult": {"videos": [{"videoUrl": "https://cdn.example/result.mp4"}]},
                            }
                        ]
                    },
                }
            )
        raise AssertionError(url)

    post_once = post

    async def get(self, url, **kwargs):
        assert url.endswith("project/detail")
        return Response({"code": 0, "data": {"nodeList": list(self.nodes.values())}})


class MemoryPersistence:
    def __init__(self):
        self.locations = {}

    async def store_canvas_task(self, account, task_id, project_uuid, node_key):
        self.locations[account, task_id] = {"project_uuid": project_uuid, "node_key": node_key}

    async def canvas_task(self, account, task_id):
        return self.locations.get((account, task_id))


@pytest.mark.asyncio
async def test_poll_from_new_client_recovers_canvas_and_preserves_user_edits(monkeypatch):
    monkeypatch.setenv("LIBTV_PROJECT_REUSE_DISABLED", "1")
    server, persistence = CanvasServer(), MemoryPersistence()
    submit = LibTVClient("t", "w", async_client=server, persistence=persistence)
    created = await submit.acreate("star-video2.5", "v", "video", requested_params(), "test", paid_submission=True)
    key = created["node_key"]
    data = json.loads(server.nodes[key]["data"])
    assert data["taskInfo"]["taskId"] == "task" and data["taskInfo"]["loading"]
    data["params"]["prompt"] = "user edited prompt"
    server.nodes[key]["data"] = json.dumps(data)
    server.nodes[key]["position"]["positionX"] = "999"
    poll = LibTVClient("t", "w", async_client=server, persistence=persistence)
    state = await poll.apoll_once("task", "video")
    node = server.nodes[key]
    data = json.loads(node["data"])
    assert state["status"] == 2 and data["taskInfo"]["status"] == 2
    assert data["url"] == ["https://cdn.example/result.mp4"]
    assert data["params"]["prompt"] == "user edited prompt" and node["position"]["positionX"] == "999"
    assert server.creates == 1


@pytest.mark.asyncio
async def test_post_submit_canvas_failure_retains_receipt_without_resubmission(monkeypatch):
    monkeypatch.setenv("LIBTV_PROJECT_REUSE_DISABLED", "1")
    server = CanvasServer()
    server.fail_sync = True
    client = LibTVClient("t", "w", async_client=server)
    created = await client.acreate("m", "v", "video", requested_params(), "test", paid_submission=True)
    assert created["task_id"] == "task" and server.creates == 1


@pytest.mark.parametrize("deleted", [False, True])
def test_old_poll_does_not_recreate_deleted_node_or_replace_new_task(deleted):
    node = build_canvas_batch("p", "video", "v", "video", "m", {})["nodes"]["create"][0]
    node = node_with_task(node, "newer-task")
    detail = {"data": {"nodeList": [] if deleted else [node]}}
    assert LibTVClient._current_canvas_node(detail, "v", "old-task") is None


@pytest.mark.parametrize("ratio", ["adaptive", "16:9"])
def test_ratio_survives_api_size_and_canvas_serialization(ratio):
    spec = copy.deepcopy(SPEC)
    spec["properties"]["ratio"]["enum"] = ["adaptive", "16:9"]
    params = build_generation_params("x", {"aspect_ratio": ratio, "size": "854x480"}, spec, "frames2video")
    assert params["ratio"] == ratio
    batch = build_canvas_batch("p", "video", "v", "v", "star-video2.5", params, spec)
    assert json.loads(batch["nodes"]["create"][0]["data"])["params"]["settings"]["ratio"] == ratio
    assert build_generation_body("star-video2.5", "v", "video", params, "v", "p")["params"]["ratio"] == ratio


def test_structured_invalid_params_never_retries_generic_message():
    from litellm.llms.libtv.client import parse_progress
    from litellm.llms.libtv.handler import _is_fresh_asset_aging_failure

    state = parse_progress(
        {
            "data": {
                "progresses": [
                    {
                        "status": 3,
                        "failedCategory": "INVALID_PARAMS",
                        "failedReason": "视频生成失败，请稍后重试",
                        "progressPercent": 17,
                    }
                ]
            }
        },
        "video",
    )
    assert not _is_fresh_asset_aging_failure(state)
    node = build_canvas_batch("p", "video", "v", "v", "m", {})["nodes"]["create"][0]
    assert json.loads(node_with_task(node, "t", state)["data"])["taskInfo"]["progressPercent"] == 17


def test_canvas_update_keeps_optimistic_version():
    body = LibTVClient._canvas_update_body({"projectUuid": "p", "nodeKey": "v"}, 123)
    assert body["version"] == 123
