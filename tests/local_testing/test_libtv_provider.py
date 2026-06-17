import json

import pytest

from litellm.llms.libtv.client import (
    LibTVClient,
    build_generation_body,
    build_node_batch_body,
    parse_progress,
    parse_task_id,
)
from litellm.llms.libtv.common import LibTVError, build_libtv_headers
from litellm.llms.libtv.transform import build_generation_params, size_to_ratio

_SEEDANCE_SPEC = {
    "vendor": "seedance2.0",
    "config": {"settings": ["ratio", "resolution", "duration", "enableSound"]},
    "properties": {
        "ratio": {"default": "16:9", "enum": ["16:9", "9:16", "1:1"]},
        "resolution": {"default": "720p", "enum": [{"value": "480p"}, {"value": "720p"}]},
        "duration": {"default": 5},
    },
}
_WAN_SPEC = {
    "vendor": "Wan",
    "config": {"settings": {"text2video": ["ratio", "resolution", "duration"]}},
    "properties": {
        "ratio": {"default": "16:9"},
        "resolution": {"default": "720p", "enum": [{"value": "480P"}, {"value": "720P"}]},
        "duration": {"default": 5},
    },
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSyncClient:
    def __init__(self, post_by_path=None, get_payload=None):
        self.post_by_path = post_by_path or {}
        self.get_payload = get_payload
        self.calls = []

    def _path(self, url):
        return url.split("api.liblib.tv", 1)[-1]

    def post(self, url, json=None, headers=None, timeout=None):
        path = self._path(url)
        self.calls.append((path, json))
        queue = self.post_by_path[path]
        return FakeResponse(queue.pop(0) if isinstance(queue, list) else queue)

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append((self._path(url), None))
        return FakeResponse(self.get_payload)


class FakeAsyncClient:
    def __init__(self, post_by_path=None, get_payload=None):
        self.post_by_path = post_by_path or {}
        self.get_payload = get_payload
        self.calls = []

    def _path(self, url):
        return url.split("api.liblib.tv", 1)[-1]

    async def post(self, url, json=None, headers=None, timeout=None):
        path = self._path(url)
        self.calls.append((path, json))
        queue = self.post_by_path[path]
        return FakeResponse(queue.pop(0) if isinstance(queue, list) else queue)

    async def get(self, url, headers=None, params=None):
        self.calls.append((self._path(url), None))
        return FakeResponse(self.get_payload)


def test_headers_carry_token_and_webid():
    h = build_libtv_headers("tok", "wid")
    assert h["token"] == "tok"
    assert h["webid"] == "wid"
    assert h["x-language"] == "zh"
    assert h["X-from-client"] == "cli"


def test_size_to_ratio():
    assert size_to_ratio("1280x720") == "16:9"
    assert size_to_ratio("1080x1920") == "9:16"
    assert size_to_ratio("1024x1024") == "1:1"
    assert size_to_ratio(None) is None
    assert size_to_ratio("not-a-size") is None


def test_build_params_flat_settings_seedance():
    params = build_generation_params("a cat", {"size": "1280x720", "seconds": "8"}, _SEEDANCE_SPEC, "text2video")
    assert params["prompt"] == "a cat"
    assert params["modeType"] == "text2video"
    assert params["count"] == 1
    assert params["settings"]["ratio"] == "16:9"
    assert params["settings"]["resolution"] == "720p"
    assert params["settings"]["duration"] == 8


def test_build_params_filters_to_mode_keys_and_fills_defaults():
    # Wan text2video declares only ratio/resolution/duration; enableSound must NOT appear
    params = build_generation_params("a cat", {"enableSound": "off"}, _WAN_SPEC, "text2video")
    assert set(params["settings"].keys()) == {"ratio", "resolution", "duration"}
    assert "enableSound" not in params["settings"]
    assert params["settings"]["ratio"] == "16:9"  # schema default
    assert params["settings"]["duration"] == 5  # schema default


def test_build_params_coerces_resolution_enum_case():
    # user/size implies 480p; Wan enum is uppercase 480P -> must coerce to schema casing
    params = build_generation_params("x", {"size": "854x480"}, _WAN_SPEC, "text2video")
    assert params["settings"]["resolution"] == "480P"


def test_build_params_unknown_mode_has_empty_settings():
    params = build_generation_params("x", {}, _WAN_SPEC, "image2video")
    assert params["settings"] == {}


def test_build_node_batch_body_video_carries_model_and_params():
    body = build_node_batch_body(
        "proj-uuid", "video", "nk-1", "视频节点", "wanxiang-plus", {"prompt": "p", "settings": {"ratio": "16:9"}}
    )
    node = body["nodes"]["create"][0]
    assert body["projectUuid"] == "proj-uuid"
    assert node["type"] == 3
    assert node["nodeKey"] == "nk-1"
    data = json.loads(node["data"])
    assert data["action"] == "video_generate"
    assert data["type"] == "video"
    assert data["poster"] == ""
    # node must carry the model + generation params (not a hollow placeholder)
    assert data["params"]["model"] == "wanxiang-plus"
    assert data["params"]["prompt"] == "p"
    assert data["params"]["settings"] == {"ratio": "16:9"}


def test_build_node_batch_body_image_has_no_poster():
    node = build_node_batch_body("p", "image", "nk", "图片节点", "nebula-ultra", {"prompt": "p"})["nodes"]["create"][0]
    assert node["type"] == 2
    assert "poster" not in json.loads(node["data"])
    assert json.loads(node["data"])["params"]["model"] == "nebula-ultra"


def test_build_generation_body_shape_and_team_id():
    body = build_generation_body("seedance2.0", "seedance2.0", "video", {"prompt": "x"}, "nk", "proj")
    assert body["model"] == "seedance2.0"
    assert body["provider"] == "seedance2.0"
    assert body["taskType"] == "video"
    assert body["metadata"] == {"node_id": "nk", "project_id": "proj"}
    assert body["params"] == {"prompt": "x"}
    assert body["requestId"]
    assert "teamId" not in body  # personal project: omitted
    team_body = build_generation_body("m", "v", "video", {}, "nk", "proj", team_id=42)
    assert team_body["teamId"] == 42


def test_parse_project_extracts_uuid_and_team():
    from litellm.llms.libtv.client import parse_project

    assert parse_project({"data": {"projectMeta": {"uuid": "u1", "teamId": 7}}}) == {"project_uuid": "u1", "team_id": 7}
    assert parse_project({"data": {"projectMeta": {"uuid": "u2"}}}) == {"project_uuid": "u2", "team_id": None}
    with pytest.raises(LibTVError):
        parse_project({"data": {"projectMeta": {}}})


def test_parse_task_id_variants():
    assert parse_task_id({"data": {"taskId": "t1"}}) == "t1"
    assert parse_task_id({"data": {"task_id": "t2"}}) == "t2"
    with pytest.raises(LibTVError):
        parse_task_id({"data": {}})


def test_parse_progress_success_extracts_video_url():
    payload = {
        "data": {"progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/v.mp4"}]})}]}
    }
    state = parse_progress(payload, "video")
    assert state["status"] == 2
    assert state["urls"] == ["https://x/v.mp4"]


def test_parse_progress_video_falls_back_to_images():
    payload = {
        "data": {"progresses": [{"status": 2, "taskResult": json.dumps({"images": [{"url": "https://x/i.png"}]})}]}
    }
    assert parse_progress(payload, "video")["urls"] == ["https://x/i.png"]


def test_parse_progress_failed():
    payload = {"data": {"progresses": [{"status": 3, "failedReason": "blocked"}]}}
    state = parse_progress(payload, "video")
    assert state["status"] == 3
    assert state["failed_reason"] == "blocked"


def test_parse_progress_loading_has_no_urls():
    payload = {"data": {"progresses": [{"status": 1}]}}
    assert parse_progress(payload, "video") == {"status": 1, "urls": [], "failed_reason": None}


def test_generate_orchestrates_full_sequence():
    client = FakeSyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "proj-9", "teamId": 55}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {"taskId": "task-9"}},
            "/api/task/generation/progress": [
                {"code": 0, "data": {"progresses": [{"status": 1}]}},
                {
                    "code": 0,
                    "data": {
                        "progresses": [
                            {"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/done.mp4"}]})}
                        ]
                    },
                },
            ],
        }
    )
    lt = LibTVClient(token="t", webid="w", sync_client=client, poll_interval=0)
    result = lt.generate(
        "seedance2.0", "seedance2.0", "video", {"prompt": "hi", "settings": {"ratio": "16:9"}}, "proj-name"
    )
    assert result["urls"] == ["https://x/done.mp4"]
    assert result["project_uuid"] == "proj-9"
    paths = [c[0] for c in client.calls]
    assert paths == [
        "/api/canvas/project/create",
        "/api/canvas/nodes/batch",
        "/api/task/generation/create",
        "/api/task/generation/progress",
        "/api/task/generation/progress",
    ]
    # node must be created carrying the model + params, not hollow
    node_data = json.loads(client.calls[1][1]["nodes"]["create"][0]["data"])
    assert node_data["params"]["model"] == "seedance2.0"
    assert node_data["params"]["prompt"] == "hi"
    gen_body = client.calls[2][1]
    assert gen_body["metadata"]["project_id"] == "proj-9"
    assert gen_body["metadata"]["node_id"] == result["node_key"]
    assert gen_body["teamId"] == 55  # teamId propagated from projectMeta


def test_generate_raises_on_failed_status():
    client = FakeSyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p"}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {"taskId": "t"}},
            "/api/task/generation/progress": [
                {"code": 0, "data": {"progresses": [{"status": 3, "failedReason": "nope"}]}}
            ],
        }
    )
    lt = LibTVClient(token="t", webid="w", sync_client=client, poll_interval=0)
    with pytest.raises(LibTVError):
        lt.generate("m", "v", "video", {"prompt": "x"}, "n")


def test_video_generation_routes_to_custom_provider():
    import litellm
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.videos.main import VideoObject

    captured = {}

    class StubLLM(CustomLLM):
        def video_generation(
            self, model, prompt, api_key, api_base, optional_params, logging_obj, timeout=None, client=None
        ):
            captured["model"] = model
            captured["prompt"] = prompt
            return VideoObject(id="vid-1", object="video", status="completed", model=model)

    litellm.custom_provider_map = [{"provider": "libtvstub", "custom_handler": StubLLM()}]
    try:
        out = litellm.video_generation(model="libtvstub/seedance2.0", prompt="a fox")
        assert isinstance(out, VideoObject)
        assert out.id == "vid-1"
        assert captured["model"] == "seedance2.0"
        assert captured["prompt"] == "a fox"
    finally:
        litellm.custom_provider_map = []


def test_video_generation_forwards_named_params_to_custom_handler():
    import litellm
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.videos.main import VideoObject

    captured = {}

    class StubLLM(CustomLLM):
        def video_generation(
            self, model, prompt, api_key, api_base, optional_params, logging_obj, timeout=None, client=None
        ):
            captured.update(optional_params)
            return VideoObject(id="v", object="video", status="completed", model=model)

    litellm.custom_provider_map = [{"provider": "libtvstub", "custom_handler": StubLLM()}]
    try:
        litellm.video_generation(model="libtvstub/seedance2.0", prompt="x", seconds="8", size="1280x720")
        assert captured.get("seconds") == "8"
        assert captured.get("size") == "1280x720"
    finally:
        litellm.custom_provider_map = []


@pytest.mark.asyncio
async def test_avideo_generation_routes_to_custom_provider():
    import litellm
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.videos.main import VideoObject

    class StubLLM(CustomLLM):
        async def avideo_generation(
            self, model, prompt, api_key, api_base, optional_params, logging_obj, timeout=None, client=None
        ):
            return VideoObject(id="async-vid", object="video", status="completed", model=model)

    litellm.custom_provider_map = [{"provider": "libtvstub", "custom_handler": StubLLM()}]
    try:
        out = await litellm.avideo_generation(model="libtvstub/seedance2.0", prompt="a fox")
        assert out.id == "async-vid"
    finally:
        litellm.custom_provider_map = []


def test_build_params_kling_style_keys():
    spec = {
        "config": {"settings": {"text2video": ["ratio_auto", "quality_high", "duration"]}},
        "properties": {
            "ratio_auto": {"default": "16:9"},
            "quality_high": {"default": "1080p"},
            "duration": {"default": 5},
        },
    }
    params = build_generation_params("x", {"size": "1280x720", "quality": "720p"}, spec, "text2video")
    assert params["settings"]["ratio_auto"] == "16:9"
    assert params["settings"]["quality_high"] == "720p"
    assert params["settings"]["duration"] == 5


def test_build_params_resolution_480_key():
    spec = {
        "config": {"settings": {"singleImage2video": ["resolution_480", "duration"]}},
        "properties": {"resolution_480": {"default": "480p"}, "duration": {"default": 5}},
    }
    params = build_generation_params("x", {"size": "854x480"}, spec, "singleImage2video")
    assert params["settings"]["resolution_480"] == "480p"


def test_count_list_property_does_not_crash():
    spec = {"config": {"settings": []}, "properties": {"count": [1, 2, 3, 4]}}
    params = build_generation_params("x", {}, spec, "text2video")
    assert params["count"] == 1


def test_resolve_credentials_requires_both(monkeypatch):
    for key in ("LIBTV_TOKEN", "LIBTV_CLI_USERTOKEN", "LIBTV_WEBID", "LIBTV_CLI_WEBID"):
        monkeypatch.delenv(key, raising=False)
    from litellm.llms.libtv.common import resolve_libtv_credentials

    assert resolve_libtv_credentials("tok", "wid") == ("tok", "wid")
    with pytest.raises(LibTVError):
        resolve_libtv_credentials(None, "wid")
    with pytest.raises(LibTVError):
        resolve_libtv_credentials("tok", None)


def test_check_raises_on_http_error_and_nonzero_code():
    lt = LibTVClient(token="t", webid="w", sync_client=FakeSyncClient())
    with pytest.raises(LibTVError):
        lt._check(FakeResponse({"msg": "boom"}, status_code=500), "step")
    with pytest.raises(LibTVError):
        lt._check(FakeResponse({"code": 1, "msg": "bad"}), "step")
    assert lt._check(FakeResponse({"code": 0, "data": {}}), "step")["code"] == 0


@pytest.mark.asyncio
async def test_agenerate_orchestrates_full_sequence():
    client = FakeAsyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p7"}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {"taskId": "tk7"}},
            "/api/task/generation/progress": [
                {"code": 0, "data": {"progresses": [{"status": 0}]}},
                {
                    "code": 0,
                    "data": {
                        "progresses": [
                            {"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/a.mp4"}]})}
                        ]
                    },
                },
            ],
        }
    )
    lt = LibTVClient(token="t", webid="w", async_client=client, poll_interval=0)
    result = await lt.agenerate("seedance2.0", "seedance2.0", "video", {"prompt": "hi"}, "proj")
    assert result["urls"] == ["https://x/a.mp4"]
    assert [c[0] for c in client.calls] == [
        "/api/canvas/project/create",
        "/api/canvas/nodes/batch",
        "/api/task/generation/create",
        "/api/task/generation/progress",
        "/api/task/generation/progress",
    ]


def test_resolve_model_spec_indexes_tool_spec():
    payload = {
        "code": 0,
        "data": {
            "tools": [
                {
                    "type": "video",
                    "metadata": json.dumps(
                        {"modelKey": "seedance2.0", "modelVendor": "seedance2.0", "modelName": "Seedance 2.0"}
                    ),
                },
                {"type": "image", "metadata": json.dumps({"modelKey": "nebula-ultra", "modelVendor": "nebula"})},
            ]
        },
    }
    lt = LibTVClient(token="t", webid="w", sync_client=FakeSyncClient(get_payload=payload))
    spec = lt.resolve_model_spec("seedance2.0")
    assert spec["vendor"] == "seedance2.0"
    assert spec["task_type"] == "video"
    with pytest.raises(LibTVError):
        lt.resolve_model_spec("does-not-exist")
