import json

import pytest

from litellm.llms.libtv.client import (
    LibTVClient,
    build_generation_body,
    build_node_batch_body,
    parse_progress,
    parse_task_id,
    parse_upload_url,
)
from litellm.llms.libtv.common import LibTVError, build_libtv_headers, build_upload_path
from litellm.llms.libtv.handler import (
    LIBTV_PROVIDER,
    LibTVLLM,
    _collect_reference_groups,
    _default_video_mode,
    _image_clarity_response_cost,
    _infer_image_mode,
    _infer_video_mode,
    _reference_payload,
)
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.videos.utils import decode_video_id_with_provider
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

# Real tool_spec metadata for nebula-ultra ("全能图片模型V2", libtv's disguised name for
# Nano Banana Pro). Flat settings list (no per-mode buckets) with count as a bare enum
# list rather than a dict, and modeType.items only advertising image2image.
_NEBULA_ULTRA_SPEC = {
    "modelKey": "nebula-ultra",
    "modelName": "全能图片模型V2",
    "modelVendor": "nebula",
    "vendor": "nebula",
    "baseType": 40,
    "properties": {
        "enableSlash": True,
        "cameraControl": True,
        "magic": True,
        "focus": True,
        "mention": True,
        "count": [1, 2, 4],
        "template": {"displayName": "风格", "maxCount": 1},
        "prompt": {"description": "", "placeholder": "", "maxLength": 0},
        "quality": {
            "displayName": "分辨率",
            "enum": ["1K", "2K", "4K"],
            "default": "2K",
            "component": "singleButton",
            "originalField": "quality",
        },
        "ratio": {
            "displayName": "比例",
            "enum": [
                {"value": "auto", "displayName": "自适应"},
                "1:1",
                "9:16",
                "16:9",
                "3:4",
                "4:3",
                "3:2",
                "2:3",
                "4:5",
                "5:4",
                "21:9",
            ],
            "default": "16:9",
            "component": "singleButton",
            "originalField": "ratio",
        },
        "searchable": {"displayName": "联网搜索", "default": 0, "component": "switch", "originalField": "searchable"},
        "modeType": {"description": "模态类型", "items": {"image2image": [0, 7]}},
    },
    "config": {"settings": ["quality", "ratio"], "advancedSettings": ["searchable"]},
    "rules": [{"require": ["prompt", "media"], "mode": "any"}, {"require": ["prompt"]}],
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
    assert params["ratio"] == "16:9"
    assert params["resolution"] == "720p"
    assert params["duration"] == 8


def test_build_params_reads_aspect_ratio_for_ratio():
    # The drama platform sends the chosen ratio as `aspect_ratio` (the wavespeed key);
    # libtv must honour it as `ratio` so a portrait source isn't forced to the 16:9
    # default. An explicit `ratio` still wins; `aspect_ratio` beats size-derived.
    params = build_generation_params("a cat", {"aspect_ratio": "9:16"}, _SEEDANCE_SPEC, "mixed2video")
    assert params["ratio"] == "9:16"
    params2 = build_generation_params(
        "a cat", {"aspect_ratio": "9:16", "size": "1280x720"}, _SEEDANCE_SPEC, "mixed2video"
    )
    assert params2["ratio"] == "9:16"  # aspect_ratio beats size-derived 16:9
    params3 = build_generation_params("a cat", {"ratio": "1:1", "aspect_ratio": "9:16"}, _SEEDANCE_SPEC, "mixed2video")
    assert params3["ratio"] == "1:1"  # explicit ratio still wins


def test_build_params_filters_to_mode_keys_and_fills_defaults():
    # Wan text2video declares only ratio/resolution/duration; enableSound must NOT appear
    params = build_generation_params("a cat", {"enableSound": "off"}, _WAN_SPEC, "text2video")
    assert params["ratio"] == "16:9"  # schema default, flattened to top level
    assert params["resolution"] == "720p"
    assert params["duration"] == 5
    assert "enableSound" not in params  # not declared for Wan text2video
    assert "settings" not in params  # settings are flattened, never nested


def test_build_params_coerces_resolution_enum_case():
    # user/size implies 480p; Wan enum is uppercase 480P -> must coerce to schema casing
    params = build_generation_params("x", {"size": "854x480"}, _WAN_SPEC, "text2video")
    assert params["resolution"] == "480P"


def test_build_params_unknown_mode_has_empty_settings():
    params = build_generation_params("x", {}, _WAN_SPEC, "image2video")
    assert "ratio" not in params and "resolution" not in params and "duration" not in params
    assert params["imageList"] == [] and params["textList"] == []


def test_build_params_nebula_ultra_coerces_lowercase_clarity_and_derives_ratio_from_size():
    # drama backend sends lowercase clarity tokens + a size, no explicit ratio.
    params = build_generation_params("a cat", {"quality": "2k", "size": "2560x1440"}, _NEBULA_ULTRA_SPEC, "text2image")
    assert params["quality"] == "2K"
    assert params["ratio"] == "16:9"


@pytest.mark.parametrize("lower,upper", [("1k", "1K"), ("2k", "2K"), ("4k", "4K")])
def test_build_params_nebula_ultra_quality_case_insensitive(lower, upper):
    params = build_generation_params("x", {"quality": lower}, _NEBULA_ULTRA_SPEC, "text2image")
    assert params["quality"] == upper


def test_build_params_nebula_ultra_explicit_aspect_ratio_wins_over_size():
    params = build_generation_params(
        "x", {"aspect_ratio": "9:16", "size": "2560x1440"}, _NEBULA_ULTRA_SPEC, "text2image"
    )
    assert params["ratio"] == "9:16"


def test_build_params_nebula_ultra_defaults_quality_when_absent():
    params = build_generation_params("x", {}, _NEBULA_ULTRA_SPEC, "text2image")
    assert params["quality"] == "2K"
    assert params["ratio"] == "16:9"


def test_build_params_nebula_ultra_count_from_n_or_count_kw():
    assert build_generation_params("x", {"n": 2}, _NEBULA_ULTRA_SPEC, "text2image")["count"] == 2
    assert build_generation_params("x", {"count": 4}, _NEBULA_ULTRA_SPEC, "text2image")["count"] == 4


def test_build_params_nebula_ultra_count_bare_list_default_does_not_crash():
    # the "count" property is a bare enum list ([1, 2, 4]), not a dict with a "default"
    # key; build_generation_params must fall back to 1 instead of raising.
    params = build_generation_params("x", {}, _NEBULA_ULTRA_SPEC, "text2image")
    assert params["count"] == 1


def test_build_params_nebula_ultra_searchable_advanced_setting_not_sent_unless_explicit():
    params = build_generation_params("x", {}, _NEBULA_ULTRA_SPEC, "text2image")
    assert "searchable" not in params
    params_explicit = build_generation_params(
        "x", {"advancedSettings": {"searchable": 1}}, _NEBULA_ULTRA_SPEC, "text2image"
    )
    assert params_explicit["searchable"] == 1


def test_build_params_nebula_ultra_image2image_still_applies_quality_and_ratio():
    params = build_generation_params("x", {"quality": "4k", "aspect_ratio": "1:1"}, _NEBULA_ULTRA_SPEC, "image2image")
    assert params["modeType"] == "image2image"
    assert params["quality"] == "4K"
    assert params["ratio"] == "1:1"


def test_infer_image_mode():
    assert _infer_image_mode([]) == "text2image"
    assert _infer_image_mode(["ref.png"]) == "image2image"


def test_build_node_batch_body_video_carries_model_and_params():
    body = build_node_batch_body(
        "proj-uuid", "video", "nk-1", "视频节点", "wanxiang-plus", {"prompt": "p", "ratio": "16:9"}
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
    assert data["params"]["ratio"] == "16:9"  # flattened, not nested


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


class UploadFake:
    """Routes by URL substring: passport getUserInfo, bridge init/complete, presigned PUT."""

    def __init__(self, put_status=200):
        self.calls = []
        self.put_status = put_status

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json, headers))
        if "getUserInfo" in url:
            return FakeResponse({"code": 0, "data": {"uuid": "user-xyz"}})
        if url.endswith("/init/4"):
            return FakeResponse(
                {"code": 0, "data": {"uploadId": "up-1", "parts": [{"partNumber": 1, "url": "https://oss/put/1"}]}}
            )
        if url.endswith("/complete/4"):
            return FakeResponse({"code": 0, "data": {"cdnUrl": "https://libtv-res/uploaded.png"}})
        raise AssertionError(f"unexpected POST {url}")

    def put_bytes(self, url, data):
        self.calls.append(("PUT", url, len(data) if data else 0, None))
        self.put_data = (self.put_data if hasattr(self, "put_data") else []) + [bytes(data or b"")]
        return self.put_status


class AsyncUploadFake(UploadFake):
    async def post(self, url, json=None, headers=None, timeout=None):
        return UploadFake.post(self, url, json, headers, timeout)


def test_build_upload_path():
    assert build_upload_path("u1", "abc123", "photo.PNG") == "upload-images/u1/abc123.PNG"
    assert build_upload_path("u1", "abc123", "noext") == "upload-images/u1/abc123"


def test_parse_upload_url_prefers_cdn():
    assert parse_upload_url({"data": {"cdnUrl": "c", "ossUrl": "o", "path": "p"}}) == "c"
    assert parse_upload_url({"data": {"ossUrl": "o", "path": "p"}}) == "o"
    assert parse_upload_url({"data": {"path": "p"}}) == "p"
    with pytest.raises(LibTVError):
        parse_upload_url({"data": {}})


def test_resolve_user_uuid_caches():
    fake = UploadFake()
    lt = LibTVClient(token="t", webid="w", sync_client=fake)
    assert lt.resolve_user_uuid() == "user-xyz"
    assert lt.resolve_user_uuid() == "user-xyz"  # cached
    assert sum(1 for c in fake.calls if "getUserInfo" in c[1]) == 1


def test_upload_media_orchestrates_init_put_complete():
    fake = UploadFake()
    lt = LibTVClient(token="t", webid="w", sync_client=fake, http_put=fake.put_bytes)
    url = lt.upload_media(b"\x89PNG-bytes", "ref.png")
    assert url == "https://libtv-res/uploaded.png"
    methods_urls = [(m, u) for m, u, *_ in fake.calls]
    assert ("POST", "https://passport.liblib.art/api/www/user/getUserInfo") in methods_urls
    assert any(m == "POST" and u.endswith("/init/4") for m, u in methods_urls)
    assert ("PUT", "https://oss/put/1") in methods_urls
    assert any(m == "POST" and u.endswith("/complete/4") for m, u in methods_urls)
    init_body = next(c[2] for c in fake.calls if c[1].endswith("/init/4"))
    assert (
        init_body["path"]
        == "upload-images/user-xyz/" + __import__("hashlib").sha1(b"\x89PNG-bytes").hexdigest() + ".png"
    )


def test_upload_media_raises_on_put_failure():
    fake = UploadFake(put_status=403)
    lt = LibTVClient(token="t", webid="w", sync_client=fake, http_put=fake.put_bytes)
    with pytest.raises(LibTVError):
        lt.upload_media(b"x", "ref.png")


def test_reference_payload_url_passthrough():
    assert _reference_payload("https://x/a.png") == ("url", "https://x/a.png", None)


def test_reference_payload_bytes_and_tuple():
    assert _reference_payload(b"abc") == ("bytes", "reference.png", b"abc")
    assert _reference_payload(("my.png", b"data")) == ("bytes", "my.png", b"data")


def test_reference_payload_none():
    assert _reference_payload(None) is None


def test_collect_reference_groups_merges_and_typed_keys():
    images, videos, audios = _collect_reference_groups(
        {
            "input_reference": "https://x/a.png",
            "image_references": ["https://x/b.png", "https://x/c.png"],
            "video_references": ["https://x/v.mp4"],
            "audio_references": ["https://x/s.mp3"],
        }
    )
    assert images == ["https://x/a.png", "https://x/b.png", "https://x/c.png"]
    assert videos == ["https://x/v.mp4"]
    assert audios == ["https://x/s.mp3"]


def test_collect_reference_groups_single_tuple_is_one_image():
    images, _, _ = _collect_reference_groups({"input_reference": ("a.png", b"data")})
    assert images == [("a.png", b"data")]


def test_default_video_mode_priority():
    assert _default_video_mode([], [], []) == "text2video"
    assert _default_video_mode(["i"], [], []) == "image2video"
    assert _default_video_mode(["i"], ["v"], []) == "video2video"
    assert _default_video_mode(["i"], ["v"], ["a"]) == "audio2video"


def test_apply_video_references_sets_lists():
    params = {}
    LibTVLLM._apply_video_references(params, "image2video", ["u1", "u2"], [], [])
    assert params["imageList"] == ["u1", "u2"]
    assert "videoList" not in params and "mixedList" not in params


def test_collect_reference_groups_wavespeed_aliases():
    images, videos, audios = _collect_reference_groups(
        {
            "reference_images": ["https://x/r1.png"],
            "image": "https://x/first.png",
            "last_image": "https://x/last.png",
            "reference_audios": ["https://x/a.mp3"],
        }
    )
    assert images == ["https://x/r1.png", "https://x/first.png", "https://x/last.png"]
    assert audios == ["https://x/a.mp3"]


def test_infer_video_mode_frames_when_last_image():
    assert _infer_video_mode({"last_image": "u"}, ["a", "b"], [], []) == "frames2video"
    assert _infer_video_mode({}, ["a"], [], []) == "image2video"
    assert _infer_video_mode({}, [], [], ["x"]) == "audio2video"


def test_build_video_object_encodes_task_id_and_status_model_queued():
    llm = LibTVLLM()
    vo = llm._build_video_object(
        "star-video2", {"task_id": "task-9", "project_uuid": "p"}, {"libtv_status_model": "seedance-2.0"}
    )
    assert vo.status == "queued"  # async submit: not done yet
    decoded = decode_video_id_with_provider(vo.id)
    assert decoded["custom_llm_provider"] == LIBTV_PROVIDER
    assert decoded["video_id"] == "task-9"  # the libtv task id, not a url
    # model_id routes /v1/videos/{id} status/content back to THIS account's deployment
    assert decoded["model_id"] == "seedance-2.0"


def _progress_route(status, url=None, reason=None):
    prog = {"status": status}
    if url is not None:
        prog["taskResult"] = json.dumps({"videos": [{"videoUrl": url}]})
    if reason is not None:
        prog["failedReason"] = reason
        prog["startTimeMs"] = 1700000000000
        prog["progressPercent"] = 100
    return {"/api/task/generation/progress": {"code": 0, "data": {"progresses": [prog]}}}


def test_video_status_polls_and_maps_completed():
    vid = LibTVLLM()._build_video_object("m", {"task_id": "task-9"}).id
    client = FakeSyncClient(post_by_path=_progress_route(2, url="https://libtv-res/v.mp4"))
    status = LibTVLLM().video_status(vid, "tok", None, {"webid": "w"}, None, client=client)
    assert status.status == "completed"
    assert status._hidden_params["url"] == "https://libtv-res/v.mp4"
    assert [c[0] for c in client.calls] == ["/api/task/generation/progress"]


def test_video_status_maps_failed_with_reason():
    vid = LibTVLLM()._build_video_object("m", {"task_id": "task-9"}).id
    client = FakeSyncClient(post_by_path=_progress_route(3, reason="生成视频可能涉及版权限制"))
    status = LibTVLLM().video_status(vid, "tok", None, {"webid": "w"}, None, client=client)
    assert status.status == "failed"
    assert status.error["message"] == "生成视频可能涉及版权限制"


def test_video_status_maps_in_progress():
    vid = LibTVLLM()._build_video_object("m", {"task_id": "task-9"}).id
    client = FakeSyncClient(post_by_path=_progress_route(1))
    assert LibTVLLM().video_status(vid, "tok", None, {"webid": "w"}, None, client=client).status == "in_progress"


class _DownloadResp:
    status_code = 200
    content = b"MP4BYTES"


class _PollAndDownloadClient(FakeSyncClient):
    def __init__(self, status, url=None):
        super().__init__(post_by_path=_progress_route(status, url=url))
        self.got = None

    def get(self, url, headers=None, timeout=None, params=None):
        self.got = url
        return _DownloadResp()


def test_video_content_polls_then_downloads():
    vid = LibTVLLM()._build_video_object("m", {"task_id": "task-9"}).id
    c = _PollAndDownloadClient(2, url="https://libtv-res/v.mp4")
    data = LibTVLLM().video_content(vid, "tok", None, {"webid": "w"}, None, client=c)
    assert data == b"MP4BYTES"
    assert c.got == "https://libtv-res/v.mp4"
    assert any(call[0] == "/api/task/generation/progress" for call in c.calls)


def test_video_content_raises_while_still_processing():
    vid = LibTVLLM()._build_video_object("m", {"task_id": "task-9"}).id
    c = _PollAndDownloadClient(1)
    with pytest.raises(LibTVError):
        LibTVLLM().video_content(vid, "tok", None, {"webid": "w"}, None, client=c)


def test_video_status_rejects_non_libtv_id():
    with pytest.raises(LibTVError):
        LibTVLLM().video_status("plain-id", None, None, {}, None)


def test_video_status_routes_through_litellm():
    import litellm
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.videos.main import VideoObject
    from litellm.types.videos.utils import encode_video_id_with_provider

    class StubLLM(CustomLLM):
        def video_status(self, video_id, api_key, api_base, optional_params, logging_obj, timeout=None, client=None):
            return VideoObject(id=video_id, object="video", status="completed")

        def video_content(self, video_id, api_key, api_base, optional_params, logging_obj, timeout=None, client=None):
            return b"BYTES"

    litellm.custom_provider_map = [{"provider": "libtvstub", "custom_handler": StubLLM()}]
    try:
        vid = encode_video_id_with_provider("https://x/v.mp4", "libtvstub")
        status = litellm.video_status(video_id=vid)
        assert status.status == "completed"
        content = litellm.video_content(video_id=vid)
        assert content == b"BYTES"
    finally:
        litellm.custom_provider_map = []


def test_apply_video_references_mixed_builds_mixedlist():
    params = {}
    LibTVLLM._apply_video_references(params, "mixed2video", ["img"], ["vid"], ["aud"])
    assert params["mixedList"] == [
        {"url": "img", "type": "image"},
        {"url": "vid", "type": "video"},
        {"url": "aud", "type": "audio"},
    ]


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
    payload = {
        "data": {"progresses": [{"status": 3, "failedReason": "blocked", "startTimeMs": 1700000000000}]}
    }
    state = parse_progress(payload, "video")
    assert state["status"] == 3
    assert state["failed_reason"] == "blocked"


def test_parse_progress_loading_has_no_urls():
    payload = {"data": {"progresses": [{"status": 1}]}}
    assert parse_progress(payload, "video") == {"status": 1, "urls": [], "failed_reason": None}


# --- progress poll racing generation/create: libtv returns a "task data abnormal" -----
# status 3 for a taskId it cannot find yet (the window right after create, before the
# task is visible upstream), with no startTimeMs/endTimeMs and progressPercent 0. This
# must not be read as a real failure or the whole generation gets killed while the
# upstream task is actually still running.

_UNKNOWN_TASK_REASON = "任务数据异常，请重新发起任务"


def _unknown_task_progress(task_id):
    return {
        "taskId": task_id,
        "status": 3,
        "failedReason": _UNKNOWN_TASK_REASON,
        "progressPercent": 0,
        "power": 0,
    }


@pytest.mark.parametrize("kind", ["video", "image"])
def test_parse_progress_unknown_task_signature_is_not_terminal(kind):
    payload = {"data": {"progresses": [_unknown_task_progress("task-9")]}}
    assert parse_progress(payload, kind, task_id="task-9") == {
        "status": None,
        "urls": [],
        "failed_reason": None,
    }


def test_parse_progress_ignores_entry_for_a_different_task_id():
    payload = {
        "data": {
            "progresses": [
                {
                    "taskId": "other-task",
                    "status": 2,
                    "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/wrong.mp4"}]}),
                },
                {"taskId": "task-9", "status": 1},
            ]
        }
    }
    state = parse_progress(payload, "video", task_id="task-9")
    assert state == {"status": 1, "urls": [], "failed_reason": None}


def test_parse_progress_real_failure_with_timing_stays_failed():
    payload = {
        "data": {
            "progresses": [
                {
                    "taskId": "task-9",
                    "status": 3,
                    "failedReason": "视频生成失败，请重试",
                    "startTimeMs": 1700000000000,
                    "endTimeMs": 1700000010000,
                    "progressPercent": 100,
                }
            ]
        }
    }
    state = parse_progress(payload, "video", task_id="task-9")
    assert state["status"] == 3
    assert state["failed_reason"] == "视频生成失败，请重试"


def test_parse_progress_real_failure_without_timing_stays_failed():
    # a task rejected before it starts (e.g. compliance) has no startTimeMs and
    # progressPercent 0, but its substantive failedReason must keep it terminal
    payload = {
        "data": {
            "progresses": [
                {
                    "taskId": "task-9",
                    "status": 3,
                    "failedReason": "生成视频可能涉及版权限制",
                    "progressPercent": 0,
                }
            ]
        }
    }
    state = parse_progress(payload, "video", task_id="task-9")
    assert state["status"] == 3
    assert state["failed_reason"] == "生成视频可能涉及版权限制"


def test_generate_compliance_rejection_without_timing_short_circuits():
    from litellm.llms.libtv.common import LibTVContentPolicyError

    client = FakeSyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p"}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {"taskId": "t"}},
            "/api/task/generation/progress": {
                "code": 0,
                "data": {
                    "progresses": [
                        {
                            "taskId": "t",
                            "status": 3,
                            "failedReason": "生成视频可能涉及版权限制",
                            "progressPercent": 0,
                        }
                    ]
                },
            },
        }
    )
    lt = LibTVClient(token="t", webid="w", sync_client=client, poll_interval=0)
    with pytest.raises(LibTVContentPolicyError):
        lt.generate("m", "v", "video", {"prompt": "x"}, "n")
    assert [c[0] for c in client.calls].count("/api/task/generation/progress") == 1


def test_generate_survives_unknown_task_polls_then_completes():
    task_id = "task-race"
    unknown = {"code": 0, "data": {"progresses": [_unknown_task_progress(task_id)]}}
    done = {
        "code": 0,
        "data": {
            "progresses": [
                {
                    "taskId": task_id,
                    "status": 2,
                    "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/done.mp4"}]}),
                }
            ]
        },
    }
    client = FakeSyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p"}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {"taskId": task_id}},
            "/api/task/generation/progress": [unknown, unknown, done],
        }
    )
    lt = LibTVClient(token="t", webid="w", sync_client=client, poll_interval=0)
    result = lt.generate("m", "v", "video", {"prompt": "x"}, "n")
    assert result["urls"] == ["https://x/done.mp4"]
    assert [c[0] for c in client.calls].count("/api/task/generation/progress") == 3


def test_video_status_maps_unknown_task_signature_to_in_progress():
    task_id = "task-race"
    vid = LibTVLLM()._build_video_object("m", {"task_id": task_id}).id
    client = FakeSyncClient(
        post_by_path={
            "/api/task/generation/progress": {
                "code": 0,
                "data": {"progresses": [_unknown_task_progress(task_id)]},
            }
        }
    )
    status = LibTVLLM().video_status(vid, "tok", None, {"webid": "w"}, None, client=client)
    assert status.status == "in_progress"


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
                {
                    "code": 0,
                    "data": {
                        "progresses": [
                            {"status": 3, "failedReason": "nope", "startTimeMs": 1700000000000}
                        ]
                    },
                }
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
    assert params["ratio_auto"] == "16:9"
    assert params["quality_high"] == "720p"
    assert params["duration"] == 5


def test_build_params_resolution_480_key():
    spec = {
        "config": {"settings": {"singleImage2video": ["resolution_480", "duration"]}},
        "properties": {"resolution_480": {"default": "480p"}, "duration": {"default": 5}},
    }
    params = build_generation_params("x", {"size": "854x480"}, spec, "singleImage2video")
    assert params["resolution_480"] == "480p"


def test_count_list_property_does_not_crash():
    spec = {"config": {"settings": []}, "properties": {"count": [1, 2, 3, 4]}}
    params = build_generation_params("x", {}, spec, "text2video")
    assert params["count"] == 1


# Real metadata for kling-v3-omni ("可灵 3.0 Omni"), the model actually shipping on
# the drama platform (kling-video-o3 was superseded before launch). Kept as the full
# payload (not just properties/config) so the fixture matches what resolve_model_spec
# actually stores for this model.
_KLING_V3_OMNI_SPEC = {
    "modelKey": "kling-v3-omni",
    "modelName": "可灵 3.0 Omni",
    "modelVendor": "Kling",
    "properties": {
        "count": [1],
        "magic": True,
        "mention": True,
        "prompt": {"maxLength": 2000, "originalField": "prompt"},
        "ratio": {"enum": ["16:9", "1:1", "9:16"], "default": "16:9", "originalField": "ratio"},
        "ratio_auto": {
            "enum": [{"value": " ", "displayName": "智能模式"}, "16:9", "1:1", "9:16"],
            "default": " ",
            "originalField": "ratio",
        },
        "duration": {"min": 3, "max": 15, "default": 5, "originalField": "duration"},
        "duration_10": {"min": 3, "max": 10, "default": 5, "originalField": "duration"},
        "quality": {
            "enum": [{"value": "low", "displayName": "标准"}, {"value": "high", "displayName": "高品质"}],
            "default": "low",
            "originalField": "quality",
        },
        "quality_4k": {
            "enum": [
                {"value": "low", "displayName": "标准"},
                {"value": "high", "displayName": "高品质"},
                {"value": "4k", "displayName": "4K"},
            ],
            "default": "low",
            "originalField": "quality",
        },
        "enableSound": {
            "enum": [{"value": "on", "displayName": "开启"}, {"value": "off", "displayName": "关闭"}],
            "default": "on",
            "originalField": "enableSound",
        },
        "modeType": {
            "originalField": "modeType",
            "items": {
                "mixed2video": [1, 7],
                "frames2video": [1, 2],
                "videoEdit2video": [1, 5],
                "audio2video": [0, 0],
                "singleImage2video": [1, 1],
            },
        },
        "smartStoryboard": {
            "displayName": "智能分镜",
            "component": "switch",
            "default": False,
            "originalField": "multi_shot",
        },
    },
    "config": {
        "settings": {
            "text2video": ["ratio", "enableSound", "duration", "quality_4k", "smartStoryboard"],
            "frames2video": ["ratio_auto", "enableSound", "duration", "quality_4k"],
            "singleImage2video": ["ratio", "enableSound", "duration", "quality_4k", "smartStoryboard"],
            "videoEdit2video": ["ratio_auto", "duration_10", "quality", "enableSound"],
            "mixed2video": ["ratio", "duration", "quality_4k", "enableSound", "smartStoryboard"],
        }
    },
    "rules": [{"require": ["prompt", "media"], "mode": "any"}, {"require": ["prompt"]}],
}


def test_build_params_kling_omni_text2video_defaults():
    params = build_generation_params("a cat", {}, _KLING_V3_OMNI_SPEC, "text2video")
    assert params["ratio"] == "16:9"
    assert params["quality_4k"] == "low"
    assert params["duration"] == 5
    assert params["enableSound"] == "on"
    assert params["smartStoryboard"] is False


def test_build_params_kling_omni_text2video_explicit_values():
    # quality_4k is the settings key omni buckets text2video quality under; without
    # prefix-matching on "quality*" this key falls out of the old exact-key
    # candidates dict and the requested quality is silently dropped to the default.
    params = build_generation_params(
        "a cat",
        {"aspect_ratio": "9:16", "quality": "high", "seconds": 8, "generate_audio": False},
        _KLING_V3_OMNI_SPEC,
        "text2video",
    )
    assert params["ratio"] == "9:16"
    assert params["quality_4k"] == "high"
    assert params["duration"] == 8
    assert isinstance(params["duration"], int)
    assert params["enableSound"] == "off"
    assert params["smartStoryboard"] is False


def test_build_params_kling_omni_canonicalizes_generic_image2video_mode():
    # litellm's own mode-inference layer (_infer_video_mode) returns the generic
    # "image2video" for a single reference image, but omni's schema buckets its
    # settings under "singleImage2video".
    params = build_generation_params("a cat", {"quality": "high"}, _KLING_V3_OMNI_SPEC, "image2video")
    assert params["quality_4k"] == "high"
    assert params["duration"] == 5
    assert params["enableSound"] == "on"
    assert params["smartStoryboard"] is False


def test_build_params_kling_omni_canonicalizes_generic_video2video_mode_to_mixed2video():
    params = build_generation_params(
        "a cat", {"aspect_ratio": "1:1", "quality": "high", "seconds": 6}, _KLING_V3_OMNI_SPEC, "video2video"
    )
    assert params["modeType"] == "mixed2video"
    assert params["ratio"] == "1:1"
    assert params["quality_4k"] == "high"
    assert params["duration"] == 6


def test_build_params_kling_omni_video_edit_uses_duration_10_and_quality():
    # videoEdit2video is the one omni bucket no generic mode aliases to; callers must
    # pass it explicitly. It uses duration_10 (max 10s, still int-coerced by the
    # existing key.startswith("duration") branch) and the low/high-only "quality"
    # key -- a requested "4k" isn't a valid option there and coerces to the default.
    params = build_generation_params("a cat", {"seconds": 8, "quality": "4k"}, _KLING_V3_OMNI_SPEC, "videoEdit2video")
    assert params["duration_10"] == 8
    assert isinstance(params["duration_10"], int)
    assert params["quality"] == "low"
    assert "quality_4k" not in params
    assert "duration" not in params


def test_build_params_kling_omni_frames2video_uses_ratio_auto_and_drops_storyboard():
    params = build_generation_params(
        "a cat", {"aspect_ratio": "9:16", "quality": "high"}, _KLING_V3_OMNI_SPEC, "frames2video"
    )
    assert params["ratio_auto"] == "9:16"
    assert "ratio" not in params
    assert params["quality_4k"] == "high"
    assert "smartStoryboard" not in params


def test_build_params_flat_settings_seedance_unaffected_by_mode_canonicalization():
    # seedance's config.settings is a flat list, so _allowed_setting_keys returns
    # before ever consulting the mode alias table; canonicalization must be a no-op.
    params = build_generation_params("a cat", {"seconds": "8"}, _SEEDANCE_SPEC, "image2video")
    assert params["modeType"] == "image2video"
    assert params["ratio"] == "16:9"
    assert params["resolution"] == "720p"
    assert params["duration"] == 8


def test_build_params_kling_omni_generate_audio_false_maps_to_enablesound_off():
    params = build_generation_params("a cat", {"generate_audio": False}, _KLING_V3_OMNI_SPEC, "text2video")
    assert params["enableSound"] == "off"


def test_build_params_kling_omni_generate_audio_true_maps_to_enablesound_on():
    params = build_generation_params("a cat", {"generate_audio": True}, _KLING_V3_OMNI_SPEC, "text2video")
    assert params["enableSound"] == "on"


def test_build_params_kling_omni_explicit_enablesound_wins_over_generate_audio():
    params = build_generation_params(
        "a cat", {"enableSound": "off", "generate_audio": True}, _KLING_V3_OMNI_SPEC, "text2video"
    )
    assert params["enableSound"] == "off"


def test_build_params_kling_omni_smart_storyboard_honors_user_value():
    params = build_generation_params("a cat", {"smartStoryboard": True}, _KLING_V3_OMNI_SPEC, "text2video")
    assert params["smartStoryboard"] is True

    params = build_generation_params("a cat", {"smartStoryboard": False}, _KLING_V3_OMNI_SPEC, "text2video")
    assert params["smartStoryboard"] is False


def test_build_params_kling_omni_count_is_forced_to_one():
    # properties.count is `[1]` (a bare list, not a {"default": ...} dict), so
    # count_default takes the `isinstance(count_prop, dict)` else-branch and is
    # hard-coded to 1 regardless of the list's contents.
    params = build_generation_params("a cat", {}, _KLING_V3_OMNI_SPEC, "text2video")
    assert params["count"] == 1


_HAPPY_HORSE_11_SPEC = {
    "modelKey": "happy-horse-1.1",
    "modelName": "Happy Horse 1.1",
    "modelVendor": "happy-horse",
    "baseType": 27,
    "properties": {
        "strikethroughPrice": {"active": True},
        "magic": False,
        "mention": True,
        "count": [1],
        "prompt": {"maxLength": 0},
        "ratio": {
            "displayName": "比例",
            "enum": ["16:9", "9:16", "1:1", "4:3", "3:4"],
            "default": "16:9",
            "component": "singleButton",
            "originalField": "ratio",
        },
        "resolution": {
            "displayName": "清晰度",
            "enum": ["720P", "1080P"],
            "default": "720P",
            "component": "singleButton",
            "originalField": "resolution",
        },
        "duration": {
            "min": 3,
            "max": 15,
            "step": 1,
            "default": 5,
            "originalField": "duration",
            "component": "slider",
        },
        "modeType": {
            "originalField": "modeType",
            "items": {
                "frames2video": [1, 1],
                "videoEdit2video": [1, 5],
                "image2video": [1, 9],
            },
            "videoEdit2videoConfig": {"videoMax": 1, "imageMaxWithVideo": 5},
        },
    },
    "config": {
        "settings": {
            "text2video": ["ratio", "resolution", "duration"],
            "frames2video": ["resolution", "duration"],
            "image2video": ["ratio", "resolution", "duration"],
            "videoEdit2video": ["resolution"],
        }
    },
    "rules": [{"require": ["prompt", "media"], "mode": "any"}],
}


def test_build_params_happy_horse_text2video_coerces_lowercase_resolution():
    params = build_generation_params(
        "a horse", {"resolution": "720p", "aspect_ratio": "9:16", "seconds": 8}, _HAPPY_HORSE_11_SPEC, "text2video"
    )
    assert params["resolution"] == "720P"
    assert params["ratio"] == "9:16"
    assert params["duration"] == 8
    assert isinstance(params["duration"], int)


def test_build_params_happy_horse_image2video_is_native_bucket_not_aliased():
    # happy-horse buckets its own "image2video" settings natively -- unlike omni it
    # has no "singleImage2video" key, so canonicalization must leave the mode alone
    # (exact-key-wins) instead of aliasing it away to a bucket that doesn't exist.
    params = build_generation_params("a horse", {"quality": "high"}, _HAPPY_HORSE_11_SPEC, "image2video")
    assert params["modeType"] == "image2video"
    assert params["ratio"] == "16:9"
    assert params["resolution"] == "720P"
    assert params["duration"] == 5


def test_build_params_happy_horse_explicit_mixed2video_canonicalizes_to_videoedit2video():
    # happy-horse has no "mixed2video" bucket at all -- only "videoEdit2video" -- so
    # the drama backend's explicit build_generation_params(..., "mixed2video") call
    # for 视频编辑 must fall back to the bucket that actually exists.
    params = build_generation_params(
        "a horse", {"aspect_ratio": "1:1", "resolution": "1080p", "seconds": 6}, _HAPPY_HORSE_11_SPEC, "mixed2video"
    )
    assert params["modeType"] == "videoEdit2video"
    assert params["resolution"] == "1080P"
    assert "ratio" not in params
    assert "duration" not in params


def test_build_params_happy_horse_generic_video2video_canonicalizes_to_videoedit2video():
    params = build_generation_params("a horse", {"resolution": "1080p"}, _HAPPY_HORSE_11_SPEC, "video2video")
    assert params["modeType"] == "videoEdit2video"
    assert params["resolution"] == "1080P"


def test_build_params_kling_omni_mixed2video_exact_match_still_wins_over_alias_chain():
    # regression guard: omni has BOTH "mixed2video" and "videoEdit2video" buckets, so
    # extending the alias chain for happy-horse must not make omni's explicit
    # mixed2video calls fall through to videoEdit2video.
    params = build_generation_params("a cat", {"seconds": 6, "quality": "high"}, _KLING_V3_OMNI_SPEC, "mixed2video")
    assert params["modeType"] == "mixed2video"
    assert params["duration"] == 6
    assert params["quality_4k"] == "high"
    assert "duration_10" not in params
    assert "quality" not in params


def test_build_params_happy_horse_frames2video_single_image():
    params = build_generation_params(
        "a horse", {"resolution": "720p", "seconds": 4}, _HAPPY_HORSE_11_SPEC, "frames2video"
    )
    assert params["modeType"] == "frames2video"
    assert params["resolution"] == "720P"
    assert params["duration"] == 4
    assert "ratio" not in params


def test_build_params_happy_horse_generate_audio_ignored_no_enablesound_key():
    # no enableSound property or settings entry exists anywhere in happy-horse's
    # spec, so generate_audio must be silently ignored rather than raising or
    # injecting a key the vendor schema doesn't recognize.
    params = build_generation_params("a horse", {"generate_audio": True}, _HAPPY_HORSE_11_SPEC, "text2video")
    assert "enableSound" not in params
    assert params["ratio"] == "16:9"


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


def _tool_spec_payload(model_key="star-video2", auto_compliance=True, frames2video=False):
    props = {
        "ratio": {"default": "9:16", "enum": ["16:9", "9:16"]},
        "resolution": {"default": "720p", "enum": [{"value": "720p"}]},
        "duration": {"default": 5},
        "portrait": True,
    }
    if auto_compliance:
        props["autoCompliance"] = {"enable": True, "default": 1}
    if frames2video:
        props["modeType"] = {"items": {"frames2video": [1, 2], "mixed2video": []}}
    meta = {
        "modelKey": model_key,
        "modelVendor": model_key,
        "properties": props,
        "config": {"settings": ["ratio", "resolution", "duration", "enableSound"]},
    }
    return {"data": {"tools": [{"type": "video", "metadata": json.dumps(meta)}]}}


_LIBTV_REF = "https://libtv-res.liblib.art/upload-images/uid/abc.png"


def _compliance_routes(verify_passed=True):
    risk = json.dumps({"passed": verify_passed, "needsReview": False, "riskDescription": "正常"})
    return {
        "/api/community/image/verify": {"code": 0, "data": {"list": [{"url": _LIBTV_REF, "riskLabels": risk}]}},
        "/api/third_asset/create": {"code": 0, "data": {"uuid": "u1"}},
        "/api/third_asset/check": {"code": 0, "data": {"list": [{"uuid": "u1", "assetId": "asset-AAA", "status": 1}]}},
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
        "/api/task/generation/progress": {
            "code": 0,
            "data": {
                "progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/o.mp4"}]})}]
            },
        },
    }


def test_portrait_compliance_converts_image_to_asset_and_sets_flag():
    fake = FakeSyncClient(post_by_path=_compliance_routes(verify_passed=True), get_payload=_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2", "subtle motion", "tok", None, {"webid": "w", "image": _LIBTV_REF}, None, client=fake
    )
    assert vo.status == "queued"
    paths = [c[0] for c in fake.calls]
    assert "/api/community/image/verify" in paths
    assert "/api/third_asset/create" in paths
    assert "/api/third_asset/check" in paths
    # verify request carries urlList (the field the upstream actually reads)
    verify_body = next(body for path, body in fake.calls if path == "/api/community/image/verify")
    assert verify_body == {"urlList": [_LIBTV_REF]}
    # generation body routes the image as asset:// via mixedList, NOT raw imageList
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["autoCompliance"] == 1
    assert gen_params["mixedList"] == [{"url": "asset://asset-AAA", "type": "image"}]
    assert gen_params.get("imageList") in ([], None)


def test_portrait_compliance_blocks_generation_when_verify_fails():
    fake = FakeSyncClient(post_by_path=_compliance_routes(verify_passed=False), get_payload=_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    with pytest.raises(LibTVError):
        llm.video_generation("star-video2", "x", "tok", None, {"webid": "w", "image": _LIBTV_REF}, None, client=fake)
    paths = [c[0] for c in fake.calls]
    assert "/api/community/image/verify" in paths
    assert "/api/task/generation/create" not in paths  # never submits a blocked portrait


def test_non_compliance_model_keeps_raw_imagelist_no_verify():
    routes = {
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
        "/api/task/generation/progress": {
            "code": 0,
            "data": {
                "progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/o.mp4"}]})}]
            },
        },
    }
    fake = FakeSyncClient(post_by_path=routes, get_payload=_tool_spec_payload(auto_compliance=False))
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation("star-video2", "x", "tok", None, {"webid": "w", "image": _LIBTV_REF}, None, client=fake)
    assert vo.status == "queued"
    paths = [c[0] for c in fake.calls]
    assert "/api/community/image/verify" not in paths
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["imageList"] == [_LIBTV_REF]
    assert "autoCompliance" not in gen_params


@pytest.mark.asyncio
async def test_portrait_compliance_async_happy_path():
    fake = FakeAsyncClient(post_by_path=_compliance_routes(verify_passed=True), get_payload=_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = await llm.avideo_generation(
        "star-video2", "subtle motion", "tok", None, {"webid": "w", "image": _LIBTV_REF}, None, client=fake
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["autoCompliance"] == 1
    assert gen_params["mixedList"] == [{"url": "asset://asset-AAA", "type": "image"}]


def test_compliance_exempt_image_keeps_cdn_url_not_asset():
    # verify passes but the asset reaches a terminal state with no assetId (non-portrait
    # exempt): the reference must fall back to the raw libtv cdn url, not asset://.
    routes = _compliance_routes(verify_passed=True)
    routes["/api/third_asset/check"] = {
        "code": 0,
        "data": {"list": [{"uuid": "u1", "assetId": None, "status": 1}]},
    }
    fake = FakeSyncClient(post_by_path=routes, get_payload=_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation("star-video2", "x", "tok", None, {"webid": "w", "image": _LIBTV_REF}, None, client=fake)
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["mixedList"] == [{"url": _LIBTV_REF, "type": "image"}]


_LIBTV_VIDEO = "https://libtv-res.liblib.art/upload-images/uid/clip.mp4"


def _mixed_compliance_routes():
    risk = json.dumps({"passed": True, "needsReview": False, "riskDescription": "正常"})
    return {
        "/api/community/image/verify": {"code": 0, "data": {"list": [{"url": _LIBTV_REF, "riskLabels": risk}]}},
        "/api/third_asset/create": [
            {"code": 0, "data": {"uuid": "u-img"}},
            {"code": 0, "data": {"uuid": "u-vid"}},
        ],
        "/api/third_asset/check": [
            {"code": 0, "data": {"list": [{"uuid": "u-img", "assetId": "asset-IMG", "status": 1}]}},
            {"code": 0, "data": {"list": [{"uuid": "u-vid", "assetId": "asset-VID", "status": 0}]}},
        ],
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
        "/api/task/generation/progress": {
            "code": 0,
            "data": {
                "progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/o.mp4"}]})}]
            },
        },
    }


def test_mixed2video_compliance_registers_reference_video_as_asset():
    # A video-edit (image + reference video) on a portrait model: the reference VIDEO can
    # itself show a real person, and the image moderation endpoint cannot score a video, so
    # the video must reach generation as a registered asset:// id (same path as the portrait
    # image), never as a raw cdn url, or libtv rejects the whole generation for missing
    # compliance.
    fake = FakeSyncClient(post_by_path=_mixed_compliance_routes(), get_payload=_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2",
        "swap the lead",
        "tok",
        None,
        {"webid": "w", "reference_images": [_LIBTV_REF], "reference_videos": [_LIBTV_VIDEO]},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    creates = [body for path, body in fake.calls if path == "/api/third_asset/create"]
    assert {c["assetType"] for c in creates} == {"image", "video"}
    assert next(c for c in creates if c["assetType"] == "video")["assetUrl"] == _LIBTV_VIDEO
    # image moderation scores only the image; it is never asked to score the video
    verify_bodies = [body for path, body in fake.calls if path == "/api/community/image/verify"]
    assert verify_bodies == [{"urlList": [_LIBTV_REF]}]
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["autoCompliance"] == 1
    assert gen_params["mixedList"] == [
        {"url": "asset://asset-IMG", "type": "image"},
        {"url": "asset://asset-VID", "type": "video"},
    ]


@pytest.mark.asyncio
async def test_mixed2video_compliance_registers_reference_video_as_asset_async():
    fake = FakeAsyncClient(post_by_path=_mixed_compliance_routes(), get_payload=_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = await llm.avideo_generation(
        "star-video2",
        "swap the lead",
        "tok",
        None,
        {"webid": "w", "reference_images": [_LIBTV_REF], "reference_videos": [_LIBTV_VIDEO]},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    creates = [body for path, body in fake.calls if path == "/api/third_asset/create"]
    assert next(c for c in creates if c["assetType"] == "video")["assetUrl"] == _LIBTV_VIDEO
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["mixedList"] == [
        {"url": "asset://asset-IMG", "type": "image"},
        {"url": "asset://asset-VID", "type": "video"},
    ]


_LIBTV_LAST = "https://libtv-res.liblib.art/upload-images/uid/last.png"


def _frames2video_compliance_routes():
    risk = json.dumps({"passed": True, "needsReview": False, "riskDescription": "正常"})
    return {
        "/api/community/image/verify": {
            "code": 0,
            "data": {"list": [{"url": _LIBTV_REF, "riskLabels": risk}, {"url": _LIBTV_LAST, "riskLabels": risk}]},
        },
        "/api/third_asset/create": [
            {"code": 0, "data": {"uuid": "u-first"}},
            {"code": 0, "data": {"uuid": "u-last"}},
        ],
        "/api/third_asset/check": [
            {"code": 0, "data": {"list": [{"uuid": "u-first", "assetId": "asset-FIRST", "status": 1}]}},
            {"code": 0, "data": {"list": [{"uuid": "u-last", "assetId": "asset-LAST", "status": 1}]}},
        ],
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
        "/api/task/generation/progress": {
            "code": 0,
            "data": {
                "progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/o.mp4"}]})}]
            },
        },
    }


def test_frames2video_compliance_keeps_first_last_order_and_mode():
    fake = FakeSyncClient(
        post_by_path=_frames2video_compliance_routes(),
        get_payload=_tool_spec_payload(frames2video=True),
    )
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2",
        "smile then wave",
        "tok",
        None,
        {"webid": "w", "image": _LIBTV_REF, "last_image": _LIBTV_LAST},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["modeType"] == "frames2video"
    assert gen_params["autoCompliance"] == 1
    assert gen_params["imageList"] == ["asset://asset-FIRST", "asset://asset-LAST"]
    assert "mixedList" not in gen_params


def test_reference_images_only_still_uses_mixed2video():
    fake = FakeSyncClient(post_by_path=_compliance_routes(verify_passed=True), get_payload=_tool_spec_payload(frames2video=True))
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2", "x", "tok", None, {"webid": "w", "reference_images": [_LIBTV_REF]}, None, client=fake
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["modeType"] == "mixed2video"
    assert gen_params["mixedList"] == [{"url": "asset://asset-AAA", "type": "image"}]


def test_frames2video_falls_back_to_mixed2video_when_spec_lacks_mode():
    fake = FakeSyncClient(post_by_path=_compliance_routes(verify_passed=True), get_payload=_tool_spec_payload(frames2video=False))
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2",
        "x",
        "tok",
        None,
        {"webid": "w", "image": _LIBTV_REF, "last_image": _LIBTV_REF},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["modeType"] == "mixed2video"
    assert "mixedList" in gen_params
    assert "imageList" not in gen_params or gen_params["imageList"] == []


@pytest.mark.asyncio
async def test_frames2video_compliance_async_keeps_first_last_order():
    fake = FakeAsyncClient(
        post_by_path=_frames2video_compliance_routes(),
        get_payload=_tool_spec_payload(frames2video=True),
    )
    llm = LibTVLLM(poll_interval=0)
    vo = await llm.avideo_generation(
        "star-video2",
        "smile then wave",
        "tok",
        None,
        {"webid": "w", "image": _LIBTV_REF, "last_image": _LIBTV_LAST},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["modeType"] == "frames2video"
    assert gen_params["imageList"] == ["asset://asset-FIRST", "asset://asset-LAST"]


def test_non_compliance_frames2video_keeps_first_last_imagelist_order():
    routes = {
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
        "/api/task/generation/progress": {
            "code": 0,
            "data": {
                "progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/o.mp4"}]})}]
            },
        },
    }
    fake = FakeSyncClient(post_by_path=routes, get_payload=_tool_spec_payload(auto_compliance=False, frames2video=True))
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2",
        "x",
        "tok",
        None,
        {"webid": "w", "image": _LIBTV_REF, "last_image": _LIBTV_LAST},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["modeType"] == "frames2video"
    assert gen_params["imageList"] == [_LIBTV_REF, _LIBTV_LAST]


def test_explicit_mixed2video_override_wins_over_last_image():
    fake = FakeSyncClient(
        post_by_path=_compliance_routes(verify_passed=True), get_payload=_tool_spec_payload(frames2video=True)
    )
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2",
        "x",
        "tok",
        None,
        {"webid": "w", "image": _LIBTV_REF, "last_image": _LIBTV_REF, "modeType": "mixed2video"},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["modeType"] == "mixed2video"
    assert "mixedList" in gen_params
    assert gen_params.get("imageList") in ([], None)


def test_frames2video_compliance_last_image_only_single_frame():
    risk = json.dumps({"passed": True, "needsReview": False, "riskDescription": "正常"})
    routes = _frames2video_compliance_routes()
    routes["/api/community/image/verify"] = {
        "code": 0,
        "data": {"list": [{"url": _LIBTV_LAST, "riskLabels": risk}]},
    }
    routes["/api/third_asset/create"] = {"code": 0, "data": {"uuid": "u-last"}}
    routes["/api/third_asset/check"] = {
        "code": 0,
        "data": {"list": [{"uuid": "u-last", "assetId": "asset-LAST", "status": 1}]},
    }
    fake = FakeSyncClient(post_by_path=routes, get_payload=_tool_spec_payload(frames2video=True))
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2", "x", "tok", None, {"webid": "w", "last_image": _LIBTV_LAST}, None, client=fake
    )
    assert vo.status == "queued"
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["modeType"] == "frames2video"
    assert gen_params["imageList"] == ["asset://asset-LAST"]


# --- video usage -> resolution-tiered cost (authoritative spend line) ---------
from litellm.llms.libtv.handler import _video_usage  # noqa: E402
from litellm.llms.openai.cost_calculation import video_generation_cost  # noqa: E402


def test_video_usage_carries_duration_and_resolution():
    assert _video_usage({"resolution": "720p", "seconds": 8}) == {
        "duration_seconds": 8.0,
        "video_resolution": "720p",
    }
    # resolution inferred from size when not given explicitly
    assert _video_usage({"size": "1920x1080", "duration": 5}) == {
        "duration_seconds": 5.0,
        "video_resolution": "1080p",
    }
    # no duration -> no usage (cost calc would otherwise bill 0 anyway)
    assert _video_usage({"resolution": "720p"}) is None


def test_build_video_object_populates_usage():
    vo = LibTVLLM()._build_video_object(
        "seedance-2.0",
        {"task_id": "task-1"},
        {"resolution": "1080p", "seconds": 8},
    )
    assert vo.usage == {"duration_seconds": 8.0, "video_resolution": "1080p"}


def test_libtv_video_cost_is_resolution_tiered():
    # The deployment model_info (litellm-config.yaml) carries the per-resolution
    # tiers; with the usage above, litellm bills $/second@resolution x seconds.
    model_info = {
        "mode": "video_generation",
        "output_cost_per_second_480p": 0.12,
        "output_cost_per_second_720p": 0.24,
        "output_cost_per_second_1080p": 0.60,
    }
    assert video_generation_cost(
        model="seedance-2.0",
        duration_seconds=8.0,
        custom_llm_provider="libtv",
        model_info=model_info,
        video_resolution="1080p",
    ) == pytest.approx(4.80)
    assert video_generation_cost(
        model="seedance-2.0",
        duration_seconds=5.0,
        custom_llm_provider="libtv",
        model_info=model_info,
        video_resolution="720p",
    ) == pytest.approx(1.20)


# --- reference media must be uploaded into the libtv project before generation -------
# libtv (canvas/star-video2) cannot fetch arbitrary external presigned urls; every
# reference (image/video/audio) must first land on libtv-res.liblib.art. Only the
# portrait image path uploaded before; videos/audios and non-compliance images leaked
# the external url verbatim. These lock the upload-before-generate contract.

_EXT_VIDEO = "https://minio.internal/bucket/clip.mp4?X-Amz-Signature=abc"
_LIBTV_UPLOADED = "https://libtv-res.liblib.art/upload-images/user-1/deadbeef.mp4"


class _FullSyncFake:
    """libtv api routes by path + passport getUserInfo + bridge init/complete by url substring."""

    def __init__(self, api_routes, get_payload=None, upload_cdn=_LIBTV_UPLOADED):
        self.api_routes = api_routes
        self.get_payload = get_payload
        self.upload_cdn = upload_cdn
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json))
        if "getUserInfo" in url:
            return FakeResponse({"code": 0, "data": {"uuid": "user-1"}})
        if url.endswith("/init/4"):
            return FakeResponse(
                {"code": 0, "data": {"uploadId": "up", "parts": [{"partNumber": 1, "url": "https://oss/put/1"}]}}
            )
        if url.endswith("/complete/4"):
            return FakeResponse({"code": 0, "data": {"cdnUrl": self.upload_cdn}})
        queue = self.api_routes[url.split("api.liblib.tv", 1)[-1]]
        return FakeResponse(queue.pop(0) if isinstance(queue, list) else queue)

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append((url, None))
        return FakeResponse(self.get_payload)


class _FullAsyncFake(_FullSyncFake):
    async def post(self, url, json=None, headers=None, timeout=None):
        return _FullSyncFake.post(self, url, json, headers, timeout)

    async def get(self, url, headers=None, params=None):
        return _FullSyncFake.get(self, url, headers, None, params)


def _gen_params(calls):
    return next(j["params"] for u, j in calls if u.endswith("/api/task/generation/create"))


def test_ensure_libtv_url_uploads_external_url_and_keeps_extension():
    fake = UploadFake()
    lt = LibTVClient(token="t", webid="w", sync_client=fake, http_put=fake.put_bytes, http_get=lambda u: b"VIDEOBYTES")
    out = lt.ensure_libtv_url("url", _EXT_VIDEO, None, "reference.mp4")
    assert out == "https://libtv-res/uploaded.png"  # UploadFake's complete cdnUrl
    init_body = next(c[2] for c in fake.calls if c[1].endswith("/init/4"))
    assert init_body["path"].endswith(".mp4")  # extension derived from clip.mp4, not hardcoded .png
    assert fake.put_data == [b"VIDEOBYTES"]  # the fetched external bytes are what got uploaded


def test_ensure_libtv_url_passes_through_libtv_res_url_without_upload():
    fake = UploadFake()
    lt = LibTVClient(token="t", webid="w", sync_client=fake, http_put=fake.put_bytes, http_get=lambda u: b"x")
    url = "https://libtv-res.liblib.art/upload-images/uid/abc.png"
    assert lt.ensure_libtv_url("url", url, None, "reference.png") == url
    assert fake.calls == []  # no getUserInfo / init / put / complete at all


def test_ensure_libtv_url_falls_back_to_default_name_when_no_extension():
    fake = UploadFake()
    lt = LibTVClient(token="t", webid="w", sync_client=fake, http_put=fake.put_bytes, http_get=lambda u: b"b")
    lt.ensure_libtv_url("url", "https://host/path/noext?sig=1", None, "reference.mp4")
    init_body = next(c[2] for c in fake.calls if c[1].endswith("/init/4"))
    assert init_body["path"].endswith(".mp4")  # default name's extension


@pytest.mark.asyncio
async def test_aensure_libtv_url_uploads_external_url():
    fake = AsyncUploadFake()
    lt = LibTVClient(token="t", webid="w", async_client=fake, http_put=fake.put_bytes, http_get=lambda u: b"VID")
    out = await lt.aensure_libtv_url("url", _EXT_VIDEO, None, "reference.mp4")
    assert out == "https://libtv-res/uploaded.png"
    init_body = next(c[2] for c in fake.calls if str(c[1]).endswith("/init/4"))
    assert init_body["path"].endswith(".mp4")


@pytest.mark.asyncio
async def test_avideo_generation_uploads_external_reference_video_in_compliance_branch():
    fake = _FullAsyncFake(_compliance_routes(verify_passed=True), get_payload=_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"VID", http_put=lambda u, d: 200)
    vo = await llm.avideo_generation(
        "star-video2",
        "edit it",
        "tok",
        None,
        {"webid": "w", "image": _LIBTV_REF, "reference_videos": [_EXT_VIDEO], "modeType": "mixed2video"},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    mixed = _gen_params(fake.calls)["mixedList"]
    # portrait image stays asset://; the external video is uploaded to libtv and then registered
    # for compliance, so it reaches generation as asset:// too (never a raw external/cdn url)
    assert {"url": "asset://asset-AAA", "type": "image"} in mixed
    assert {"url": "asset://asset-AAA", "type": "video"} in mixed
    assert all("minio.internal" not in m["url"] for m in mixed)
    # the upload still happens: the video is registered as a third_asset under its libtv cdn url
    video_create = next(
        j for u, j in fake.calls if u.endswith("/api/third_asset/create") and j.get("assetType") == "video"
    )
    assert video_create["assetUrl"] == _LIBTV_UPLOADED


def test_video_generation_uploads_external_reference_image_non_compliance_model():
    routes = {
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
        "/api/task/generation/progress": {
            "code": 0,
            "data": {
                "progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/o.mp4"}]})}]
            },
        },
    }
    fake = _FullSyncFake(
        routes,
        get_payload=_tool_spec_payload(auto_compliance=False),
        upload_cdn="https://libtv-res.liblib.art/up/img.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = llm.video_generation(
        "star-video2", "x", "tok", None, {"webid": "w", "image": "https://minio.internal/b/img.png"}, None, client=fake
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert gen_params["imageList"] == ["https://libtv-res.liblib.art/up/img.png"]  # uploaded, not the external url
    assert "/api/community/image/verify" not in [u.split("api.liblib.tv", 1)[-1] for u, _ in fake.calls]


def test_video_generation_kling_omni_explicit_mixed2video_with_images_only_no_video_required():
    routes = {
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
    }
    fake = _FullSyncFake(
        routes,
        get_payload=_tool_spec_payload(model_key="kling-v3-omni", auto_compliance=False),
    )
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "kling-v3-omni",
        "a dragon flying",
        "tok",
        None,
        {
            "webid": "w",
            "reference_images": [_LIBTV_REF, "https://libtv-res.liblib.art/upload-images/uid/def.png"],
            "modeType": "mixed2video",
        },
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "mixed2video"
    assert gen_params["mixedList"] == [
        {"url": _LIBTV_REF, "type": "image"},
        {"url": "https://libtv-res.liblib.art/upload-images/uid/def.png", "type": "image"},
    ]


def _nebula_ultra_tool_spec_payload():
    meta = {
        "modelKey": "nebula-ultra",
        "modelVendor": "nebula",
        "properties": _NEBULA_ULTRA_SPEC["properties"],
        "config": _NEBULA_ULTRA_SPEC["config"],
    }
    return {"data": {"tools": [{"type": "image", "metadata": json.dumps(meta)}]}}


_NEBULA_IMAGE_ROUTES = {
    "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
    "/api/canvas/nodes/batch": {"code": 0, "data": {}},
    "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
    "/api/task/generation/progress": {
        "code": 0,
        "data": {
            "progresses": [{"status": 2, "taskResult": json.dumps({"images": [{"imageUrl": "https://x/o.png"}]})}]
        },
    },
}


def test_image_generation_text_only_stays_text2image_and_omits_imagelist():
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    llm.image_generation(
        "nebula-ultra", "a cat", "tok", None, ImageResponse(), {"webid": "w", "quality": "2k"}, None, client=fake
    )
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "text2image"
    assert gen_params["imageList"] == []
    assert gen_params["quality"] == "2K"


def test_image_generation_with_reference_infers_image2image_and_uploads():
    fake = _FullSyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = llm.image_generation(
        "nebula-ultra",
        "restyle it",
        "tok",
        None,
        ImageResponse(),
        {"webid": "w", "reference_images": ["https://minio.internal/b/ref.png"]},
        None,
        client=fake,
    )
    assert vo.data[0].url == "https://x/o.png"
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "image2image"
    assert gen_params["imageList"] == ["https://libtv-res.liblib.art/up/ref.png"]  # uploaded, not the external url


def test_image_generation_explicit_mode_type_wins_over_inference():
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    llm.image_generation(
        "nebula-ultra",
        "a cat",
        "tok",
        None,
        ImageResponse(),
        {"webid": "w", "modeType": "text2image", "reference_images": [_LIBTV_REF]},
        None,
        client=fake,
    )
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "text2image"


@pytest.mark.asyncio
async def test_aimage_generation_with_reference_infers_image2image_and_uploads():
    fake = _FullAsyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = await llm.aimage_generation(
        "nebula-ultra",
        "restyle it",
        ImageResponse(),
        "tok",
        None,
        {"webid": "w", "reference_images": ["https://minio.internal/b/ref.png"]},
        None,
        client=fake,
    )
    assert vo.data[0].url == "https://x/o.png"
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "image2image"
    assert gen_params["imageList"] == ["https://libtv-res.liblib.art/up/ref.png"]


def test_image_generation_routes_to_custom_provider():
    import litellm
    from litellm.llms.custom_llm import CustomLLM

    captured = {}

    class StubLLM(CustomLLM):
        def image_generation(
            self,
            model,
            prompt,
            api_key,
            api_base,
            model_response,
            optional_params,
            logging_obj,
            timeout=None,
            client=None,
        ):
            captured["model"] = model
            captured["prompt"] = prompt
            model_response.data = [ImageObject(url="https://x/y.png")]
            return model_response

    litellm.custom_provider_map = [{"provider": "libtvstub", "custom_handler": StubLLM()}]
    try:
        out = litellm.image_generation(model="libtvstub/nebula-ultra", prompt="a fox")
        assert out.data[0]["url"] == "https://x/y.png"
        assert captured["model"] == "nebula-ultra"
        assert captured["prompt"] == "a fox"
    finally:
        litellm.custom_provider_map = []


# --- image_edit: the real transport for nano-banana-pro (nebula-ultra) reference
# requests. The drama backend and drama-cli both send reference images as multipart
# POST /v1/images/edits (files under "image"), never optional_params["reference_images"]
# on a JSON /v1/images/generations body, so image_edit/aimage_edit on LibTVLLM must
# work standalone from image_generation/aimage_generation. ---

_REF_TUPLE = ("reference_0.png", b"IMGBYTES", "image/png")


def test_image_edit_single_reference_uploads_and_carries_quality():
    fake = _FullSyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = llm.image_edit(
        "nebula-ultra",
        [_REF_TUPLE],
        "restyle it",
        ImageResponse(),
        "tok",
        None,
        {"webid": "w", "quality": "2k"},
        None,
        client=fake,
    )
    assert vo.data[0].url == "https://x/o.png"
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "image2image"
    assert gen_params["imageList"] == ["https://libtv-res.liblib.art/up/ref.png"]
    assert gen_params["quality"] == "2K"


def test_image_edit_single_file_not_wrapped_in_list_still_uploads():
    # litellm always normalizes `image` to a list before calling the custom
    # handler, but a direct/older caller might still pass a bare file; image_edit
    # must accept both shapes the way _as_list already does elsewhere.
    fake = _FullSyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = llm.image_edit(
        "nebula-ultra", _REF_TUPLE, "restyle it", ImageResponse(), "tok", None, {"webid": "w"}, None, client=fake
    )
    assert vo.data[0].url == "https://x/o.png"
    assert _gen_params(fake.calls)["imageList"] == ["https://libtv-res.liblib.art/up/ref.png"]


def test_image_edit_multiple_references_uploads_all():
    fake = _FullSyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    refs = [
        ("reference_0.png", b"A", "image/png"),
        ("reference_1.png", b"B", "image/png"),
        ("reference_2.png", b"C", "image/png"),
    ]
    llm.image_edit("nebula-ultra", refs, "restyle it", ImageResponse(), "tok", None, {"webid": "w"}, None, client=fake)
    gen_params = _gen_params(fake.calls)
    assert gen_params["imageList"] == ["https://libtv-res.liblib.art/up/ref.png"] * 3


def test_image_edit_without_reference_stays_text2image():
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    llm.image_edit(
        "nebula-ultra", [], "a cat from scratch", ImageResponse(), "tok", None, {"webid": "w"}, None, client=fake
    )
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "text2image"
    assert gen_params["imageList"] == []


@pytest.mark.asyncio
async def test_aimage_edit_uploads_reference_and_infers_image2image():
    fake = _FullAsyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = await llm.aimage_edit(
        "nebula-ultra",
        [_REF_TUPLE],
        "restyle it",
        ImageResponse(),
        "tok",
        None,
        {"webid": "w", "quality": "4k"},
        None,
        client=fake,
    )
    assert vo.data[0].url == "https://x/o.png"
    gen_params = _gen_params(fake.calls)
    assert gen_params["modeType"] == "image2image"
    assert gen_params["imageList"] == ["https://libtv-res.liblib.art/up/ref.png"]
    assert gen_params["quality"] == "4K"


# --- clarity-tiered image cost accrual (nano-banana-pro / libtv nebula-ultra) -------
# litellm's internally-accrued team spend is our authoritative wallet ledger, but the
# libtv handler never told litellm what an image actually cost, so spend always fell
# through to the flat config price regardless of the 1K/2K vs 4K clarity actually
# billed upstream. Deployment config now declares per-tier unit prices as
# litellm_params keys (output_cost_per_image_1k/_2k/_4k, mirroring how libtv video
# deployments declare output_cost_per_second_<resolution>); the router spreads
# litellm_params into image_generation()'s kwargs, and any key litellm doesn't
# recognize as a first-class param lands in optional_params for custom providers,
# which is how these tests set them up.

_NANO_BANANA_TIER_PRICES = {
    "output_cost_per_image_1k": 0.134,
    "output_cost_per_image_2k": 0.134,
    "output_cost_per_image_4k": 0.24,
}

_NEBULA_IMAGE_ROUTES_TWO_IMAGES = {
    **_NEBULA_IMAGE_ROUTES,
    "/api/task/generation/progress": {
        "code": 0,
        "data": {
            "progresses": [
                {
                    "status": 2,
                    "taskResult": json.dumps(
                        {"images": [{"imageUrl": "https://x/o1.png"}, {"imageUrl": "https://x/o2.png"}]}
                    ),
                }
            ]
        },
    },
}


def test_image_clarity_response_cost_multiplies_tier_price_by_count():
    optional_params = {"output_cost_per_image_4k": 0.24}
    assert _image_clarity_response_cost(optional_params, "4K", 1) == pytest.approx(0.24)
    assert _image_clarity_response_cost(optional_params, "4K", 3) == pytest.approx(0.72)


def test_image_clarity_response_cost_none_when_quality_or_tier_missing():
    assert _image_clarity_response_cost({"output_cost_per_image_4k": 0.24}, None, 1) is None
    assert _image_clarity_response_cost({"output_cost_per_image_4k": 0.24}, "2K", 1) is None
    assert _image_clarity_response_cost({}, "4K", 1) is None
    assert _image_clarity_response_cost({"output_cost_per_image_4k": 0.24}, "4K", 0) is None


def test_image_generation_clarity_tier_cost_4k():
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.image_generation(
        "nebula-ultra",
        "a cat",
        "tok",
        None,
        ImageResponse(),
        {"webid": "w", "quality": "4k", **_NANO_BANANA_TIER_PRICES},
        None,
        client=fake,
    )
    assert vo._hidden_params["response_cost"] == pytest.approx(0.24)


def test_image_generation_clarity_tier_cost_2k():
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.image_generation(
        "nebula-ultra",
        "a cat",
        "tok",
        None,
        ImageResponse(),
        {"webid": "w", "quality": "2k", **_NANO_BANANA_TIER_PRICES},
        None,
        client=fake,
    )
    assert vo._hidden_params["response_cost"] == pytest.approx(0.134)


def test_image_generation_clarity_tier_cost_defaults_when_quality_absent():
    # nebula-ultra's spec defaults quality to "2K" when the caller doesn't set it;
    # the accrued cost must follow the same default tier actually sent upstream.
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.image_generation(
        "nebula-ultra",
        "a cat",
        "tok",
        None,
        ImageResponse(),
        {"webid": "w", **_NANO_BANANA_TIER_PRICES},
        None,
        client=fake,
    )
    assert vo._hidden_params["response_cost"] == pytest.approx(0.134)


def test_image_generation_clarity_tier_cost_scales_with_count():
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES_TWO_IMAGES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.image_generation(
        "nebula-ultra",
        "a cat",
        "tok",
        None,
        ImageResponse(),
        {"webid": "w", "quality": "4k", "n": 2, **_NANO_BANANA_TIER_PRICES},
        None,
        client=fake,
    )
    assert len(vo.data) == 2
    assert vo._hidden_params["response_cost"] == pytest.approx(0.48)


def test_image_generation_no_tier_keys_leaves_response_cost_unset():
    # Other libtv image models (no output_cost_per_image_<tier> config) must be
    # byte-identical to pre-fix behavior: no hidden response_cost override, so
    # litellm's normal default_image_cost_calculator + flat config price applies.
    fake = _FullSyncFake(_NEBULA_IMAGE_ROUTES, get_payload=_nebula_ultra_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.image_generation(
        "nebula-ultra", "a cat", "tok", None, ImageResponse(), {"webid": "w", "quality": "2k"}, None, client=fake
    )
    assert "response_cost" not in vo._hidden_params


def test_image_edit_clarity_tier_cost_parity():
    # image_edit is the real transport for nano-banana-pro reference requests (see
    # comment above test_image_edit_single_reference_uploads_and_carries_quality);
    # cost accrual must work identically on that path.
    fake = _FullSyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = llm.image_edit(
        "nebula-ultra",
        [_REF_TUPLE],
        "restyle it",
        ImageResponse(),
        "tok",
        None,
        {"webid": "w", "quality": "4k", **_NANO_BANANA_TIER_PRICES},
        None,
        client=fake,
    )
    assert vo._hidden_params["response_cost"] == pytest.approx(0.24)


@pytest.mark.asyncio
async def test_aimage_edit_clarity_tier_cost_parity():
    fake = _FullAsyncFake(
        _NEBULA_IMAGE_ROUTES,
        get_payload=_nebula_ultra_tool_spec_payload(),
        upload_cdn="https://libtv-res.liblib.art/up/ref.png",
    )
    llm = LibTVLLM(poll_interval=0, http_get=lambda u: b"IMG", http_put=lambda u, d: 200)
    vo = await llm.aimage_edit(
        "nebula-ultra",
        [_REF_TUPLE],
        "restyle it",
        ImageResponse(),
        "tok",
        None,
        {"webid": "w", "quality": "2k", **_NANO_BANANA_TIER_PRICES},
        None,
        client=fake,
    )
    assert vo._hidden_params["response_cost"] == pytest.approx(0.134)


def test_libtv_overrides_base_class_image_edit():
    from litellm.llms.custom_llm import CustomLLM

    # Without the override, image_edit/aimage_edit inherit CustomLLM's stub which
    # unconditionally raises CustomLLMError(status_code=500, "Not implemented yet!") --
    # exactly the guaranteed-500 this fix removes.
    assert LibTVLLM.image_edit is not CustomLLM.image_edit
    assert LibTVLLM.aimage_edit is not CustomLLM.aimage_edit


def test_image_edit_routes_to_custom_provider_via_main_entry():
    # Exercises the real litellm.images.main.image_edit dispatch for a custom
    # provider (mirrors test_image_generation_routes_to_custom_provider), and
    # proves the images/main.py fix: n/quality/size are bound to image_edit's
    # named params, not **kwargs, so the custom-provider branch must fold them
    # back into optional_params or a custom handler never sees them.
    import litellm
    from litellm.llms.custom_llm import CustomLLM

    captured = {}

    class StubLLM(CustomLLM):
        def image_edit(
            self,
            model,
            image,
            prompt,
            model_response,
            api_key,
            api_base,
            optional_params,
            logging_obj,
            timeout=None,
            client=None,
        ):
            captured["model"] = model
            captured["prompt"] = prompt
            captured["image"] = image
            captured["optional_params"] = dict(optional_params)
            model_response.data = [ImageObject(url="https://x/y.png")]
            return model_response

    litellm.custom_provider_map = [{"provider": "libtvstub", "custom_handler": StubLLM()}]
    try:
        out = litellm.image_edit(
            model="libtvstub/nebula-ultra",
            image=[_REF_TUPLE],
            prompt="a fox",
            quality="2k",
            size="2048x2048",
            n=1,
        )
        assert out.data[0]["url"] == "https://x/y.png"
        assert captured["model"] == "nebula-ultra"
        assert captured["prompt"] == "a fox"
        assert captured["image"] == [_REF_TUPLE]
        assert captured["optional_params"]["quality"] == "2k"
        assert captured["optional_params"]["size"] == "2048x2048"
        assert captured["optional_params"]["n"] == 1
    finally:
        litellm.custom_provider_map = []


@pytest.mark.asyncio
async def test_aimage_edit_routes_to_custom_provider_via_main_entry():
    import litellm
    from litellm.llms.custom_llm import CustomLLM

    captured = {}

    class StubLLM(CustomLLM):
        async def aimage_edit(
            self,
            model,
            image,
            prompt,
            model_response,
            api_key,
            api_base,
            optional_params,
            logging_obj,
            timeout=None,
            client=None,
        ):
            captured["optional_params"] = dict(optional_params)
            model_response.data = [ImageObject(url="https://x/z.png")]
            return model_response

    litellm.custom_provider_map = [{"provider": "libtvstub", "custom_handler": StubLLM()}]
    try:
        out = await litellm.aimage_edit(
            model="libtvstub/nebula-ultra",
            image=[_REF_TUPLE],
            prompt="a fox",
            quality="4k",
            size="1024x1024",
        )
        assert out.data[0]["url"] == "https://x/z.png"
        assert captured["optional_params"]["quality"] == "4k"
        assert captured["optional_params"]["size"] == "1024x1024"
    finally:
        litellm.custom_provider_map = []


# --- libtv compliance classification: content rejections must fast-fail (no fallback) ---

from litellm.llms.libtv.common import (  # noqa: E402
    LibTVContentPolicyError,
    is_compliance_failure,
)

_CAPTURED_COMPLIANCE_REASON = "生成视频可能涉及版权限制，积分将会在2小时内返还，请调整描述或素材后重试"


def test_is_compliance_failure_matches_captured_copyright_reason():
    assert is_compliance_failure(_CAPTURED_COMPLIANCE_REASON) is True


@pytest.mark.parametrize(
    "reason",
    ["版权", "涉黄内容", "内容审核未通过", "疑似侵权", "copyright detected", "nsfw content"],
)
def test_is_compliance_failure_true_for_content_terms(reason):
    assert is_compliance_failure(reason) is True


@pytest.mark.parametrize(
    "reason",
    ["算力不足", "积分不足", "余额不足，请充值", "网络超时", "generation failed", "", None],
)
def test_is_compliance_failure_false_for_capacity_billing_network(reason):
    assert is_compliance_failure(reason) is False


def _failed_routes(reason):
    return {
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t"}},
        "/api/task/generation/progress": [
            {
                "code": 0,
                "data": {
                    "progresses": [
                        {"status": 3, "failedReason": reason, "startTimeMs": 1700000000000}
                    ]
                },
            }
        ],
    }


def test_generate_raises_content_policy_error_on_compliance_reason():
    lt = LibTVClient(
        token="t",
        webid="w",
        sync_client=FakeSyncClient(post_by_path=_failed_routes(_CAPTURED_COMPLIANCE_REASON)),
        poll_interval=0,
    )
    with pytest.raises(LibTVContentPolicyError):
        lt.generate("m", "v", "video", {"prompt": "x"}, "n")


def test_generate_capacity_failure_is_plain_error_not_content_policy():
    lt = LibTVClient(
        token="t",
        webid="w",
        sync_client=FakeSyncClient(post_by_path=_failed_routes("算力不足")),
        poll_interval=0,
    )
    with pytest.raises(LibTVError) as ei:
        lt.generate("m", "v", "video", {"prompt": "x"}, "n")
    assert not isinstance(ei.value, LibTVContentPolicyError)


def _video_failed_routes(reason):
    return {
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
        "/api/task/generation/progress": [{"code": 0, "data": {"progresses": [{"status": 3, "failedReason": reason}]}}],
    }


def _submit_routes():
    return {
        "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
        "/api/canvas/nodes/batch": {"code": 0, "data": {}},
        "/api/task/generation/create": {"code": 0, "data": {"taskId": "task-1"}},
    }


def test_video_generation_returns_queued_without_polling():
    fake = FakeSyncClient(post_by_path=_submit_routes(), get_payload=_tool_spec_payload(auto_compliance=False))
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "star-video2", "a fox", "tok", None, {"webid": "w", "libtv_status_model": "seedance-2.0"}, None, client=fake
    )
    assert vo.status == "queued"  # async: submit does NOT block on the render
    assert decode_video_id_with_provider(vo.id)["video_id"] == "task-1"
    assert decode_video_id_with_provider(vo.id)["model_id"] == "seedance-2.0"
    assert "/api/task/generation/progress" not in [c[0] for c in fake.calls]


def test_video_generation_capacity_failure_propagates_for_fallback():
    # 算力不足 at CREATE stays a retryable 5xx LibTVError so the router falls over
    # to the next libtv account / wavespeed (NOT a content-policy short-circuit).
    import litellm

    routes = _submit_routes()
    routes["/api/task/generation/create"] = {"code": 1200000136, "data": None, "msg": "算力不足"}
    fake = FakeSyncClient(post_by_path=routes, get_payload=_tool_spec_payload(auto_compliance=False))
    with pytest.raises(LibTVError) as ei:
        LibTVLLM(poll_interval=0).video_generation(
            "star-video2", "a fox", "tok", None, {"webid": "w"}, None, client=fake
        )
    assert not isinstance(ei.value, litellm.ContentPolicyViolationError)
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_avideo_generation_returns_queued_without_polling():
    fake = FakeAsyncClient(post_by_path=_submit_routes(), get_payload=_tool_spec_payload(auto_compliance=False))
    vo = await LibTVLLM(poll_interval=0).avideo_generation(
        "star-video2", "a fox", "tok", None, {"webid": "w", "libtv_status_model": "_fallback2/seedance-2.0"}, None, client=fake
    )
    assert vo.status == "queued"
    assert decode_video_id_with_provider(vo.id)["model_id"] == "_fallback2/seedance-2.0"
    assert "/api/task/generation/progress" not in [c[0] for c in fake.calls]


# --- async video: client split into create (no poll) + poll_once (single tick) ---

def test_create_does_not_poll_and_returns_task_id():
    client = FakeSyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p9", "teamId": 7}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {"taskId": "task-async-1"}},
        }
    )
    lt = LibTVClient(token="t", webid="w", sync_client=client, poll_interval=0)
    out = lt.create("star-video2-fast", "star-video2-fast", "video", {"prompt": "x"}, "proj")
    assert out == {"task_id": "task-async-1", "project_uuid": "p9", "node_key": out["node_key"]}
    paths = [c[0] for c in client.calls]
    assert paths == [
        "/api/canvas/project/create",
        "/api/canvas/nodes/batch",
        "/api/task/generation/create",
    ]
    assert "/api/task/generation/progress" not in paths  # create MUST NOT poll


def test_poll_once_single_progress_call_maps_state():
    for raw, expect in [
        ({"code": 0, "data": {"progresses": [{"status": 1}]}}, {"status": 1, "urls": [], "failed_reason": None}),
        (
            {"code": 0, "data": {"progresses": [{"status": 2, "taskResult": json.dumps({"videos": [{"videoUrl": "https://x/v.mp4"}]})}]}},
            {"status": 2, "urls": ["https://x/v.mp4"], "failed_reason": None},
        ),
        (
            {
                "code": 0,
                "data": {
                    "progresses": [
                        {
                            "status": 3,
                            "failedReason": "生成视频可能涉及版权限制",
                            "startTimeMs": 1700000000000,
                        }
                    ]
                },
            },
            {"status": 3, "urls": [], "failed_reason": "生成视频可能涉及版权限制"},
        ),
    ]:
        client = FakeSyncClient(post_by_path={"/api/task/generation/progress": raw})
        lt = LibTVClient(token="t", webid="w", sync_client=client, poll_interval=0)
        assert lt.poll_once("task-async-1", "video") == expect
        assert [c[0] for c in client.calls] == ["/api/task/generation/progress"]


@pytest.mark.asyncio
async def test_acreate_does_not_poll():
    client = FakeAsyncClient(
        post_by_path={
            "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p9"}}},
            "/api/canvas/nodes/batch": {"code": 0, "data": {}},
            "/api/task/generation/create": {"code": 0, "data": {"taskId": "task-async-2"}},
        }
    )
    lt = LibTVClient(token="t", webid="w", async_client=client, poll_interval=0)
    out = await lt.acreate("star-video2-fast", "star-video2-fast", "video", {"prompt": "x"}, "proj")
    assert out["task_id"] == "task-async-2"
    assert "/api/task/generation/progress" not in [c[0] for c in client.calls]
