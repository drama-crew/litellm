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
    _infer_video_mode,
    _reference_payload,
)
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


def test_build_video_object_encodes_url_into_id():
    llm = LibTVLLM()
    vo = llm._build_video_object("star-video2", {"urls": ["https://libtv-res/x.mp4"], "task_id": "t1"})
    assert vo.status == "completed"
    decoded = decode_video_id_with_provider(vo.id)
    assert decoded["custom_llm_provider"] == LIBTV_PROVIDER
    assert decoded["video_id"] == "https://libtv-res/x.mp4"
    assert vo._hidden_params["url"] == "https://libtv-res/x.mp4"


def test_video_status_returns_completed_with_url():
    llm = LibTVLLM()
    vid = llm._build_video_object("m", {"urls": ["https://libtv-res/v.mp4"]}).id
    status = llm.video_status(vid, None, None, {}, None)
    assert status.status == "completed"
    assert status._hidden_params["url"] == "https://libtv-res/v.mp4"


def test_video_content_downloads_decoded_url():
    llm = LibTVLLM()
    vid = llm._build_video_object("m", {"urls": ["https://libtv-res/v.mp4"]}).id

    class _DLClient:
        def __init__(self):
            self.got = None

        def get(self, url):
            self.got = url

            class _R:
                status_code = 200
                content = b"MP4BYTES"

            return _R()

    dl = _DLClient()
    data = llm.video_content(vid, None, None, {}, None, client=dl)
    assert data == b"MP4BYTES"
    assert dl.got == "https://libtv-res/v.mp4"


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


def _tool_spec_payload(model_key="star-video2", auto_compliance=True):
    props = {
        "ratio": {"default": "9:16", "enum": ["16:9", "9:16"]},
        "resolution": {"default": "720p", "enum": [{"value": "720p"}]},
        "duration": {"default": 5},
        "portrait": True,
    }
    if auto_compliance:
        props["autoCompliance"] = {"enable": True, "default": 1}
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
    assert vo.status == "completed"
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
    assert vo.status == "completed"
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
    assert vo.status == "completed"
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
    assert vo.status == "completed"
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
    assert vo.status == "completed"
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
    assert vo.status == "completed"
    creates = [body for path, body in fake.calls if path == "/api/third_asset/create"]
    assert next(c for c in creates if c["assetType"] == "video")["assetUrl"] == _LIBTV_VIDEO
    gen_params = next(body for path, body in fake.calls if path == "/api/task/generation/create")["params"]
    assert gen_params["mixedList"] == [
        {"url": "asset://asset-IMG", "type": "image"},
        {"url": "asset://asset-VID", "type": "video"},
    ]


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
        {"urls": ["http://x/v.mp4"]},
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
    assert vo.status == "completed"
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
    assert vo.status == "completed"
    gen_params = _gen_params(fake.calls)
    assert gen_params["imageList"] == ["https://libtv-res.liblib.art/up/img.png"]  # uploaded, not the external url
    assert "/api/community/image/verify" not in [u.split("api.liblib.tv", 1)[-1] for u, _ in fake.calls]
