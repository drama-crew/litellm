import json

import pytest

from litellm.exceptions import BadRequestError
from litellm.llms.libtv.handler import LibTVLLM
from litellm.llms.libtv.transform import build_topaz_upscale_params

_TOPAZ_SOURCE_URL = "https://libtv-res.liblib.art/upload-images/uid/source.mp4"


def _topaz_tool_spec_payload():
    meta = {
        "modelKey": "topaz-video-upscaler",
        "modelVendor": "topazlabs",
        "properties": {
            "resolution": {"default": "1080p", "enum": ["1080p", "2K", "4K"]},
            "specifiedModel": {"default": "prob-4", "enum": ["apo-8", "prob-4"]},
            "fps": {"default": 30, "enum": [24, 30, 60, 90, 120]},
            "slowmo": {"default": "1", "enum": ["1", "2", "3", "5"]},
        },
        "config": {"settings": ["resolution", "specifiedModel", "fps", "slowmo"]},
    }
    return {"data": {"tools": [{"type": "video", "metadata": json.dumps(meta)}]}}


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.headers = headers or {}

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
        item = queue.pop(0) if isinstance(queue, list) else queue
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append((self._path(url), None))
        return FakeResponse(self.get_payload)


class FakeAsyncClient(FakeSyncClient):
    async def post(self, url, json=None, headers=None, timeout=None):
        return FakeSyncClient.post(self, url, json, headers, timeout)

    async def get(self, url, headers=None, timeout=None, params=None):
        return FakeSyncClient.get(self, url, headers, timeout, params)


_CREATE_ROUTES = {
    "/api/canvas/project/create": {"code": 0, "data": {"projectMeta": {"uuid": "p1"}}},
    "/api/canvas/nodes/batch": {"code": 0, "data": {}},
    "/api/task/generation/create": {"code": 0, "data": {"taskId": "t1"}},
}


def _gen_params(calls):
    return next(j["params"] for path, j in calls if path == "/api/task/generation/create")


def test_build_topaz_upscale_params_defaults():
    params = build_topaz_upscale_params("upscale me", {})
    assert params["prompt"] == "upscale me"
    assert params["resolution"] == "1080p"
    assert params["specifiedModel"] == "prob-4"
    assert params["slowmo"] == "1"
    assert "modeType" not in params
    assert "fps" not in params  # only present when specifiedModel != prob-4


def test_build_topaz_upscale_params_explicit_values():
    params = build_topaz_upscale_params(
        "x",
        {"resolution": "4K", "specifiedModel": "apo-8", "fps": 60, "slowmo": "3"},
    )
    assert params["resolution"] == "4K"
    assert params["specifiedModel"] == "apo-8"
    assert params["fps"] == 60
    assert params["slowmo"] == "3"


def test_build_topaz_upscale_params_invalid_values_fall_back_to_defaults():
    params = build_topaz_upscale_params("x", {"resolution": "8K", "specifiedModel": "bogus", "slowmo": "9"})
    assert params["resolution"] == "1080p"
    assert params["specifiedModel"] == "prob-4"
    assert params["slowmo"] == "1"


def test_build_topaz_upscale_params_fps_only_present_for_non_prob4_model():
    params = build_topaz_upscale_params("x", {"specifiedModel": "prob-4", "fps": 60})
    assert "fps" not in params

    params2 = build_topaz_upscale_params("x", {"specifiedModel": "apo-8", "fps": 24})
    assert params2["fps"] == 24


def test_build_topaz_upscale_params_invalid_fps_falls_back_to_30():
    params = build_topaz_upscale_params("x", {"specifiedModel": "apo-8", "fps": 999})
    assert params["fps"] == 30


def test_video_generation_topaz_upscale_sets_no_mode_type_and_uploads_source_video():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "video_references": _TOPAZ_SOURCE_URL},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert "modeType" not in gen_params
    assert gen_params["videoList"] == [_TOPAZ_SOURCE_URL]
    assert gen_params["resolution"] == "1080p"
    assert gen_params["specifiedModel"] == "prob-4"
    assert gen_params["slowmo"] == "1"
    assert "fps" not in gen_params


def test_video_generation_topaz_upscale_accepts_input_reference_alias():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "input_reference": _TOPAZ_SOURCE_URL},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert gen_params["videoList"] == [_TOPAZ_SOURCE_URL]


def test_video_generation_topaz_upscale_forwards_settings():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    llm.video_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {
            "webid": "w",
            "video_references": _TOPAZ_SOURCE_URL,
            "resolution": "4K",
            "specifiedModel": "apo-8",
            "fps": 60,
            "slowmo": "5",
        },
        None,
        client=fake,
    )
    gen_params = _gen_params(fake.calls)
    assert gen_params["resolution"] == "4K"
    assert gen_params["specifiedModel"] == "apo-8"
    assert gen_params["fps"] == 60
    assert gen_params["slowmo"] == "5"


def test_video_generation_topaz_upscale_missing_source_video_raises_bad_request():
    fake = FakeSyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    with pytest.raises(BadRequestError):
        llm.video_generation(
            "topaz-video-upscaler", "upscale this", "tok", None, {"webid": "w"}, None, client=fake
        )


@pytest.mark.asyncio
async def test_avideo_generation_topaz_upscale_sets_no_mode_type_and_uploads_source_video():
    fake = FakeAsyncClient(post_by_path=_CREATE_ROUTES, get_payload=_topaz_tool_spec_payload())
    llm = LibTVLLM(poll_interval=0)
    vo = await llm.avideo_generation(
        "topaz-video-upscaler",
        "upscale this",
        "tok",
        None,
        {"webid": "w", "video_references": _TOPAZ_SOURCE_URL},
        None,
        client=fake,
    )
    assert vo.status == "queued"
    gen_params = _gen_params(fake.calls)
    assert "modeType" not in gen_params
    assert gen_params["videoList"] == [_TOPAZ_SOURCE_URL]
