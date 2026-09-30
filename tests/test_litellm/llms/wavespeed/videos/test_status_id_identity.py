import httpx
from unittest.mock import MagicMock

from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider


def _status(config, caller_id):
    logging_obj = MagicMock()
    logging_obj.optional_params = {"video_id": caller_id}
    raw = httpx.Response(
        200,
        json={"code": 200, "data": {"id": "tid1", "status": "completed", "outputs": ["http://x/y.mp4"]}},
        request=httpx.Request("GET", "http://x"),
    )
    return config.transform_video_status_retrieve_response(raw, logging_obj, "wavespeed")


def test_status_id_reencodes_with_model_of_caller_id():
    create_id = encode_video_id_with_provider("tid1", "wavespeed", "dep-1")
    assert _status(WaveSpeedVideoConfig(), create_id).id == create_id


def test_status_id_without_caller_id_stays_model_less():
    config = WaveSpeedVideoConfig()
    logging_obj = MagicMock()
    logging_obj.optional_params = {}
    raw = httpx.Response(
        200, json={"code": 200, "data": {"id": "tid1", "status": "completed"}}, request=httpx.Request("GET", "http://x")
    )
    video = config.transform_video_status_retrieve_response(raw, logging_obj, "wavespeed")
    assert decode_video_id_with_provider(video.id)["model_id"] == ""
