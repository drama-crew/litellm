import httpx

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject


def test_wavespeed_video_cost_uses_duration_seconds() -> None:
    litellm.model_cost["wavespeed/bytedance/seedance-2.0-fast"] = {
        "litellm_provider": "wavespeed",
        "mode": "video_generation",
        "output_cost_per_video_per_second": 0.03,
    }
    video = VideoObject(
        id="video_x",
        object="video",
        status="queued",
        model="wavespeed/bytedance/seedance-2.0-fast",
    )
    video.usage = {"duration_seconds": 5.0}

    cost = completion_cost(
        completion_response=video,
        model="wavespeed/bytedance/seedance-2.0-fast",
        call_type="create_video",
    )

    assert round(cost, 4) == 0.15


def test_wavespeed_video_cost_is_metered_when_seconds_omitted() -> None:
    litellm.model_cost["wavespeed/bytedance/seedance-2.0-fast"] = {
        "litellm_provider": "wavespeed",
        "mode": "video_generation",
        "output_cost_per_video_per_second": 0.03,
    }
    config = WaveSpeedVideoConfig()
    data, _files, _url = config.transform_video_create_request(
        model="bytedance/seedance-2.0-fast",
        prompt="POV walking through a rainy neon alley.",
        api_base="https://api.wavespeed.ai/api/v3",
        video_create_optional_request_params={
            "input_reference": "https://assets.local/first.png",
        },
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        headers={},
    )
    video = config.transform_video_create_response(
        model="wavespeed/bytedance/seedance-2.0-fast",
        raw_response=httpx.Response(200, json={"id": "task-1", "status": "created"}),
        logging_obj=None,
        custom_llm_provider="wavespeed",
        request_data=data,
    )

    cost = completion_cost(
        completion_response=video,
        model="wavespeed/bytedance/seedance-2.0-fast",
        call_type="create_video",
    )

    assert cost > 0
    assert round(cost, 4) == 0.15
