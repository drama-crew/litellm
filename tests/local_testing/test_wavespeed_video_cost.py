import litellm
from litellm.cost_calculator import completion_cost
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
