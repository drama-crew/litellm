"""Ref2VA 提交必须进它自己的队列。

领任务只按 task_type 过滤，没有模型维度，而一个 vLLM 进程只能服务一种
task-type。ref2va 任务若进共享的 video_generate 流，接 vdn8 后端的
video-worker-01 就会领到它并失败；反过来 ref2va worker 也会抢走 t2v 任务。
所以分流这件事是在入队时决定的，这里钉住它。
"""

from __future__ import annotations

import pytest

# video_generate 与 causyn.topaz 之间有既有的循环导入（video_generate:24 →
# causyn/__init__ → handler → topaz → 回到 video_generate）。先把 causyn 这一侧
# 导完，环就不会在 video_generate 半初始化时被撞上。与本次改动无关。
import litellm.llms.causyn  # noqa: F401
from litellm.llms.libtv.video_generate import (
    TASK_TYPE_VIDEO_GENERATE,
    TASK_TYPE_VIDEO_GENERATE_REF2VA,
    stream_key,
    task_type_for_references,
)


def _ref(role: str) -> dict[str, str]:
    return {"role": role, "media_type": "image", "url": "https://oss.example/a.jpg"}


def test_reference_roles_route_to_the_ref2va_stream():
    assert (
        task_type_for_references((_ref("reference"),))
        == TASK_TYPE_VIDEO_GENERATE_REF2VA
    )


@pytest.mark.parametrize("count", [1, 5, 9])
def test_any_supported_reference_count_routes_to_ref2va(count):
    references = tuple(_ref("reference") for _ in range(count))
    assert task_type_for_references(references) == TASK_TYPE_VIDEO_GENERATE_REF2VA


def test_keyframe_roles_stay_on_the_shared_stream():
    """首尾帧走的是 vdn8 那条既有链路，不能被改道。"""
    assert (
        task_type_for_references((_ref("first_frame"), _ref("last_frame")))
        == TASK_TYPE_VIDEO_GENERATE
    )


def test_no_references_stays_on_the_shared_stream():
    """纯文生视频同样走既有链路。"""
    assert task_type_for_references(()) == TASK_TYPE_VIDEO_GENERATE


def test_the_two_streams_are_distinct():
    # 同一条流就等于没有隔离——这是分流存在的全部理由。
    assert stream_key(TASK_TYPE_VIDEO_GENERATE_REF2VA) != stream_key(
        TASK_TYPE_VIDEO_GENERATE
    )


def test_ref2va_stream_name_matches_the_platform_registry():
    """跨仓库的字符串契约：平台侧 protocol.py 的 TASK_TYPE_VIDEO_GENERATE_REF2VA
    必须逐字相同，否则任务进了一条没有消费者的流，表现为永远 pending。"""
    assert TASK_TYPE_VIDEO_GENERATE_REF2VA == "video_generate_ref2va"
    assert stream_key(TASK_TYPE_VIDEO_GENERATE_REF2VA) == (
        "worker:tasks:video_generate_ref2va"
    )
