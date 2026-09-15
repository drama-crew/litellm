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
    assert task_type_for_references((_ref("reference"),)) == TASK_TYPE_VIDEO_GENERATE_REF2VA


@pytest.mark.parametrize("count", [1, 5, 9])
def test_any_supported_reference_count_routes_to_ref2va(count):
    references = tuple(_ref("reference") for _ in range(count))
    assert task_type_for_references(references) == TASK_TYPE_VIDEO_GENERATE_REF2VA


def test_keyframe_roles_stay_on_the_shared_stream():
    """首尾帧走的是 vdn8 那条既有链路，不能被改道。"""
    assert task_type_for_references((_ref("first_frame"), _ref("last_frame"))) == TASK_TYPE_VIDEO_GENERATE


def test_no_references_stays_on_the_shared_stream():
    """纯文生视频同样走既有链路。"""
    assert task_type_for_references(()) == TASK_TYPE_VIDEO_GENERATE


def test_the_two_streams_are_distinct():
    # 同一条流就等于没有隔离——这是分流存在的全部理由。
    assert stream_key(TASK_TYPE_VIDEO_GENERATE_REF2VA) != stream_key(TASK_TYPE_VIDEO_GENERATE)


def test_ref2va_stream_name_matches_the_platform_registry():
    """跨仓库的字符串契约：平台侧 protocol.py 的 TASK_TYPE_VIDEO_GENERATE_REF2VA
    必须逐字相同，否则任务进了一条没有消费者的流，表现为永远 pending。"""
    assert TASK_TYPE_VIDEO_GENERATE_REF2VA == "video_generate_ref2va"
    assert stream_key(TASK_TYPE_VIDEO_GENERATE_REF2VA) == ("worker:tasks:video_generate_ref2va")


class TestCausynAdmissionHonoursRouting:
    """causyn-1.1 的准入路径绕过了通用的 xadd，必须自己带上路由结果。

    `enqueue_video_generate` 对 causyn-1.1 会走 `admit_video()` 然后直接 return，
    根本到不了后面那两处 `stream_key(task_type)` 的 xadd。而 `admit_video` 曾把
    "worker:tasks:video_generate" 写死在 Lua 调用里，于是信封的 type 是
    video_generate_ref2va、XADD 却落进共享流——ref2va 的活被投给了跑不了它的
    worker。2026-09-15 生产实证：ref2va 流 entries-added 恒为 0。
    """

    @staticmethod
    def _ref(role="reference"):
        return {"role": role, "media_type": "image", "url": "https://oss.example/a.jpg"}

    class _Redis:
        def __init__(self):
            self.eval_keys: list[str] = []

        async def eval(self, script, numkeys, *values):
            # KEYS[3] 是 XADD 的目标流。
            self.eval_keys.append(values[2])
            return 1

    async def _admit(self, references):
        import litellm.llms.causyn  # noqa: F401
        from litellm.llms.causyn.video_admission import admit_video
        from litellm.llms.libtv.video_generate import stream_key, task_type_for_references

        redis = self._Redis()
        await admit_video(
            redis,
            task_id="t",
            metadata="{}",
            envelope="{}",
            deadline=1.0,
            stream=stream_key(task_type_for_references(tuple(references))),
        )
        return redis.eval_keys[0]

    @pytest.mark.asyncio
    async def test_reference_roles_admit_onto_the_ref2va_stream(self):
        assert await self._admit([self._ref()]) == "worker:tasks:video_generate_ref2va"

    @pytest.mark.asyncio
    async def test_keyframes_still_admit_onto_the_shared_stream(self):
        assert await self._admit([self._ref("first_frame"), self._ref("last_frame")]) == "worker:tasks:video_generate"

    @pytest.mark.asyncio
    async def test_no_references_still_admit_onto_the_shared_stream(self):
        assert await self._admit([]) == "worker:tasks:video_generate"

    def test_admission_is_called_with_the_routed_stream(self):
        """防止下次有人把 admit_video 的调用点改回不传 stream。

        不传就会静默用回共享流的默认值——正是这次的故障形态。
        """
        import inspect

        import litellm.llms.causyn  # noqa: F401
        from litellm.llms.libtv import video_generate as vg

        source = inspect.getsource(vg.enqueue_video_generate)
        assert "admit_video(" in source
        assert "stream=stream_key(task_type)" in source
