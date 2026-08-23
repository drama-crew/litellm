"""causyn video handler — the queue-backed provider's own contract.

The engine it sits on (llms/libtv/video_generate) is already covered by
tests/litellm/proxy/video_endpoints. What is new here is the handler: what it
puts on the queue, what it hands back, and the two things it deliberately
refuses to do (sign object-store URLs, or run while the gray-rollout flag is
off).
"""

from __future__ import annotations

import json

import pytest

import litellm.llms.causyn.handler as mod
from litellm.llms.causyn.handler import VIDEO_ID_PREFIX, CausynVideoHandler
from litellm.llms.libtv.video_generate import VideoGenerateError


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_SOURCE_HOSTS", "source.example")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_TARGET_HOSTS", "target.example")
    monkeypatch.setenv("LIBTV_VIDEO_GENERATE_ALLOW_HTTP", "false")
    monkeypatch.setenv("DRAMA_INTERNAL_VIDEO_ENABLED", "true")


class _Recorder:
    """Captures the payload instead of reaching Redis."""

    def __init__(self):
        self.payloads = []

    async def __call__(self, payload, *, redis_factory, settings):
        self.payloads.append(payload)
        return payload["task_id"]


@pytest.fixture
def enqueued(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(mod, "enqueue_video_generate", rec)
    return rec


def _params(**over):
    base = {
        "seconds": 5,
        "size": "768x512",
        "aspect_ratio": "3:2",
        "reference_images": ["https://source.example/a.png"],
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# Gray rollout
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_while_the_gray_flag_is_off(monkeypatch, enqueued):
    """Registering in model_list publishes the model on /v1/models to every key
    allowed to call it. Without this the platform could still be refusing the
    model while a sandbox agent reached it directly."""
    monkeypatch.delenv("DRAMA_INTERNAL_VIDEO_ENABLED", raising=False)
    with pytest.raises(VideoGenerateError) as exc:
        await CausynVideoHandler().avideo_generation(
            model="causyn-1.0", prompt="a cat", optional_params=_params()
        )
    assert exc.value.code == "forbidden"
    assert enqueued.payloads == []


# --------------------------------------------------------------------------
# What goes on the queue
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueues_without_a_staging_upload(enqueued):
    """The platform's worker-runner presigns and injects it at claim time; this
    side holds no object-store credentials (design §3.2c)."""
    await CausynVideoHandler().avideo_generation(
        model="causyn-1.0", prompt="a cat", optional_params=_params()
    )
    (payload,) = enqueued.payloads
    assert "staging_upload" not in payload


@pytest.mark.asyncio
async def test_maps_openai_params_onto_the_task_envelope(enqueued):
    await CausynVideoHandler().avideo_generation(
        model="causyn-1.0", prompt="a cat", optional_params=_params()
    )
    request = enqueued.payloads[0]["request"]
    assert request["prompt"] == "a cat"
    assert request["duration_seconds"] == 5
    assert request["resolution"] == "768x512"
    assert request["ratio"] == "3:2"
    assert request["references"] == [
        {"role": "reference", "media_type": "image", "url": "https://source.example/a.png"}
    ]


@pytest.mark.asyncio
async def test_accepts_already_shaped_references(enqueued):
    """drama-cli and the platform build reference lists differently; neither
    should need a per-caller branch in here."""
    shaped = [
        {"role": "reference", "media_type": "image", "url": "https://source.example/b.png"}
    ]
    await CausynVideoHandler().avideo_generation(
        model="causyn-1.0",
        prompt="a cat",
        optional_params=_params(reference_images=None, references=shaped),
    )
    assert enqueued.payloads[0]["request"]["references"] == shaped


# --------------------------------------------------------------------------
# The id we hand back
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_id_reveals_no_internal_topology(enqueued):
    """The sibling libtv encoding base64s custom_llm_provider and the internal
    deployment id into the video id -- plain base64, so any caller can decode
    the vendor name and account-pool index. This one carries a task id we
    generated and nothing else."""
    import base64

    video = await CausynVideoHandler().avideo_generation(
        model="causyn-1.0", prompt="a cat", optional_params=_params()
    )
    assert video.id.startswith(VIDEO_ID_PREFIX)
    task_id = enqueued.payloads[0]["task_id"]
    assert video.id == f"{VIDEO_ID_PREFIX}{task_id}"
    for candidate in (video.id, video.id[len(VIDEO_ID_PREFIX) :]):
        try:
            decoded = base64.b64decode(candidate + "=" * (-len(candidate) % 4)).decode(
                "utf-8", "ignore"
            )
        except Exception:
            decoded = ""
        assert "provider" not in decoded and "causyn/" not in decoded


@pytest.mark.asyncio
async def test_status_rejects_a_foreign_video_id():
    with pytest.raises(VideoGenerateError):
        await CausynVideoHandler().avideo_status(video_id="video_someothervendor")


# --------------------------------------------------------------------------
# Status translation
# --------------------------------------------------------------------------


def _stub_status(monkeypatch, body):
    async def _fetch(task_id, *, redis):
        return body

    monkeypatch.setattr(mod, "fetch_video_generate_status", _fetch)
    monkeypatch.setattr(mod, "_redis_factory", lambda: object())


@pytest.mark.parametrize(
    "engine_status,expected",
    [("queued", "queued"), ("running", "in_progress"), ("succeeded", "completed")],
)
@pytest.mark.asyncio
async def test_status_translation(monkeypatch, engine_status, expected):
    result = {
        "staging_key": "staging/video-tasks/t.mp4",
        "content_type": "video/mp4",
        "bytes": 123,
        "duration_seconds": 5.0,
    }
    _stub_status(
        monkeypatch,
        {"ok": True, "task_id": "t", "status": engine_status, "result": result}
        if engine_status == "succeeded"
        else {"ok": True, "task_id": "t", "status": engine_status},
    )
    video = await CausynVideoHandler().avideo_status(video_id=f"{VIDEO_ID_PREFIX}t")
    assert video.status == expected


@pytest.mark.asyncio
async def test_completed_status_carries_the_object_store_result(monkeypatch):
    """This is what lets the platform finalise with a server-side copy instead
    of pulling the bytes back through the proxy (design §3.2b)."""
    result = {
        "staging_key": "staging/video-tasks/t.mp4",
        "content_type": "video/mp4",
        "bytes": 123,
        "duration_seconds": 5.0,
    }
    _stub_status(
        monkeypatch, {"ok": True, "task_id": "t", "status": "succeeded", "result": result}
    )
    video = await CausynVideoHandler().avideo_status(video_id=f"{VIDEO_ID_PREFIX}t")
    assert video.object_store_result == result
    assert video.seconds == "5.0"


@pytest.mark.asyncio
async def test_failure_surfaces_the_engine_error(monkeypatch):
    _stub_status(
        monkeypatch,
        {
            "ok": False,
            "task_id": "t",
            "status": "failed",
            "error": {"code": "invalid_params", "message": "bad prompt"},
        },
    )
    video = await CausynVideoHandler().avideo_status(video_id=f"{VIDEO_ID_PREFIX}t")
    assert video.status == "failed"
    assert video.error["code"] == "invalid_params"


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_refuses_when_no_signed_url_is_present(monkeypatch):
    """This module signs nothing. If the platform did not attach a URL there is
    nothing to fetch, and inventing one would mean putting object-store
    credentials in the proxy."""
    _stub_status(
        monkeypatch,
        {
            "ok": True,
            "task_id": "t",
            "status": "succeeded",
            "result": {
                "staging_key": "k",
                "content_type": "video/mp4",
                "bytes": 1,
                "duration_seconds": 5.0,
            },
        },
    )
    with pytest.raises(VideoGenerateError) as exc:
        await CausynVideoHandler().avideo_content(video_id=f"{VIDEO_ID_PREFIX}t")
    assert exc.value.code == "result_unavailable"


def test_module_holds_no_object_store_client():
    """Guards design §3.2c/§3.2d: signing and object-store access stay on the
    platform. An import of oss2/boto3 appearing here would mean the proxy had
    quietly grown a second, unverified path to the bucket."""
    source = open(mod.__file__, encoding="utf-8").read()
    assert "oss2" not in source
    assert "boto3" not in source


# --------------------------------------------------------------------------
# End-to-end through the real enqueue
# --------------------------------------------------------------------------


class _FakeRedis:
    """Enough of the client for enqueue_video_generate's happy path."""

    def __init__(self):
        self.entries = []
        self.kv = {}

    async def zrangebyscore(self, *a, **k):
        return [b"worker-1"]

    async def hgetall(self, *a, **k):
        return {b"worker-1": b"1"}

    async def set(self, key, value, **k):
        self.kv[key] = value
        return True

    async def xadd(self, stream, fields, **k):
        self.entries.append((stream, fields))
        return b"1-0"


@pytest.mark.asyncio
async def test_submit_survives_the_real_enqueue_without_a_staging_upload(monkeypatch):
    """Exercises enqueue_video_generate for real rather than stubbing it.

    The handler tests above replace that function wholesale, which is exactly
    how a KeyError inside it reached production: validation had been relaxed to
    let staging_upload be absent, but the envelope construction still indexed
    it directly, so every causyn submit failed with a bare APIConnectionError
    naming only the missing key. Mocking the callee hid the one place the bug
    lived.
    """
    redis = _FakeRedis()
    monkeypatch.setattr(mod, "_redis_factory", lambda: redis)

    video = await CausynVideoHandler().avideo_generation(
        model="causyn-1.0",
        prompt="a cat",
        optional_params=_params(),
    )

    assert video.id.startswith(VIDEO_ID_PREFIX)
    assert len(redis.entries) == 1
    _, fields = redis.entries[0]
    payload = json.loads(fields["payload"] if "payload" in fields else fields[b"payload"])
    assert "staging_upload" not in payload
    assert payload["request"]["duration_seconds"] == 5
    assert payload["request"]["resolution"] == "768x512"
