"""causyn video handler — the queue-backed provider's own contract.

The engine it sits on (llms/libtv/video_generate) is already covered by
tests/litellm/proxy/video_endpoints. What is new here is the handler: what it
puts on the queue, what it hands back, and the two things it deliberately
refuses to do (sign object-store URLs, or run while the gray-rollout flag is
off).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import litellm.llms.causyn.handler as mod
from litellm.llms.custom_llm import CustomLLMError
from litellm.llms.causyn.handler import (
    LEGACY_VIDEO_ID_PREFIX,
    PROVIDER,
    CausynVideoHandler,
)
from litellm.llms.libtv.video_generate import VideoGenerateError
from litellm.types.videos.utils import decode_video_id_with_provider

TASK_ID = "0123456789abcdef0123456789abcdef"
VALID_LEGACY_ID = f"{LEGACY_VIDEO_ID_PREFIX}{TASK_ID}"


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
        "seconds": "5",
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
    with pytest.raises(CustomLLMError) as exc:
        await CausynVideoHandler().avideo_generation(
            model="causyn-1.0", prompt="a cat", api_key=None, api_base=None, optional_params=_params(), logging_obj=None
        )
    assert exc.value.status_code == 403
    assert enqueued.payloads == []


# --------------------------------------------------------------------------
# What goes on the queue
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueues_without_a_staging_upload(enqueued):
    """The platform's worker-runner presigns and injects it at claim time; this
    side holds no object-store credentials (design §3.2c)."""
    await CausynVideoHandler().avideo_generation(
        model="causyn-1.0", prompt="a cat", api_key=None, api_base=None, optional_params=_params(), logging_obj=None
    )
    (payload,) = enqueued.payloads
    assert "staging_upload" not in payload


@pytest.mark.asyncio
async def test_maps_openai_params_onto_the_task_envelope(enqueued):
    await CausynVideoHandler().avideo_generation(
        model="causyn-1.0", prompt="a cat", api_key=None, api_base=None, optional_params=_params(), logging_obj=None
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
    shaped = [{"role": "reference", "media_type": "image", "url": "https://source.example/b.png"}]
    await CausynVideoHandler().avideo_generation(
        model="causyn-1.0",
        prompt="a cat",
        api_key=None,
        api_base=None,
        optional_params=_params(reference_images=None, references=shaped),
        logging_obj=None,
    )
    assert enqueued.payloads[0]["request"]["references"] == shaped


# --------------------------------------------------------------------------
# The id we hand back
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_id_we_hand_back_routes_back_to_us(enqueued):
    """The test this file was missing, and the omission was not academic.

    litellm/videos/main.py picks the handler for a retrieval by decoding the
    provider out of the video id; an id it cannot decode falls through to the
    default provider. The first cut returned a plain `causyn_<uuid>`, so in
    production every status poll was sent to api.openai.com and timed out
    there -- submission worked and nothing could ever be read back. Asserting
    the id's *shape* did not catch that; asserting it decodes does.
    """
    video = await CausynVideoHandler().avideo_generation(
        model="causyn-1.0", prompt="a cat", api_key=None, api_base=None, optional_params=_params(), logging_obj=None
    )
    decoded = decode_video_id_with_provider(video.id)
    assert decoded["custom_llm_provider"] == PROVIDER
    assert decoded["video_id"] == enqueued.payloads[0]["task_id"]


@pytest.mark.asyncio
async def test_the_id_names_no_third_party_and_no_pool_index(enqueued):
    """What made libtv's use of this encoding worth avoiding does not apply here.

    The encoding carries custom_llm_provider and model_id in plain base64. For
    libtv that means a third-party vendor name and an account-pool index. Ours
    says "causyn" -- our own name -- and deliberately leaves model_id empty.
    """
    import base64

    video = await CausynVideoHandler().avideo_generation(
        model="causyn-1.0", prompt="a cat", api_key=None, api_base=None, optional_params=_params(), logging_obj=None
    )
    body = video.id.split("_", 1)[1]
    plain = base64.b64decode(body + "=" * (-len(body) % 4)).decode("utf-8", "ignore")
    assert "libtv" not in plain and "wavespeed" not in plain
    assert "model_id:causyn-1-0" in plain


@pytest.mark.asyncio
async def test_legacy_ids_stay_queryable(monkeypatch):
    """Ids already handed out before the encoding was adopted must not become
    unreadable -- there were live tasks holding them."""
    # "claimed", not "running" -- the engine has no such status, a fiction the
    # new drift guard caught immediately.
    _stub_status(monkeypatch, {"ok": True, "task_id": TASK_ID, "status": "claimed"})
    video = await CausynVideoHandler().avideo_status(
        video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
    )
    assert video.status == "in_progress"


@pytest.mark.asyncio
async def test_status_rejects_a_foreign_video_id():
    with pytest.raises(CustomLLMError):
        await CausynVideoHandler().avideo_status(
            video_id="someothervendor_abc",
            api_key=None,
            api_base=None,
            optional_params={},
            logging_obj=None,
        )


# --------------------------------------------------------------------------
# Status translation
# --------------------------------------------------------------------------


def _stub_status(monkeypatch, body):
    async def _fetch(task_id, *, redis):
        return body

    monkeypatch.setattr(mod, "fetch_video_generate_status", _fetch)
    monkeypatch.setattr(mod, "_redis_factory", lambda: object())


def _valid_worker_result(**overrides):
    result = {
        "validation_version": "video-v1",
        "staging_key": f"staging/video-tasks/{TASK_ID}.mp4",
        "etag": "etag-1",
        "bytes": 123,
        "content_type": "video/mp4",
        "duration_seconds": 5.0,
        "width": 768,
        "height": 512,
        "sha256": "a" * 64,
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize(
    "engine_status,expected",
    [
        ("queued", "queued"),
        # "claimed" is the worker-has-it-and-is-rendering state. It was absent
        # from the translation table, so a task mid-render was reported as
        # "failed" -- terminal, meaning the platform would stop polling and
        # mark a perfectly healthy generation dead. Seen in production on the
        # 8-second job while the 3-second one had already succeeded.
        ("claimed", "in_progress"),
        ("succeeded", "completed"),
    ],
)
@pytest.mark.asyncio
async def test_status_translation(monkeypatch, engine_status, expected):
    result = _valid_worker_result()
    _stub_status(
        monkeypatch,
        {"ok": True, "task_id": TASK_ID, "status": engine_status, "result": result}
        if engine_status == "succeeded"
        else {"ok": True, "task_id": TASK_ID, "status": engine_status},
    )
    video = await CausynVideoHandler().avideo_status(
        video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
    )
    assert video.status == expected


@pytest.mark.asyncio
async def test_completed_status_carries_the_object_store_result(monkeypatch):
    """This is what lets the platform finalise with a server-side copy instead
    of pulling the bytes back through the proxy (design §3.2b)."""
    result = _valid_worker_result()
    _stub_status(monkeypatch, {"ok": True, "task_id": TASK_ID, "status": "succeeded", "result": result})
    video = await CausynVideoHandler().avideo_status(
        video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
    )
    assert video.object_store_result == result
    assert video.seconds == "5.0"


@pytest.mark.parametrize(
    "missing_field",
    [
        "validation_version",
        "staging_key",
        "etag",
        "bytes",
        "content_type",
        "duration_seconds",
        "width",
        "height",
        "sha256",
    ],
)
@pytest.mark.asyncio
async def test_completed_status_rejects_missing_worker_validation_field(monkeypatch, missing_field):
    result = _valid_worker_result()
    result.pop(missing_field)
    _stub_status(monkeypatch, {"ok": True, "task_id": TASK_ID, "status": "succeeded", "result": result})

    with pytest.raises(CustomLLMError) as exc:
        await CausynVideoHandler().avideo_status(
            video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
        )

    assert exc.value.status_code == 503


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bytes", 0),
        ("bytes", -1),
        ("content_type", "video/webm"),
        ("duration_seconds", 2.0),
        ("duration_seconds", 9.0),
        ("width", 769),
        ("height", 511),
        ("sha256", "g" * 64),
        ("sha256", "a" * 63),
    ],
)
@pytest.mark.asyncio
async def test_completed_status_rejects_invalid_worker_validation_field(monkeypatch, field, value):
    result = _valid_worker_result(**{field: value})
    _stub_status(monkeypatch, {"ok": True, "task_id": TASK_ID, "status": "succeeded", "result": result})

    with pytest.raises(CustomLLMError) as exc:
        await CausynVideoHandler().avideo_status(
            video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_failure_surfaces_the_engine_error(monkeypatch):
    _stub_status(
        monkeypatch,
        {
            "ok": False,
            "task_id": TASK_ID,
            "status": "failed",
            "error": {"code": "invalid_params", "message": "bad prompt"},
        },
    )
    video = await CausynVideoHandler().avideo_status(
        video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
    )
    assert video.status == "failed"
    assert video.error["code"] == "invalid_params"


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_refuses_when_no_signed_url_is_present(monkeypatch):
    """A content read fails closed when the internal refresh service is absent."""
    _stub_status(
        monkeypatch,
        {
            "ok": True,
            "task_id": TASK_ID,
            "status": "succeeded",
            "result": {
                "validation_version": "video-v1",
                "staging_key": "k",
                "etag": "etag-1",
                "content_type": "video/mp4",
                "bytes": 1,
                "duration_seconds": 5.0,
                "width": 768,
                "height": 512,
                "sha256": "a" * 64,
            },
        },
    )
    with pytest.raises(CustomLLMError) as exc:
        await CausynVideoHandler().avideo_content(
            video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
        )
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_content_refreshes_an_expired_result_url_before_fetching(monkeypatch):
    _stub_status(
        monkeypatch,
        {
            "ok": True,
            "task_id": TASK_ID,
            "status": "succeeded",
            "result": _valid_worker_result(),
        },
    )
    fetched = []

    async def refresh(task_id, timeout):
        assert task_id == TASK_ID
        return "https://target.example/fresh-url"

    async def content_get(url, timeout, follow_redirects):
        fetched.append(url)
        return SimpleNamespace(status_code=200, content=b"video")

    handler = CausynVideoHandler(refresh_staging_url=refresh, content_get=content_get)
    content = await handler.avideo_content(
        video_id=VALID_LEGACY_ID,
        api_key=None,
        api_base=None,
        optional_params={},
        logging_obj=None,
    )
    assert content == b"video"
    assert fetched == ["https://target.example/fresh-url"]


@pytest.mark.asyncio
async def test_content_refreshes_each_time_and_does_not_reuse_a_stale_url(monkeypatch):
    _stub_status(
        monkeypatch,
        {
            "ok": True,
            "task_id": TASK_ID,
            "status": "succeeded",
            "result": _valid_worker_result(staging_url="https://target.example/expired"),
        },
    )
    refreshed = []

    async def refresh(task_id, timeout):
        url = f"https://target.example/fresh-{len(refreshed)}"
        refreshed.append(url)
        return url

    async def content_get(url, timeout, follow_redirects):
        return SimpleNamespace(status_code=200, content=url.encode())

    handler = CausynVideoHandler(refresh_staging_url=refresh, content_get=content_get)
    first = await handler.avideo_content(VALID_LEGACY_ID, None, None, {}, None)
    second = await handler.avideo_content(VALID_LEGACY_ID, None, None, {}, None)
    assert first == b"https://target.example/fresh-0"
    assert second == b"https://target.example/fresh-1"


@pytest.mark.asyncio
async def test_content_hides_refresh_service_failures(monkeypatch):
    _stub_status(
        monkeypatch,
        {
            "ok": True,
            "task_id": TASK_ID,
            "status": "succeeded",
            "result": _valid_worker_result(),
        },
    )

    async def refresh(task_id, timeout):
        raise RuntimeError("internal URL and service key must not leak")

    handler = CausynVideoHandler(refresh_staging_url=refresh)
    with pytest.raises(CustomLLMError) as exc:
        await handler.avideo_content(VALID_LEGACY_ID, None, None, {}, None)
    assert exc.value.status_code == 502
    assert "internal URL" not in str(exc.value)


@pytest.mark.asyncio
async def test_refresh_uses_service_auth_and_rejects_a_wrong_returned_key(monkeypatch):
    monkeypatch.setenv("DRAMA_CAUSYN_PLATFORM_URL", "https://platform.example/")
    monkeypatch.setenv("DRAMA_CAUSYN_SERVICE_API_KEY", "service-secret")

    class _Response:
        status_code = 200

        def json(self):
            return {"staging_key": "staging/video-tasks/other.mp4", "url": "https://target.example/u"}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers):
            assert (
                url == "https://platform.example/api/service/causyn/tasks/0123456789abcdef0123456789abcdef/staging-url"
            )
            assert headers == {"X-Service-API-Key": "service-secret"}
            return _Response()

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kwargs: _Client())
    with pytest.raises(VideoGenerateError):
        await mod._default_refresh_staging_url(TASK_ID, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform_url",
    ["", "platform.example", "ftp://platform.example", "https:///missing-host"],
)
async def test_refresh_rejects_an_invalid_platform_url(monkeypatch, platform_url):
    monkeypatch.setenv("DRAMA_CAUSYN_PLATFORM_URL", platform_url)
    monkeypatch.setenv("DRAMA_CAUSYN_SERVICE_API_KEY", "service-secret")

    with pytest.raises(VideoGenerateError) as exc:
        await mod._default_refresh_staging_url(TASK_ID, None)

    assert exc.value.code == "misconfigured"


@pytest.mark.asyncio
async def test_refresh_rejects_a_missing_service_key(monkeypatch):
    monkeypatch.setenv("DRAMA_CAUSYN_PLATFORM_URL", "https://platform.example")
    monkeypatch.delenv("DRAMA_CAUSYN_SERVICE_API_KEY", raising=False)

    with pytest.raises(VideoGenerateError) as exc:
        await mod._default_refresh_staging_url(TASK_ID, None)

    assert exc.value.code == "misconfigured"


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
        api_key=None,
        api_base=None,
        optional_params=_params(),
        logging_obj=None,
    )

    assert decode_video_id_with_provider(video.id)["custom_llm_provider"] == PROVIDER
    assert len(redis.entries) == 1
    _, fields = redis.entries[0]
    payload = json.loads(fields["payload"] if "payload" in fields else fields[b"payload"])
    assert "staging_upload" not in payload
    assert payload["request"]["duration_seconds"] == 5
    assert payload["request"]["resolution"] == "768x512"


@pytest.mark.asyncio
async def test_an_unmapped_status_raises_rather_than_reporting_failure(monkeypatch):
    """The default that hid the bug above.

    `.get(raw, "failed")` turns anything unrecognised into a *terminal* state,
    so a status this module simply does not know about ends the job instead of
    being retried. Raising makes the poll fail and come back.
    """
    _stub_status(monkeypatch, {"ok": True, "task_id": TASK_ID, "status": "paused"})
    with pytest.raises(CustomLLMError) as exc:
        await CausynVideoHandler().avideo_status(
            video_id=VALID_LEGACY_ID, api_key=None, api_base=None, optional_params={}, logging_obj=None
        )
    assert exc.value.status_code == 503


def test_translation_table_covers_every_status_the_engine_can_return():
    """Drift guard, read off the engine rather than restated by hand.

    fetch_video_generate_status returns _STATUS_TO_PUBLIC's values for the
    in-flight states and the literals "succeeded"/"failed" for the terminal
    ones; cancellation and all error paths are folded into "failed" there. If
    the engine grows a status, this fails here instead of in production.
    """
    from litellm.llms.causyn.handler import _STATUS_TO_OPENAI
    from litellm.llms.libtv.video_generate import _STATUS_TO_PUBLIC

    engine_can_return = set(_STATUS_TO_PUBLIC.values()) | {"succeeded", "failed"}
    assert engine_can_return <= set(_STATUS_TO_OPENAI), f"unmapped: {engine_can_return - set(_STATUS_TO_OPENAI)}"
    assert set(_STATUS_TO_OPENAI) <= engine_can_return, (
        f"maps statuses the engine never emits: {set(_STATUS_TO_OPENAI) - engine_can_return}"
    )
