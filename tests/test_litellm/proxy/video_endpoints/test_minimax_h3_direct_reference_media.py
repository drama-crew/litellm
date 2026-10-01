from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest

from litellm.llms.causyn import public_media_policy as policy
from litellm.proxy.video_endpoints.minimax_h3_models import MiniMaxH3Create, MiniMaxH3DirectCreate

from .test_minimax_h3_public_media import HEADERS, png, data_url, stack  # noqa: F401

DIRECT = "/video/minimax-h3/direct/v2/video_generation"
IR = "/video/minimax-h3/v2/video_generation"


def item(kind: str, n: int) -> dict:
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": f"https://media.example/{n}.png"}, "role": "reference_image"}
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": f"https://media.example/{n}.mp4"}, "role": "reference_video"}
    return {"type": "audio_url", "audio_url": {"url": f"https://media.example/{n}.mp3"}, "role": "reference_audio"}


def payload(*kinds: str, **extra) -> dict:
    content = [{"type": "text", "text": "A scene."}] + [item(kind, n) for n, kind in enumerate(kinds)]
    return {"model": "minimax-h3", "content": content, "resolution": "768P", "duration": 5, "ratio": "16:9", **extra}


@pytest.mark.parametrize(
    "kinds",
    [
        ("image", "video"),
        ("image", "audio"),
        ("video", "audio"),
        ("video",) * 3,
        ("video",) * 3 + ("audio",) * 3,
        ("image",) * 9 + ("video",) * 3,
        ("image", "video", "audio", "image", "audio"),
    ],
)
def test_direct_accepts(kinds):
    assert MiniMaxH3DirectCreate.model_validate(payload(*kinds))


REJECTS = [
    (("audio",), {}, "reference audio requires at least one reference image or video"),
    (("audio",) * 3, {}, "reference audio requires at least one reference image or video"),
    (("video",) * 4, {}, "reference count exceeds 9 images, 3 videos or 3 audio clips"),
    (("video", "audio") + ("audio",) * 3, {}, "reference count exceeds 9 images, 3 videos or 3 audio clips"),
    (("image",) * 10, {}, "reference count exceeds 9 images, 3 videos or 3 audio clips"),
    (("image",) * 9 + ("video",) * 3 + ("audio",), {}, "at most 12 reference items are allowed"),
    (("video",), {"ratio": "adaptive"}, "reference media requires an explicit ratio"),
    (("audio", "image"), {"ratio": "adaptive"}, "reference media requires an explicit ratio"),
]


@pytest.mark.parametrize("kinds,extra,message", REJECTS)
def test_direct_rejects(kinds, extra, message):
    with pytest.raises(ValueError) as caught:
        MiniMaxH3DirectCreate.model_validate(payload(*kinds, **extra))
    assert caught.value.errors()[0]["msg"] == "Value error, " + message


@pytest.mark.parametrize(
    "kinds,extra,message",
    [r for r in REJECTS if r[0] in (("audio",), ("video",) * 4) or r[1]],
)
def test_direct_endpoint_rejections(stack, kinds, extra, message):
    client, _, state = stack
    response = client.post(DIRECT, json=payload(*kinds, **extra), headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Value error, " + message
    assert not state["platform"]


def test_direct_rejects_video_mixed_with_first_frame():
    data = payload("video")
    data["content"].append(
        {"type": "image_url", "image_url": {"url": "https://media.example/f.png"}, "role": "first_frame"}
    )
    with pytest.raises(ValueError) as caught:
        MiniMaxH3DirectCreate.model_validate(data)
    assert caught.value.errors()[0]["msg"] == "Value error, first/last frames and reference media cannot be mixed"


@pytest.mark.parametrize("kinds", [("video",), ("audio", "image"), ("image", "video", "audio"), ("video",) * 3])
def test_ir_model_accepts_video_audio_with_the_same_limits_as_direct(kinds):
    assert MiniMaxH3Create.model_validate(payload(*kinds))


@pytest.mark.parametrize("kinds,extra,message", REJECTS)
def test_ir_model_rejects_with_the_same_messages_as_direct(kinds, extra, message):
    with pytest.raises(ValueError) as caught:
        MiniMaxH3Create.model_validate(payload(*kinds, **extra))
    assert caught.value.errors()[0]["msg"] == "Value error, " + message


def test_internal_body_order_and_shape():
    kinds = ("video", "image", "audio", "video", "image", "audio")
    body = MiniMaxH3DirectCreate.model_validate(payload(*kinds)).internal_body()
    assert body["references"] == [
        {
            "role": "reference",
            "media_type": kind,
            "url": f"https://media.example/{n}." + {"image": "png", "video": "mp4", "audio": "mp3"}[kind],
        }
        for n, kind in enumerate(kinds)
    ]
    assert body["model"] == "causyn-1.1" and body["aspect_ratio"] == "16:9" and body["generate_audio"] is True


def test_image_only_body_unchanged_on_direct():
    body = MiniMaxH3DirectCreate.model_validate(payload("image", "image")).internal_body()
    assert [r["media_type"] for r in body["references"]] == ["image", "image"]


@pytest.mark.parametrize(
    "kinds,extra,message",
    [r for r in REJECTS if r[0] in (("audio",), ("video",) * 4) or r[1]],
)
def test_ir_endpoint_applies_the_direct_limits(stack, kinds, extra, message):
    client, _, state = stack
    response = client.post(IR, json=payload(*kinds, **extra), headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Value error, " + message
    assert not state["platform"]


def test_ir_endpoint_accepts_video_and_audio_without_direct_prompt_processing(stack, monkeypatch):
    client, _, state = stack
    seen = []

    async def spy(data):
        seen.append(data)

    monkeypatch.setattr(policy, "validate_public_payload", spy)
    response = client.post(IR, json=payload("image", "video", "audio"), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert [r["media_type"] for r in seen[0]["references"]] == ["image", "video", "audio"]
    sent = state["platform"][-1][1]["payload"]
    assert "prompt_processing" not in sent, "the IR prefix must be rewritten, not direct"
    assert [r["media_type"] for r in sent["references"]] == ["image", "video", "audio"]


def test_direct_prefix_routes_video_audio_with_direct_prompt_processing(stack, monkeypatch):
    client, _, state = stack
    seen = []

    async def spy(data):
        seen.append(data)

    monkeypatch.setattr(policy, "validate_public_payload", spy)
    response = client.post(DIRECT, json=payload("image", "video", "audio"), headers=HEADERS)
    assert response.status_code == 200, response.text
    assert [r["media_type"] for r in seen[0]["references"]] == ["image", "video", "audio"]
    sent = state["platform"][-1][1]["payload"]
    assert sent["prompt_processing"] == "direct"
    assert [r["media_type"] for r in sent["references"]] == ["image", "video", "audio"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_submit_policy_rejects_short_inline_video(stack, monkeypatch):
    client, _, state = stack

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "short.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=320x320:r=30:d=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            check=True,
        )
        url = "data:video/mp4;base64," + base64.b64encode(out.read_bytes()).decode()
    data = payload("image")
    data["content"].append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
    response = client.post(DIRECT, json=data, headers=HEADERS)
    assert response.status_code == 400, response.text
    assert "reference video 1" in response.json()["error"]["message"]
    assert not state["platform"]
