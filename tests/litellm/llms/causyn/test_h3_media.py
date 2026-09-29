from __future__ import annotations

import base64
import io

import av
import httpx
import pytest
from PIL import Image

from litellm.llms.causyn.context_ir_callback import verify_callback
from litellm.llms.causyn.h3_media import fetch_media, inspect_audio, inspect_video, prepare_image, prepare_media
from litellm.llms.causyn.h3_prompt import ContextIRRequest, RewriteError, validate_prompt


def picture(size=(512, 512), format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", size, "red").save(output, format=format)
    return output.getvalue()


def clip(seconds=2, fps=24):
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = 256
        stream.pix_fmt = "yuv420p"
        for _ in range(seconds * fps):
            for packet in stream.encode(av.VideoFrame.from_image(Image.new("RGB", (256, 256), "red"))):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


AUDIO_FORMATS = {
    "wav": {"container_format": "wav", "codec": "pcm_s16le"},
    "mp3": {"container_format": "mp3", "codec": "mp3"},
    "aac": {"container_format": "adts", "codec": "aac"},
    "m4a": {"container_format": "ipod", "codec": "aac"},
    "flac": {"container_format": "flac", "codec": "flac"},
    "ogg": {"container_format": "ogg", "codec": "vorbis", "fmt": "fltp", "options": {"strict": "experimental"}},
}


def audio(seconds=2.0, container_format="wav", codec="pcm_s16le", fmt="s16", options=None, sample_rate=44100):
    output = io.BytesIO()
    with av.open(output, "w", format=container_format) as container:
        stream = container.add_stream(codec, rate=sample_rate, options=options or {})
        samples_per_frame = 1024
        total_samples = int(seconds * sample_rate)
        pts = 0
        for start in range(0, total_samples, samples_per_frame):
            frame = av.AudioFrame(format=fmt, layout="mono", samples=min(samples_per_frame, total_samples - start))
            frame.sample_rate = sample_rate
            frame.pts = pts
            pts += frame.samples
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def request(content):
    return ContextIRRequest.model_validate(
        {
            "model": "MiniMax-H3",
            "duration": 5,
            "ratio": "16:9",
            "content": [
                {"type": "text", "text": "Continue moving right."},
                *content,
            ],
        }
    )


def test_image_limits_are_checked_before_thumbnail():
    assert prepare_image(picture()).startswith("data:image/jpeg;base64,")
    for raw in (picture((255, 512)), picture((256, 1000)), picture(format="GIF")):
        with pytest.raises(RewriteError):
            prepare_image(raw)


def test_video_codec_geometry_fps_and_duration():
    assert inspect_video(clip()) == 2
    for raw in (clip(seconds=1), clip(fps=20)):
        with pytest.raises(RewriteError):
            inspect_video(raw)


@pytest.mark.parametrize("name", sorted(AUDIO_FORMATS))
def test_audio_accepts_each_allowed_format(name):
    assert inspect_audio(audio(seconds=2, **AUDIO_FORMATS[name])) == pytest.approx(2, abs=0.1)


def test_audio_rejects_disallowed_container_format():
    with pytest.raises(RewriteError):
        inspect_audio(audio(seconds=2, container_format="matroska", codec="pcm_s16le"))


def test_audio_rejects_a_video_stream_disguised_as_audio():
    with pytest.raises(RewriteError):
        inspect_audio(clip())


def test_audio_rejects_clips_longer_than_15_seconds():
    with pytest.raises(RewriteError):
        inspect_audio(audio(seconds=16))


@pytest.mark.asyncio
async def test_reference_bytes_are_frozen_and_total_duration_checked():
    raw = clip(seconds=6)
    url = "data:video/mp4;base64," + base64.b64encode(raw).decode()
    video = {"type": "video_url", "video_url": {"url": url}}
    async with httpx.AsyncClient(trust_env=False) as client:
        prepared = await prepare_media(client, request([video]))
        assert prepared.ordered_media[0].video_url.url == url
        with pytest.raises(RewriteError, match="Combined reference video duration"):
            await prepare_media(client, request([video, video, video]))
        with pytest.raises(RewriteError, match="permitted size"):
            await fetch_media(client, url, 4)


@pytest.mark.asyncio
async def test_mixed_reference_order_is_preserved_after_preparation():
    audio_url = "data:audio/wav;base64," + base64.b64encode(audio(seconds=2)).decode()
    image_url = "data:image/png;base64," + base64.b64encode(picture()).decode()
    image = {"type": "image_url", "image_url": {"url": image_url}, "role": "reference_image"}
    video_raw = clip(seconds=2)
    video = {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video_raw).decode()},
    }
    audio_item = {"type": "audio_url", "audio_url": {"url": audio_url}}
    async with httpx.AsyncClient(trust_env=False) as client:
        prepared = await prepare_media(client, request([audio_item, image, video]))
        assert [type(item).__name__ for item in prepared.content[1:]] == ["AudioItem", "ImageItem", "VideoItem"]
        assert prepared.content[1].audio_url.url == audio_url


@pytest.mark.asyncio
async def test_audio_reference_total_duration_is_checked_independently_of_video():
    url = "data:audio/wav;base64," + base64.b64encode(audio(seconds=6)).decode()
    image_url = "data:image/png;base64," + base64.b64encode(picture()).decode()
    image_item = {"type": "image_url", "image_url": {"url": image_url}, "role": "reference_image"}
    clip_item = {"type": "audio_url", "audio_url": {"url": url}}
    async with httpx.AsyncClient(trust_env=False) as client:
        prepared = await prepare_media(client, request([image_item, clip_item]))
        assert prepared.ordered_media[-1].audio_url.url == url
        with pytest.raises(RewriteError, match="Combined reference audio duration"):
            await prepare_media(client, request([image_item, clip_item, clip_item, clip_item]))


@pytest.mark.asyncio
async def test_private_reference_and_callback_are_rejected():
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(RewriteError, match="could not be validated"):
            await prepare_media(
                client, request([{"type": "image_url", "image_url": {"url": "http://127.0.0.1/internal"}}])
            )
    with pytest.raises(RewriteError, match="callback_url"):
        await verify_callback("http://127.0.0.1/internal")


def test_broken_dialogue_close_and_shot_order_are_rejected():
    for description in ("[Shot 1] The actor says <d>[English]Hello<d>.", "[Shot 2] A cat walks."):
        with pytest.raises(RewriteError):
            validate_prompt(
                f"integrated_multimodal_description: {description}\noverall_soundscape: N/A\nnon_diegetic_music: N/A",
                request([]),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [503, 429, "read", "timeout", 404])
async def test_reference_download_classifies_retryable_failure(monkeypatch, failure):
    import litellm.llms.causyn.h3_media as media

    monkeypatch.setattr(media, "validate_url", lambda url: (url, "source.example"))

    def transport(request):
        if failure == "read":
            raise httpx.ReadError("interrupted")
        if failure == "timeout":
            raise httpx.ReadTimeout("interrupted")
        return httpx.Response(failure)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(RewriteError) as error:
            await prepare_media(
                client, request([{"type": "image_url", "image_url": {"url": "https://source.example/ref.png"}}])
            )
    assert error.value.retryable is (failure != 404)
