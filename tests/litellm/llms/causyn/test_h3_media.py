from __future__ import annotations

import base64
import io

import av
import httpx
import pytest
from PIL import Image

from litellm.llms.causyn.context_ir_callback import verify_callback
from litellm.llms.causyn.h3_media import (
    BASE_VIDEO_LIMITS,
    fetch_media,
    inspect_audio,
    inspect_video,
    prepare_image,
    prepare_media,
)
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


COVER_ART_AUDIO_FORMATS = {
    "mp3": {"container_format": "mp3", "codec": "mp3", "fmt": "s16p", "samples": 1152},
    "m4a": {"container_format": "mp4", "codec": "aac", "fmt": "fltp", "samples": 1024},
}


def audio_with_cover_art(container_format, codec, fmt, samples, sample_rate=44100):
    output = io.BytesIO()
    with av.open(output, "w", format=container_format) as container:
        astream = container.add_stream(codec, rate=sample_rate)
        vstream = container.add_stream("mjpeg", rate=1)
        vstream.width = vstream.height = 16
        vstream.pix_fmt = "yuvj420p"
        vstream.disposition = av.stream.Disposition.attached_pic
        for packet in vstream.encode(av.VideoFrame.from_image(Image.new("RGB", (16, 16), "red"))):
            container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
        frame = av.AudioFrame(format=fmt, layout="mono", samples=samples)
        frame.sample_rate = sample_rate
        frame.pts = 0
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in astream.encode(frame):
            container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    return output.getvalue()


def audio_with_real_video(container_format="mp4", codec="aac", fmt="fltp", samples=1024, sample_rate=44100):
    output = io.BytesIO()
    with av.open(output, "w", format=container_format) as container:
        astream = container.add_stream(codec, rate=sample_rate)
        vstream = container.add_stream("libx264", rate=24)
        vstream.width = vstream.height = 64
        vstream.pix_fmt = "yuv420p"
        for packet in vstream.encode(av.VideoFrame.from_image(Image.new("RGB", (64, 64), "red"))):
            container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
        frame = av.AudioFrame(format=fmt, layout="mono", samples=samples)
        frame.sample_rate = sample_rate
        frame.pts = 0
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in astream.encode(frame):
            container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    return output.getvalue()


def audio_with_two_audio_streams(sample_rate=44100):
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        streams = [container.add_stream("aac", rate=sample_rate) for _ in range(2)]
        for stream in streams:
            frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
            frame.sample_rate = sample_rate
            frame.pts = 0
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
    assert inspect_video(clip(), BASE_VIDEO_LIMITS) == 2
    for raw in (clip(seconds=1), clip(fps=20)):
        with pytest.raises(RewriteError):
            inspect_video(raw, BASE_VIDEO_LIMITS)


@pytest.mark.parametrize("name", sorted(AUDIO_FORMATS))
def test_audio_accepts_each_allowed_format(name):
    assert inspect_audio(audio(seconds=2, **AUDIO_FORMATS[name])) == pytest.approx(2, abs=0.1)


def test_audio_rejects_disallowed_container_format():
    with pytest.raises(RewriteError):
        inspect_audio(audio(seconds=2, container_format="matroska", codec="pcm_s16le"))


def test_audio_rejects_a_video_stream_disguised_as_audio():
    with pytest.raises(RewriteError):
        inspect_audio(clip())


@pytest.mark.parametrize("name", sorted(COVER_ART_AUDIO_FORMATS))
def test_audio_accepts_cover_art_alongside_the_audio_stream(name):
    """I-B: real-world MP3/M4A files commonly embed a cover-art thumbnail as an
    attached_pic video stream (ID3 APIC / MP4 cover art). ARK's own validation
    ignores attached_pic streams; ours must too, or every downloaded music file
    with album art gets rejected after the task was already accepted and paid for."""
    raw = audio_with_cover_art(**COVER_ART_AUDIO_FORMATS[name])
    assert inspect_audio(raw) >= 0


def test_audio_still_rejects_a_real_video_stream_even_with_an_audio_stream_present():
    with pytest.raises(RewriteError, match="must not contain a video stream"):
        inspect_audio(audio_with_real_video())


def test_audio_accepts_more_than_one_audio_stream():
    """ARK's _validate_audio only requires at least one decodable audio stream,
    not exactly one; a stricter check here would reject files ARK accepts."""
    assert inspect_audio(audio_with_two_audio_streams()) >= 0


def test_audio_rejects_clips_longer_than_15_seconds():
    with pytest.raises(RewriteError):
        inspect_audio(audio(seconds=16))


@pytest.mark.asyncio
async def test_audio_reference_size_cap_matches_arks_20mb_default():
    """I-B: LiteLLM capped reference audio at 50MB while ARK's default
    H3_ARK_MAX_AUDIO_REFERENCE_BYTES is 20MB, so a 20-50MB file would pass here
    and only fail at ARK after the rewrite already ran and was billed."""
    raw = b"0" * (20 * 1024 * 1024 + 1)
    url = "data:audio/wav;base64," + base64.b64encode(raw).decode()
    audio_item = {"type": "audio_url", "audio_url": {"url": url}}
    image_url = "data:image/png;base64," + base64.b64encode(picture()).decode()
    image_item = {"type": "image_url", "image_url": {"url": image_url}, "role": "reference_image"}
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(RewriteError, match="permitted size"):
            await prepare_media(client, request([image_item, audio_item]))


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


def causyn_request(content):
    return request(content).model_copy(update={"model": "causyn-1.1"})


def video_item(seconds):
    url = "data:video/mp4;base64," + base64.b64encode(clip(seconds=seconds)).decode()
    return {"type": "video_url", "video_url": {"url": url}}


def test_inspect_video_limits_are_required_and_tolerance_is_per_model():
    from litellm.llms.causyn.h3_media import BASE_VIDEO_LIMITS, CAUSYN_VIDEO_LIMITS

    assert inspect_video(clip(seconds=6), BASE_VIDEO_LIMITS) == 6
    with pytest.raises(RewriteError, match=r"measured 6.0s; each must be 2-5 seconds"):
        inspect_video(clip(seconds=6), CAUSYN_VIDEO_LIMITS)
    assert inspect_video(clip(seconds=5), CAUSYN_VIDEO_LIMITS) == 5


def _fractional_clip(seconds, fps=25):
    return clip_frames(round(seconds * fps), fps)


def clip_frames(frames, fps):
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = 256
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            for packet in stream.encode(av.VideoFrame.from_image(Image.new("RGB", (256, 256), "red"))):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def test_causyn_video_tolerance_is_point_one_second_and_base_keeps_epsilon():
    from litellm.llms.causyn.h3_media import BASE_VIDEO_LIMITS, CAUSYN_VIDEO_LIMITS

    assert inspect_video(_fractional_clip(5.04), CAUSYN_VIDEO_LIMITS) == pytest.approx(5.04, abs=0.02)
    with pytest.raises(RewriteError, match=r"measured 5\.2s; each must be 2-5 seconds"):
        inspect_video(_fractional_clip(5.2), CAUSYN_VIDEO_LIMITS)
    assert inspect_video(_fractional_clip(5.2), BASE_VIDEO_LIMITS) == pytest.approx(5.2, abs=0.02)


@pytest.mark.asyncio
async def test_causyn_1_1_rejects_a_reference_video_longer_than_five_seconds():
    async with httpx.AsyncClient(trust_env=False) as client:
        await prepare_media(client, causyn_request([video_item(5)]))
        with pytest.raises(RewriteError, match="each must be 2-5 seconds"):
            await prepare_media(client, causyn_request([video_item(6)]))
        # the public MiniMax-H3 / LibTV path keeps the 2-15 s window
        await prepare_media(client, request([video_item(6)]))


@pytest.mark.asyncio
async def test_causyn_1_1_total_reference_video_is_capped_at_five_seconds():
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(RewriteError, match="Combined reference video duration exceeds 5 seconds"):
            await prepare_media(client, causyn_request([video_item(3), video_item(3)]))
        await prepare_media(client, request([video_item(3), video_item(3)]))


def _image_ref(color):
    raw = io.BytesIO()
    Image.new("RGB", (1600, 1200), color).save(raw, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}, "role": "reference_image"}


def _decoded_size(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).size


@pytest.mark.asyncio
async def test_many_reference_images_are_downscaled_harder_and_keep_canvas_order():
    colors = ("red", "green", "blue", "yellow", "white", "black", "orange", "purple", "pink")
    async with httpx.AsyncClient(trust_env=False) as client:
        prepared = await prepare_media(client, request([_image_ref(c) for c in colors]))
        few = await prepare_media(client, request([_image_ref(c) for c in colors[:4]]))
    sizes = [_decoded_size(item.image_url.url) for item in prepared.ordered_media]
    assert len(sizes) == 9 and all(max(size) == 768 for size in sizes)
    assert all(max(_decoded_size(item.image_url.url)) == 1024 for item in few.ordered_media)
    texts = [
        part["text"]
        for part in prepared.user_content()
        if part["type"] == "text" and part["text"].endswith("reference image:\n")
    ]
    assert texts == [f"<Picture {n}> reference image:\n" for n in range(1, 10)]
    reds = [Image.open(io.BytesIO(base64.b64decode(i.image_url.url.split(",", 1)[1]))).getpixel((0, 0)) for i in prepared.ordered_media]
    assert reds[0][0] > 200 and reds[1][1] > 100 and reds[2][2] > 200
