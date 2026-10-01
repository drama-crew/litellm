from __future__ import annotations

import io
import math
import os
import time

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from litellm.llms.causyn import ref_media_facts as rmf  # noqa: E402

W, H, FPS = 160, 90, 24


def _encode_video(frames: list, fps: int = FPS) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        try:
            stream = out.add_stream("libx264", rate=fps)
            stream.options = {"preset": "ultrafast", "crf": "18"}
        except Exception:
            stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return buf.getvalue()


def _textured(seed: int, base: int) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 90, (H, W, 3)) + base
    return np.clip(img, 0, 255).astype("uint8")


def _two_shot(seconds: float = 4.0, cut_at: float = 2.0) -> bytes:
    a, b = _textured(1, 20), _textured(2, 150)
    n = int(seconds * FPS)
    return _encode_video([a if i / FPS < cut_at else b for i in range(n)])


def _moving(seconds: float = 3.0) -> bytes:
    base = _textured(3, 60)
    n = int(seconds * FPS)
    return _encode_video([np.roll(base, i * 2, axis=1) for i in range(n)])


def test_hard_cut_detected() -> None:
    f = rmf.analyze_video(_two_shot())
    assert f.cuts is not None and len(f.cuts) == 1
    assert abs(f.cuts[0] - 2.0) <= 0.1
    assert (f.width, f.height) == (W, H)
    assert abs(f.duration - 4.0) < 0.2 and abs(f.fps - FPS) < 0.5
    assert {k.shot for k in f.keyframes} == {1, 2}
    assert all(k.jpeg[:2] == b"\xff\xd8" for k in f.keyframes)
    assert [k.t for k in f.keyframes] == sorted(k.t for k in f.keyframes)


def test_single_moving_shot_has_no_cuts() -> None:
    f = rmf.analyze_video(_moving())
    assert f.cuts == ()
    assert len(f.keyframes) >= 2 and all(k.shot == 1 for k in f.keyframes)


def test_crossfade_not_a_cut_storm() -> None:
    a, b = _textured(1, 20), _textured(2, 150)
    frames = []
    for i in range(4 * FPS):
        t = i / FPS
        mix = min(1.0, max(0.0, (t - 1.5) / 1.0))
        frames.append((a * (1 - mix) + b * mix).astype("uint8"))
    f = rmf.analyze_video(_encode_video(frames))
    assert f.cuts is not None and len(f.cuts) <= 1


def test_keyframe_cap_and_budget() -> None:
    a, b = _textured(1, 20), _textured(2, 150)
    frames = []
    for i in range(12 * FPS):
        frames.append(a if (i // FPS) % 2 == 0 else b)  # a cut every second
    raw = _encode_video(frames)
    assert len(rmf.analyze_video(raw, max_keyframes=8).keyframes) <= 8
    assert len(rmf.analyze_video(raw, max_keyframes=3).keyframes) <= 3
    assert [rmf.keyframe_budget(n) for n in (1, 2, 3, 4, 8, 16)] == [8, 8, 5, 4, 2, 1]
    assert rmf.keyframe_budget(0) == 8 and rmf.keyframe_budget(40) == 0


def test_keyframe_jpeg_short_side_capped() -> None:
    from PIL import Image

    big = [np.zeros((720, 1280, 3), dtype="uint8") for _ in range(6)]
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        s = out.add_stream("mpeg4", rate=24)
        s.width, s.height, s.pix_fmt = 1280, 720, "yuv420p"
        for arr in big:
            for p in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    f = rmf.analyze_video(buf.getvalue())
    assert f.keyframes
    assert min(Image.open(io.BytesIO(f.keyframes[0].jpeg)).size) == 448


def test_corrupt_video_degrades() -> None:
    f = rmf.analyze_video(b"definitely not a video" * 50)
    assert f.cuts is None and f.keyframes == ()
    f = rmf.analyze_video(b"")
    assert f.cuts is None


def test_truncated_video_does_not_raise() -> None:
    raw = _two_shot()
    f = rmf.analyze_video(raw[: len(raw) // 3])
    assert isinstance(f, rmf.VideoFacts)


def test_deadline_exceeded_returns_cuts_none() -> None:
    f = rmf.analyze_video(_two_shot(), deadline=time.monotonic() - 1)
    assert f.cuts is None and f.keyframes == ()
    assert f.duration > 0


def _wav_like(samples: "np.ndarray", sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="wav") as out:
        s = out.add_stream("pcm_s16le", rate=sr, layout="mono")
        pcm = (np.clip(samples, -1, 1) * 32767).astype("int16")[None, :]
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = sr
        for p in s.encode(frame):
            out.mux(p)
        for p in s.encode():
            out.mux(p)
    return buf.getvalue()


def test_sine_is_tonal() -> None:
    sr = 16000
    t = np.arange(sr * 4) / sr
    f = rmf.analyze_audio(_wav_like(0.5 * np.sin(2 * np.pi * 440 * t), sr))
    assert f.character == "tonal/music"
    assert abs(f.duration - 4.0) < 0.1 and f.sample_rate == sr and f.channels == 1
    assert len(f.loudness_db) == 8 and f.change_points == ()


def test_clicks_are_percussive() -> None:
    sr = 16000
    x = np.zeros(sr * 4)
    rng = np.random.default_rng(0)
    for k in range(32):  # 8 clicks per second
        i = int(k * sr / 8)
        x[i : i + 160] = rng.uniform(-1, 1, 160) * np.hanning(160)
    f = rmf.analyze_audio(_wav_like(x, sr))
    assert f.character == "percussive/rhythmic"
    assert f.onset_rate >= 3


def test_silence_is_quiet_and_change_points() -> None:
    sr = 16000
    t = np.arange(sr * 4) / sr
    x = 0.5 * np.sin(2 * np.pi * 300 * t)
    x[sr * 2 :] *= 0.01  # -40 dB step at 2.0 s
    f = rmf.analyze_audio(_wav_like(x, sr))
    assert f.change_points == (2.0,)
    assert rmf.analyze_audio(_wav_like(np.zeros(sr * 2), sr)).character == "quiet/ambient"


def test_corrupt_audio_raises_media_facts_error() -> None:
    with pytest.raises(rmf.MediaFactsError):
        rmf.analyze_audio(b"garbage" * 100)


def test_fmt_ts() -> None:
    assert rmf.fmt_ts(0) == "00:00.000"
    assert rmf.fmt_ts(3.5834) == "00:03.583"
    assert rmf.fmt_ts(75.0) == "01:15.000"
    assert rmf.fmt_ts(-1) == "00:00.000"


def test_render_video_facts() -> None:
    f = rmf.VideoFacts(5.0, 24.0, 1344, 768, (3.583,), ())
    assert rmf.render_video_facts("<Video 1>", f) == (
        "<Video 1>: 5.00 s, 24.00 fps, 1344x768; shots: "
        "[Shot 1] 00:00.000–00:03.583, [Shot 2] 00:03.583–00:05.000"
    )
    one = rmf.VideoFacts(5.0, 24.0, 1344, 768, (), ())
    assert rmf.render_video_facts("<Video 1>", one).endswith("shots: [Shot 1] 00:00.000–00:05.000")
    none = rmf.VideoFacts(5.0, 24.0, 1344, 768, None, ())
    assert rmf.render_video_facts("<Video 2>", none).endswith("shot boundaries unavailable")


def test_render_audio_facts() -> None:
    f = rmf.AudioFacts(2.0, 16000, 1, (-20.0, -20.0, -60.0, -60.0), (1.0,), 0.5, "tonal/music")
    s = rmf.render_audio_facts("<Audio 1>", f)
    assert s.startswith("<Audio 1>: 2.00 s, 16000 Hz, 1 ch")
    assert "00:00.000–00:01.000 -20 dB" in s and "00:01.000–00:02.000 -60 dB" in s
    assert "changes at 00:01.000" in s and "onsets 0.5/s" in s and "tonal/music" in s


REAL = "/Users/feng/codes/s_codes/drama-ai-coding/htmls/h3-ref2va-official-prompt-20261001/inputs/refvid1_5s.mp4"


@pytest.mark.skipif(not os.path.exists(REAL), reason="real reference video absent")
def test_real_reference_video() -> None:
    f = rmf.analyze_video(open(REAL, "rb").read())
    assert f.cuts is not None and len(f.cuts) == 1
    assert abs(f.cuts[0] - 3.583) <= 0.1
    assert 0 < len(f.keyframes) <= 8
