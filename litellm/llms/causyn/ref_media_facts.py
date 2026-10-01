"""Perception of reference media for the H3 ref2va prompt rewrite.

Pure CPU, PyAV + numpy + Pillow only (no ffmpeg binary). Failure policy:
``analyze_video`` never raises (degrades to ``cuts=None`` / no keyframes);
``analyze_audio`` raises ``MediaFactsError`` when the audio cannot be decoded.
"""

from __future__ import annotations

import bisect
import io
import math
import time
from dataclasses import dataclass
from statistics import median

MAX_SAMPLE_FPS = 48.0
PROXY_SIZE = (64, 36)
CUT_MIN_SCORE = 0.12
CUT_MEDIAN_FACTOR = 4.0
CUT_MIN_GAP = 0.5
CUT_EDGE_GUARD = 0.25
KEYFRAME_SHORT_SIDE = 448
KEYFRAME_QUALITY = 80
KEYFRAME_START_OFFSET = 0.15
LONG_SHOT_SECONDS = 4.0
TOTAL_KEYFRAMES = 16
RMS_STEP = 0.5
CHANGE_DB = 6.0
SILENCE_DB = -90.0


class MediaFactsError(Exception):
    pass


@dataclass(frozen=True)
class Keyframe:
    t: float
    shot: int
    jpeg: bytes


@dataclass(frozen=True)
class VideoFacts:
    duration: float
    fps: float
    width: int
    height: int
    cuts: tuple[float, ...] | None
    keyframes: tuple[Keyframe, ...]


@dataclass(frozen=True)
class AudioFacts:
    duration: float
    sample_rate: int
    channels: int
    loudness_db: tuple[float, ...]
    change_points: tuple[float, ...]
    onset_rate: float
    character: str


def keyframe_budget(n_videos: int) -> int:
    return min(8, TOTAL_KEYFRAMES // max(1, n_videos))


def fmt_ts(t: float) -> str:
    ms = max(0, round(t * 1000))
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}"


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() > deadline


def _frame_time(frame, index: int, fps: float) -> float:
    t = getattr(frame, "time", None)
    return float(t) if t is not None else index / fps


# ---------------------------------------------------------------- video


def _open_video(raw: bytes):
    import av

    return av.open(io.BytesIO(raw), mode="r", format="mov", options={"enable_drefs": "0"})


def _detect_cuts(raw: bytes, fps: float, duration: float, deadline: float | None) -> tuple[float, ...] | None:
    import numpy as np

    times: list[float] = []
    scores: list[float] = []
    prev = None
    last_t = -1e9
    min_dt = 1.0 / MAX_SAMPLE_FPS - 1e-4
    with _open_video(raw) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(stream)):
            if _expired(deadline):
                return None
            t = _frame_time(frame, i, fps)
            if t - last_t < min_dt:
                continue
            last_t = t
            proxy = frame.reformat(width=PROXY_SIZE[0], height=PROXY_SIZE[1], format="gray").to_ndarray()
            proxy = proxy.astype(np.float32) / 255.0
            if prev is not None:
                times.append(t)
                scores.append(float(np.abs(proxy - prev).mean()))
            prev = proxy
    if not scores:
        return ()
    threshold = max(CUT_MIN_SCORE, CUT_MEDIAN_FACTOR * float(median(scores)))
    cuts: list[float] = []
    for t, s in zip(times, scores):
        if s <= threshold or t < CUT_EDGE_GUARD or (duration and t > duration - CUT_EDGE_GUARD):
            continue
        if cuts and t - cuts[-1] < CUT_MIN_GAP:
            continue
        cuts.append(round(t, 4))
    return tuple(cuts)


def _pick_keyframe_times(cuts: tuple[float, ...], duration: float, cap: int) -> list[float]:
    bounds = [0.0, *cuts, duration]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    if len(shots) > cap:  # more shots than budget: spread evenly across shots
        idx = sorted({round(i * (len(shots) - 1) / (cap - 1)) for i in range(cap)}) if cap > 1 else [0]
        shots = [shots[i] for i in idx]
    tiers: list[list[float]] = [[], [], []]
    for s, e in shots:
        tiers[0].append(min(s + KEYFRAME_START_OFFSET, max(s, (s + e) / 2)))
        tiers[1].append((s + e) / 2)
        if e - s > LONG_SHOT_SECONDS:
            n = int((e - s) // LONG_SHOT_SECONDS)
            tiers[2].extend(s + (e - s) * k / (n + 1) for k in range(1, n + 1))
    chosen: list[float] = []
    for t in (t for tier in tiers for t in tier):
        if len(chosen) >= cap:
            break
        if all(abs(t - u) >= 0.1 for u in chosen):
            chosen.append(t)
    return sorted(chosen)


def _extract_keyframes(
    raw: bytes, targets: list[float], cuts: tuple[float, ...], fps: float, deadline: float | None
) -> tuple[Keyframe, ...]:
    from PIL import Image

    out: list[Keyframe] = []
    pending = list(targets)
    last = None
    with _open_video(raw) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(stream)):
            if _expired(deadline):
                break
            t = _frame_time(frame, i, fps)
            last = (t, frame)
            while pending and t >= pending[0] - 1e-6:
                pending.pop(0)
                out.append(_encode_keyframe(t, frame, cuts, Image))
            if not pending:
                break
        if pending and last is not None and not _expired(deadline):
            out.append(_encode_keyframe(last[0], last[1], cuts, Image))
    uniq: dict[float, Keyframe] = {k.t: k for k in out}
    return tuple(uniq[t] for t in sorted(uniq))


def _encode_keyframe(t: float, frame, cuts: tuple[float, ...], image_mod) -> Keyframe:
    img = frame.to_image()
    short = min(img.size)
    if short > KEYFRAME_SHORT_SIDE:
        scale = KEYFRAME_SHORT_SIDE / short
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), image_mod.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=KEYFRAME_QUALITY)
    return Keyframe(t=round(t, 4), shot=bisect.bisect_right(cuts, t - 1e-6) + 1, jpeg=buf.getvalue())


def analyze_video(raw: bytes, *, max_keyframes: int = 8, deadline: float | None = None) -> VideoFacts:
    """Never raises. cuts=None means detection failed or the deadline was exceeded."""
    duration, fps, width, height = 0.0, 0.0, 0, 0
    try:
        import av

        with _open_video(raw) as c:
            stream = c.streams.video[0]
            width, height = int(stream.codec_context.width), int(stream.codec_context.height)
            fps = float(stream.average_rate or 0)
            if c.duration:
                duration = float(c.duration) / av.time_base
            elif stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
    except Exception:
        return VideoFacts(duration, fps, width, height, None, ())
    if not math.isfinite(duration):
        duration = 0.0
    try:
        cuts = _detect_cuts(raw, fps or 24.0, duration, deadline)
    except Exception:
        cuts = None
    if cuts is None and _expired(deadline):
        return VideoFacts(duration, fps, width, height, None, ())
    try:
        if max_keyframes <= 0 or duration <= 0:
            return VideoFacts(duration, fps, width, height, cuts, ())
        shots = cuts or ()
        targets = _pick_keyframe_times(shots, duration, max_keyframes)
        frames = _extract_keyframes(raw, targets, shots, fps or 24.0, deadline)
    except Exception:
        frames = ()
    return VideoFacts(duration, fps, width, height, cuts, frames)


def render_video_facts(label: str, f: VideoFacts) -> str:
    head = f"{label}: {f.duration:.2f} s, {f.fps:.2f} fps, {f.width}x{f.height}; "
    if f.cuts is None:
        return head + "shot boundaries unavailable"
    bounds = [0.0, *f.cuts, f.duration]
    shots = ", ".join(
        f"[Shot {i + 1}] {fmt_ts(bounds[i])}–{fmt_ts(bounds[i + 1])}" for i in range(len(bounds) - 1)
    )
    return head + "shots: " + shots


# ---------------------------------------------------------------- audio


def _decode_mono(raw: bytes):
    import av
    import numpy as np

    try:
        with av.open(io.BytesIO(raw), mode="r", options={"enable_drefs": "0"}) as c:
            if not c.streams.audio:
                raise MediaFactsError("no audio stream")
            stream = c.streams.audio[0]
            sr = int(stream.codec_context.sample_rate)
            channels = int(stream.codec_context.channels)
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
            chunks = []
            for frame in c.decode(stream):
                for r in resampler.resample(frame):
                    chunks.append(r.to_ndarray().reshape(-1))
            for r in resampler.resample(None) or []:
                chunks.append(r.to_ndarray().reshape(-1))
    except MediaFactsError:
        raise
    except Exception as exc:
        raise MediaFactsError(f"cannot decode audio: {exc}") from exc
    if not chunks or sr <= 0:
        raise MediaFactsError("audio contains no samples")
    return np.concatenate(chunks).astype(np.float32), sr, channels


def _spectral_stats(x, sr: int) -> tuple[float, float]:
    """(onsets per second, mean spectral flatness of active frames).

    Frames of 25 ms with 10 ms hop, Hann window. Onsets = peaks of the positive
    spectral flux of log-compressed magnitudes above median+3*std (and an
    absolute floor), at least 60 ms apart. Flatness = geometric/arithmetic mean
    of the power spectrum, averaged over frames within 40 dB of the loudest.
    """
    import numpy as np

    n, hop = int(sr * 0.025), int(sr * 0.010)
    if len(x) < n + hop:
        return 0.0, 0.0
    win = np.hanning(n).astype(np.float32)
    idx = np.arange(n)[None, :] + hop * np.arange((len(x) - n) // hop + 1)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    logmag = np.log1p(100.0 * mag)
    flux = np.maximum(logmag[1:] - logmag[:-1], 0).sum(axis=1) / logmag.shape[1]
    thr = max(float(np.median(flux) + 3 * flux.std()), 0.15)
    gap = max(1, int(0.06 / 0.010))
    onsets, last = 0, -gap
    for i in range(1, len(flux) - 1):
        if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and i - last >= gap:
            onsets += 1
            last = i
    power = mag**2 + 1e-12
    energy = power.sum(axis=1)
    active = energy > energy.max() * 1e-4
    if not active.any():
        return onsets / (len(x) / sr), 0.0
    p = power[active]
    flat = np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)
    return onsets / (len(x) / sr), float(flat.mean())


def _character(mean_db: float, db_std: float, onset_rate: float, flatness: float) -> str:
    """Documented heuristics (dBFS RMS timeline, onsets/s, spectral flatness 0..1):
    quiet/ambient: mean loudness < -50 dB, or noise-like (flatness >= .25) with few onsets;
    percussive/rhythmic: onset rate >= 3/s with noise-like spectra (flatness >= .25), or >= 8/s;
    speech-like: 1.5-8 onsets/s, loudness varying (std >= 5 dB), tonal-ish spectra (flatness < .25);
    tonal/music: everything else (steady, harmonic content).
    """
    if mean_db < -50:
        return "quiet/ambient"
    if onset_rate >= 8 or (onset_rate >= 3 and flatness >= 0.25):
        return "percussive/rhythmic"
    if flatness >= 0.25:
        return "quiet/ambient"
    if 1.5 <= onset_rate <= 8 and db_std >= 5:
        return "speech-like"
    return "tonal/music"


def analyze_audio(raw: bytes) -> AudioFacts:
    """Raises MediaFactsError when the audio cannot be decoded."""
    import numpy as np

    x, sr, channels = _decode_mono(raw)
    step = int(sr * RMS_STEP)
    nbuckets = max(1, math.ceil(len(x) / step))
    db: list[float] = []
    for b in range(nbuckets):
        seg = x[b * step : (b + 1) * step]
        rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
        db.append(round(max(SILENCE_DB, 20 * math.log10(rms)) if rms > 0 else SILENCE_DB, 1))
    changes = tuple(round(i * RMS_STEP, 3) for i in range(1, len(db)) if abs(db[i] - db[i - 1]) > CHANGE_DB)
    onset_rate, flatness = _spectral_stats(x, sr)
    mean_db = float(np.mean(db))
    std_db = float(np.std(db))
    return AudioFacts(
        duration=len(x) / sr,
        sample_rate=sr,
        channels=channels,
        loudness_db=tuple(db),
        change_points=changes,
        onset_rate=round(onset_rate, 2),
        character=_character(mean_db, std_db, onset_rate, flatness),
    )


def render_audio_facts(label: str, f: AudioFacts) -> str:
    bounds = [0.0, *f.change_points, len(f.loudness_db) * RMS_STEP]
    segs = []
    for i in range(len(bounds) - 1):
        lo, hi = int(round(bounds[i] / RMS_STEP)), int(round(bounds[i + 1] / RMS_STEP))
        vals = f.loudness_db[lo:hi] or f.loudness_db[-1:]
        end = min(bounds[i + 1], f.duration) if i == len(bounds) - 2 else bounds[i + 1]
        segs.append(f"{fmt_ts(bounds[i])}–{fmt_ts(end)} {sum(vals) / len(vals):.0f} dB")
    changes = ", ".join(fmt_ts(t) for t in f.change_points) or "none"
    return (
        f"{label}: {f.duration:.2f} s, {f.sample_rate} Hz, {f.channels} ch; loudness: {'; '.join(segs)}; "
        f"changes at {changes}; onsets {f.onset_rate:g}/s; character: {f.character}"
    )
