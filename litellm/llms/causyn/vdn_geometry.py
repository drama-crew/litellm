"""Admitted native canvases and output rectangles, independent of GPU state."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, Self, TypedDict


class VideoFrames(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: tuple[slice, ...]) -> Self: ...


class GeometryRecord(TypedDict):
    aspect_ratio: str
    canvas: list[int]
    output: list[int]
    crop: list[int]


@dataclass(frozen=True)
class Geometry:
    ratio: str
    width: int
    height: int
    output_width: int
    output_height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def crop(self) -> tuple[int, int, int, int]:
        left = (self.width - self.output_width) // 2
        top = (self.height - self.output_height) // 2
        return left, top, left + self.output_width, top + self.output_height

    def crop_video(self, frames: VideoFrames) -> VideoFrames:
        if frames.ndim != 5 or tuple(frames.shape[2:4]) != self.shape:
            raise ValueError("decoded video differs from the requested native canvas")
        left, top, right, bottom = self.crop
        return frames[:, :, top:bottom, left:right, :]

    def record(self) -> GeometryRecord:
        return {
            "aspect_ratio": self.ratio,
            "canvas": [self.width, self.height],
            "output": [self.output_width, self.output_height],
            "crop": list(self.crop),
        }


GEOMETRIES = {
    g.ratio: g
    for g in (
        Geometry("16:9", 1344, 768, 1344, 756),
        Geometry("4:3", 1024, 768, 1024, 768),
        Geometry("1:1", 768, 768, 768, 768),
        Geometry("3:4", 768, 1024, 768, 1024),
        Geometry("9:16", 768, 1344, 756, 1344),
    )
}
LEGACY = Geometry("7:4", 1344, 768, 1344, 768)


CANVAS_MULTIPLE = 32
MAX_PIXELS = 1344 * 768
MAX_SPATIAL_TEMPORAL_CELLS = 60_000
MAX_EDGE = 1536
MIN_INPUT_EDGE = 256
MAX_INPUT_EDGE = 5760
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5


def pixel_budget(frames: int) -> int:
    if type(frames) is not int or not 5 <= frames <= 362 or frames % 17 != 5:
        raise ValueError("frame count must use the admitted temporal grid")
    latent_frames = (frames - 5) // 17 * 5 + 2
    return min(MAX_PIXELS, MAX_SPATIAL_TEMPORAL_CELLS // latent_frames * 1024)


def validate_canvas(width: int, height: int) -> None:
    if any(type(v) is not int or v < 32 or v % 32 or v > MAX_EDGE for v in (width, height)):
        raise ValueError("native canvas dimensions must be aligned and bounded")
    if width * height > MAX_PIXELS:
        raise ValueError("native canvas exceeds the spatial pixel budget")


def validate_image_size(size: tuple[int, int]) -> None:
    width, height = size
    if any(type(v) is not int or not MIN_INPUT_EDGE <= v <= MAX_INPUT_EDGE for v in size):
        raise ValueError("keyframe width and height must each be between 256 and 5760 pixels")
    if not MIN_ASPECT <= width / height <= MAX_ASPECT:
        raise ValueError("keyframe aspect ratio must be between 2:5 and 5:2")


def _output_rectangle(width: int, height: int, aspect: Fraction) -> tuple[int, int]:
    numerator, denominator = aspect.numerator, aspect.denominator
    multiple = 2 if numerator % 2 or denominator % 2 else 1
    scale = min(width // numerator, height // denominator) // multiple * multiple
    if scale > 0 and scale * scale * numerator * denominator >= width * height * 0.95:
        return scale * numerator, scale * denominator
    ratio = float(aspect)
    if width / height > ratio:
        return max(2, round(height * ratio / 2) * 2), height
    return width, max(2, round(width / ratio / 2) * 2)


def _fit_geometry(aspect: Fraction, ratio: str, frames: int) -> Geometry:
    target = float(aspect)
    target_width, target_height = (768 * target, 768.0) if target >= 1 else (768.0, 768 / target)
    budget = pixel_budget(frames)
    scale = min(1.0, math.sqrt(budget / (target_width * target_height)), MAX_EDGE / max(target_width, target_height))
    width, height = [max(32, round(v * scale / 32) * 32) for v in (target_width, target_height)]
    while width * height > budget:
        choices = [(width - 32, height), (width, height - 32)]
        width, height = min(choices, key=lambda pair: abs(math.log((pair[0] / pair[1]) / target)))
    validate_canvas(width, height)
    output_width, output_height = _output_rectangle(width, height, aspect)
    return Geometry(ratio, width, height, output_width, output_height)


def resolve_geometry(
    width: int | None = None,
    height: int | None = None,
    ratio: str | None = None,
    short_edge: int | None = None,
    *,
    first_size: tuple[int, int] | None = None,
    frames: int = 107,
) -> Geometry:
    for value in (width, height):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("width and height must be positive integers")
    if short_edge is not None and (type(short_edge) is not int or short_edge != 768):
        raise ValueError("VDN supports the 768p canvas profile only")
    if ratio is not None and ratio not in (*GEOMETRIES, LEGACY.ratio, "adaptive"):
        raise ValueError("unsupported VDN aspect ratio")
    if first_size is not None:
        validate_image_size(first_size)
        return _fit_geometry(Fraction(*first_size), "adaptive", frames)
    if ratio == "adaptive":
        raise ValueError("text-to-video requires a concrete aspect ratio")
    if ratio is not None:
        result = LEGACY if ratio == LEGACY.ratio else GEOMETRIES[ratio]
        supplied = (width, height)
        valid = ((result.width, result.height), (result.output_width, result.output_height))
        if not any(all(a is None or a == b for a, b in zip(supplied, pair)) for pair in valid):
            raise ValueError("width/height and aspect_ratio disagree")
    else:
        supplied = (1344 if width is None else width, 768 if height is None else height)
        matched = next(
            (
                g
                for g in [LEGACY, *GEOMETRIES.values()]
                if supplied in ((g.width, g.height), (g.output_width, g.output_height))
            ),
            None,
        )
        if matched is None:
            raise ValueError("VDN output dimensions are outside the admitted aspect ratio profiles")
        result = matched
    if result.width * result.height > pixel_budget(frames):
        return _fit_geometry(Fraction(result.ratio.replace(":", "/")), result.ratio, frames)
    return result
