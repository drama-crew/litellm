from math import gcd
from typing import Any, Dict, List, Optional

DEFAULT_DURATION_SEC = 5
VISION_1080P_MODEL = "seedance2.0_vision"

_RATIO_BY_WH = {
    (1, 1): "1:1",
    (16, 9): "16:9",
    (9, 16): "9:16",
    (4, 3): "4:3",
    (3, 4): "3:4",
}


def _parse_wh(size: Optional[str]):
    if not size or "x" not in size.lower():
        return None
    try:
        w, h = (int(x) for x in size.lower().split("x", 1))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h


def size_to_ratio(size: Optional[str]) -> Optional[str]:
    wh = _parse_wh(size)
    if wh is None:
        return None
    w, h = wh
    g = gcd(w, h)
    return _RATIO_BY_WH.get((w // g, h // g))


def resolution_from_size(size: Optional[str]) -> Optional[str]:
    wh = _parse_wh(size)
    if wh is None:
        return None
    shortest = min(wh)
    if shortest >= 1080:
        return "1080p"
    if shortest >= 720:
        return "720p"
    return "480p"


def resolve_resolution(model: str, requested: Optional[str], warn) -> Optional[str]:
    """1080p is only accepted by seedance2.0_vision; upstream rejects any other model
    with ret=2 (a BadRequestError the router treats as an abort), so downgrade here
    instead of sending a request we already know will fail."""
    if requested == "1080p" and model != VISION_1080P_MODEL:
        warn(
            "xiaoyunque video_generation model=%s: 1080p is only supported by %s, downgrading to 720p",
            model,
            VISION_1080P_MODEL,
        )
        return "720p"
    return requested


def _asset_refs(asset_ids: List[str]) -> List[Dict[str, str]]:
    return [{"pippit_asset_id": asset_id} for asset_id in asset_ids]


def build_video_part_tool_param(
    prompt: str,
    model: str,
    optional_params: Dict[str, Any],
    image_asset_ids: List[str],
    video_asset_ids: List[str],
    audio_asset_ids: List[str],
    generate_type: Optional[int],
    warn,
) -> Dict[str, Any]:
    op = optional_params or {}
    ratio = op.get("ratio") or op.get("aspect_ratio") or size_to_ratio(op.get("size"))
    resolution = op.get("resolution") or resolution_from_size(op.get("size"))
    resolution = resolve_resolution(model, resolution, warn)
    try:
        duration_sec = int(op.get("seconds") or op.get("duration") or DEFAULT_DURATION_SEC)
    except (TypeError, ValueError):
        duration_sec = DEFAULT_DURATION_SEC

    param: Dict[str, Any] = {"prompt": prompt, "model": model, "duration_sec": duration_sec}
    if ratio:
        param["ratio"] = ratio
    if resolution:
        param["resolution"] = resolution
    if generate_type is not None:
        param["generate_type"] = generate_type
    if image_asset_ids:
        param["images"] = _asset_refs(image_asset_ids)
    if video_asset_ids:
        param["videos"] = _asset_refs(video_asset_ids)
    if audio_asset_ids:
        param["audios"] = _asset_refs(audio_asset_ids)
    return param
