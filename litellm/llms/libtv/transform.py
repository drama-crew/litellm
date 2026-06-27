from math import gcd
from typing import Any, Dict, List, Optional

_RATIO_BY_WH = {
    (1, 1): "1:1",
    (16, 9): "16:9",
    (9, 16): "9:16",
    (4, 3): "4:3",
    (3, 4): "3:4",
    (21, 9): "21:9",
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
    key = (w // g, h // g)
    return _RATIO_BY_WH.get(key, f"{key[0]}:{key[1]}")


def _resolution_from_size(size: Optional[str]) -> Optional[str]:
    wh = _parse_wh(size)
    if wh is None:
        return None
    shortest = min(wh)
    if shortest >= 1080:
        return "1080p"
    if shortest >= 720:
        return "720p"
    return "480p"


def _enum_values(prop: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for e in prop.get("enum") or []:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, dict) and isinstance(e.get("value"), str):
            out.append(e["value"])
    return out


def _coerce_to_enum(value: Any, prop: Dict[str, Any]) -> Any:
    options = _enum_values(prop)
    if not options or value is None:
        return value
    if value in options:
        return value
    if isinstance(value, str):
        for opt in options:
            if opt.lower() == value.lower():
                return opt
    return None


def _allowed_setting_keys(spec: Dict[str, Any], mode: str) -> List[str]:
    cfg_settings = (spec.get("config") or {}).get("settings")
    if isinstance(cfg_settings, dict):
        return list(cfg_settings.get(mode) or [])
    if isinstance(cfg_settings, list):
        return list(cfg_settings)
    return []


def build_generation_params(
    prompt: str,
    optional_params: Dict[str, Any],
    spec: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    op = dict(optional_params or {})
    props = spec.get("properties") or {}
    size = op.get("size")

    ratio = op.get("ratio") or op.get("aspect_ratio") or size_to_ratio(size)
    candidates: Dict[str, Any] = {
        "ratio": ratio,
        "ratio_auto": ratio,
        "resolution": op.get("resolution") or _resolution_from_size(size),
        "resolution_480": op.get("resolution") or _resolution_from_size(size),
        "duration": op.get("seconds") or op.get("duration"),
        "quality": op.get("quality"),
        "quality_high": op.get("quality"),
        "enableSound": op.get("enableSound"),
    }

    settings: Dict[str, Any] = {}
    for key in _allowed_setting_keys(spec, mode):
        prop = props.get(key) or {}
        val = candidates.get(key)
        if key.startswith("duration") and val is not None:
            try:
                val = int(val)
            except (TypeError, ValueError):
                val = None
        if val is not None and prop.get("enum"):
            val = _coerce_to_enum(val, prop)
        if val is None:
            val = prop.get("default")
        if val is not None:
            settings[key] = val

    count_prop = props.get("count")
    count_default = count_prop.get("default") if isinstance(count_prop, dict) else 1
    count = op.get("n") or op.get("count") or count_default or 1

    params: Dict[str, Any] = {
        "prompt": prompt,
        "modeType": mode,
        "count": int(count),
        "textList": [],
        "imageList": [],
        "videoList": [],
        "audioList": [],
    }
    params.update(settings)
    advanced = op.get("advancedSettings")
    if isinstance(advanced, dict):
        params.update(advanced)
    return params
