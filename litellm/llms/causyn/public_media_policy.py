"""Submit-time media admission for PUBLIC causyn-1.1 requests.

One validator shared by the MiniMax-H3 facade (direct and Context IR namespaces) and the public
``/v1/videos`` causyn-1.1 path. It applies the official MiniMax v2 media limits to everything whose
bytes are available at submit time (Base64 data URLs) and an SSRF pre-check to everything that is
only a pointer (http/https URLs). It is never applied to internal producer/canvas traffic, which
enters through a moderation admission ticket and never reaches this code.

Choices worth knowing:

* The bytes decide the format, not the declared ``data:`` media type. The declared type only has to
  belong to the right family (``image/*`` for an image slot), so a declared ``image/png`` that really
  holds JPEG is accepted as JPEG. Bytes that are not an allowed format are rejected whatever they claim.
* Dimensions are read from the header (Pillow's lazy ``Image.open``) and checked *before* any pixel
  is decoded, so a decompression bomb is rejected without being inflated. Integrity (truncated or
  corrupt data) is then verified with a full decode of an image that is already known to be at most
  5760x5760, in a worker thread with at most two decodes running at once.
* http(s) URLs: bytes are not available here, so only the SSRF pre-check runs. The app enforces format
  and size when it fetches. IP literals in any form are refused outright (a legitimate public media
  URL uses a hostname), as are single-label and internal suffixes, hosts that resolve (any A/AAAA
  answer) to a non-global address, and ports other than 80/443/>=1024.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import ipaddress
import re
import socket
import functools
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from litellm.llms.causyn.h3_media import CAUSYN_VIDEO_LIMITS, inspect_audio, inspect_video
from litellm.llms.causyn.h3_prompt import CAUSYN_VIDEO_REF_TOTAL_MAX_SECONDS, RewriteError

MB = 1024 * 1024
IMAGE_MAX_BYTES = 30 * MB
VIDEO_MAX_BYTES = 50 * MB
AUDIO_MAX_BYTES = 15 * MB
SIDE_MIN, SIDE_MAX = 256, 5760
ASPECT_MIN, ASPECT_MAX = 0.4, 2.5
AUDIO_MIN_SECONDS = 2.0
AUDIO_TOTAL_MAX_SECONDS = 15.0 + 1e-6
MAX_REFERENCES = 16
DNS_TIMEOUT_S = 3.0
DECODE_TIMEOUT_S = 10.0
DECODE_SLOTS, DECODE_MAX_WAITING = 2, 16
DNS_SLOTS, DNS_MAX_WAITING = 8, 16
IMAGE_FORMATS = "JPEG, PNG, WEBP, HEIC, HEIF"
PUBLIC_HOST_MESSAGE = "media URL must point to a public internet host"

Kind = Literal["image", "video", "audio"]



class MediaBusyError(Exception):
    """Validation capacity is saturated; the caller maps this to 503 (retryable), never queues unboundedly."""


# Dedicated pools: media work must never occupy the default executor that the rest of the proxy
# (including upstream DNS) depends on.
_DECODE_POOL = ThreadPoolExecutor(max_workers=DECODE_SLOTS, thread_name_prefix="media-decode")
_DNS_POOL = ThreadPoolExecutor(max_workers=DNS_SLOTS, thread_name_prefix="media-dns")


class _Gate:
    """Bounded admission in front of a pool. The slot is released when the THREAD finishes (not when the
    caller times out), so slots always reflect real thread occupancy."""

    def __init__(self, pool: ThreadPoolExecutor, slots: int, max_waiting: int):
        self.pool, self.max_waiting = pool, max_waiting
        self.semaphore = asyncio.Semaphore(slots)
        self.waiting = 0

    async def run(self, fn, timeout: float):
        if self.semaphore.locked() and self.waiting >= self.max_waiting:
            raise MediaBusyError("media validation is busy, retry later")
        self.waiting += 1
        try:
            await self.semaphore.acquire()
        finally:
            self.waiting -= 1
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.pool, fn)

        def finished(done: asyncio.Future) -> None:
            self.semaphore.release()
            if not done.cancelled():
                done.exception()  # retrieve, so an abandoned (timed-out) job never logs "never retrieved"

        future.add_done_callback(finished)
        return await asyncio.wait_for(asyncio.shield(future), timeout)


_gates: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _Gate]]" = weakref.WeakKeyDictionary()


def _gate(name: str) -> _Gate:
    loop_gates = _gates.setdefault(asyncio.get_running_loop(), {})
    if name not in loop_gates:
        loop_gates[name] = (
            _Gate(_DECODE_POOL, DECODE_SLOTS, DECODE_MAX_WAITING)
            if name == "decode"
            else _Gate(_DNS_POOL, DNS_SLOTS, DNS_MAX_WAITING)
        )
    return loop_gates[name]


_HEIF_BRANDS = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"}
)
_AVIF_BRANDS = frozenset({b"avif", b"avis"})
_BLOCKED_SUFFIXES = (
    ".localhost",
    ".internal",
    ".local",
    ".svc",
    ".cluster.local",
    ".localdomain",
    ".home.arpa",
)
# Addresses Python's ``is_global`` still calls routable but that are cloud fabric, tunnels or
# embeddings of another address space (NAT64 / 6to4 / Teredo can wrap a private IPv4).
_EXTRA_BLOCKED = tuple(
    ipaddress.ip_network(net)
    for net in (
        "100.64.0.0/10",  # carrier-grade NAT, includes the Aliyun metadata address 100.100.100.200
        "168.63.129.16/32",  # Azure wire server
        "::/96",  # IPv4-compatible (::127.0.0.1)
        "fec0::/10",  # deprecated site-local, which Python still treats as global
        "64:ff9b::/96",
        "64:ff9b:1::/48",
        "2002::/16",
        "2001::/32",
    )
)


class MediaPolicyError(ValueError):
    """A precise, non-leaking rejection; the caller maps it to HTTP 400."""


@dataclass(frozen=True)
class MediaRef:
    kind: Kind
    url: str
    role: str
    label: str


# ------------------------------------------------------------------ extraction


def _kind(value: object) -> Kind:
    return value if value in ("image", "video", "audio") else "image"  # type: ignore[return-value]


def extract_media_refs(payload: Mapping[str, object]) -> list[MediaRef]:
    """Every media pointer in a public create payload, in request order, with a stable label."""
    found: list[tuple[Kind, str, str]] = []
    for name, role in (("image", "first_frame"), ("last_image", "last_frame")):
        value = payload.get(name)
        if isinstance(value, str):
            found.append(("image", value, role))
    references = payload.get("references")
    if isinstance(references, list):
        for item in references:
            if isinstance(item, Mapping) and isinstance(item.get("url"), str):
                found.append((_kind(item.get("media_type")), str(item["url"]), str(item.get("role") or "reference")))
    content = payload.get("content")
    if isinstance(content, (list, tuple)):
        for item in content:
            if not isinstance(item, Mapping):
                continue
            for key, kind in (("image_url", "image"), ("video_url", "video"), ("audio_url", "audio")):
                holder = item.get(key)
                if item.get("type") == key and isinstance(holder, Mapping) and isinstance(holder.get("url"), str):
                    found.append((kind, str(holder["url"]), str(item.get("role") or "reference")))  # type: ignore[arg-type]
    counts: dict[str, int] = {}
    refs: list[MediaRef] = []
    for kind, url, role in found:
        if kind == "image" and role in ("first_frame", "last_frame"):
            label = role.replace("_", " ") + " image"
        else:
            counts[kind] = counts.get(kind, 0) + 1
            label = f"reference {kind} {counts[kind]}"
        refs.append(MediaRef(kind=kind, url=url, role=role, label=label))
    return refs


# ------------------------------------------------------------------ format sniffing


def _ftyp(raw: bytes) -> tuple[bytes, set[bytes]] | None:
    if len(raw) < 12 or raw[4:8] != b"ftyp":
        return None
    size = int.from_bytes(raw[:4], "big")
    end = min(size if size >= 16 else 16, len(raw), 512)
    brands = {raw[i : i + 4] for i in range(16, end - 3, 4)}
    return raw[8:12], brands


def sniff_image(raw: bytes) -> str | None:
    """Canonical name of the detected format, allowed or not. ``None`` means unrecognised."""
    head = raw[:16]
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "WEBP"
    box = _ftyp(raw)
    if box is not None:
        major, brands = box
        if major in _AVIF_BRANDS or b"avif" in brands:
            return "AVIF"
        if major in _HEIF_BRANDS:
            return "HEIF"
        return "MP4"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "GIF"
    if head.startswith(b"BM"):
        return "BMP"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "TIFF"
    if head.startswith(b"\x00\x00\x01\x00"):
        return "ICO"
    if head.startswith(b"%PDF"):
        return "PDF"
    if head.startswith(b"\x7fELF"):
        return "ELF"
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "ZIP"
    text = raw[:512].lstrip().lower()
    if text.startswith((b"<!doctype html", b"<html", b"<head", b"<body", b"<script")):
        return "HTML"
    if text.startswith((b"<svg", b"<?xml")) or b"<svg" in text:
        return "SVG"
    return None


def sniff_video(raw: bytes) -> bool:
    box = _ftyp(raw)
    return box is not None and box[0] not in _AVIF_BRANDS and box[0] not in _HEIF_BRANDS


def sniff_audio(raw: bytes) -> str | None:
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "WAV"
    if raw[:3] == b"ID3":
        return "MP3"
    if len(raw) >= 2 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0 and (raw[1] >> 1) & 0x03 == 0x01:
        return "MP3"
    if len(raw) >= 2 and raw[0] == 0xFF and raw[1] & 0xF6 == 0xF0:
        return "AAC"
    if raw[:4] == b"OggS":
        return "OGG"
    if raw[:4] == b"fLaC":
        return "FLAC"
    if _ftyp(raw) is not None:
        return "M4A/MP4"
    return None


# ------------------------------------------------------------------ data URLs


def _split_data_url(url: str) -> tuple[str, str]:
    header, _, body = url.partition(",")
    parts = header[5:].split(";")
    if "base64" not in (part.lower() for part in parts[1:]) or not body:
        raise MediaPolicyError("media must be a public HTTP(S) URL or Base64 data URL")
    return parts[0].strip().lower(), body


def _decode(body: str, limit: int, noun: str) -> bytes:
    # Bound memory before decoding: base64 inflates by 4/3.
    if len(body) > limit * 4 // 3 + 8:
        raise MediaPolicyError(f"{noun} exceeds {limit // MB} MB")
    try:
        raw = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError):
        raise MediaPolicyError("invalid Base64 media") from None
    if len(raw) > limit:
        raise MediaPolicyError(f"{noun} exceeds {limit // MB} MB")
    return raw


def _image_bytes(raw: bytes) -> None:
    from PIL import Image

    detected = sniff_image(raw)
    if detected not in ("JPEG", "PNG", "WEBP", "HEIF"):
        named = f" ({detected})" if detected else ""
        raise MediaPolicyError(f"unsupported format{named}; allowed: {IMAGE_FORMATS}")
    if detected == "HEIF":
        try:
            from pillow_heif import register_heif_opener  # pyright: ignore[reportMissingTypeStubs]

            register_heif_opener()
        except ImportError:
            raise MediaPolicyError("HEIC/HEIF images cannot be inspected by this service") from None
    try:
        with Image.open(io.BytesIO(raw)) as source:
            # Pillow reports a multi-picture JPEG (common phone photo) as "MPO"; its first frame is what is used.
            if source.format not in ("JPEG", "MPO", "PNG", "WEBP", "HEIF"):
                raise MediaPolicyError("image is corrupt or truncated")
            width, height = source.size
            if not (SIDE_MIN <= width <= SIDE_MAX and SIDE_MIN <= height <= SIDE_MAX):
                raise MediaPolicyError(f"dimensions {width}x{height} are outside {SIDE_MIN}-{SIDE_MAX} pixels per side")
            if not ASPECT_MIN <= width / height <= ASPECT_MAX:
                raise MediaPolicyError(
                    f"aspect ratio {width / height:.2f} is outside {ASPECT_MIN}-{ASPECT_MAX} (width/height)"
                )
            source.load()  # integrity: the header already proved this is at most 5760x5760
    except MediaPolicyError:
        raise
    except Image.DecompressionBombError:
        raise MediaPolicyError(f"dimensions are outside {SIDE_MIN}-{SIDE_MAX} pixels per side") from None
    except Exception:  # noqa: BLE001 - Pillow raises OSError/SyntaxError/ValueError/struct.error for bad data
        raise MediaPolicyError("image is corrupt or truncated") from None


def _video_bytes(raw: bytes) -> float:
    if not sniff_video(raw):
        detected = sniff_image(raw)
        raise MediaPolicyError(
            "unsupported format" + (f" ({detected})" if detected and detected != "MP4" else "") + "; allowed: MP4, MOV"
        )
    try:
        return inspect_video(raw, CAUSYN_VIDEO_LIMITS)
    except RewriteError as exc:
        raise MediaPolicyError(str(exc)) from None
    except MediaPolicyError:
        raise
    except Exception:  # noqa: BLE001
        raise MediaPolicyError("video is corrupt or cannot be read") from None


def _audio_bytes(raw: bytes) -> float:
    detected = sniff_audio(raw)
    if detected not in ("WAV", "MP3"):
        named = f" ({detected})" if detected else ""
        raise MediaPolicyError(f"unsupported format{named}; allowed: WAV, MP3")
    try:
        duration = inspect_audio(raw)
    except RewriteError as exc:
        raise MediaPolicyError(str(exc)) from None
    except Exception:  # noqa: BLE001
        raise MediaPolicyError("audio is corrupt or cannot be read") from None
    if duration < AUDIO_MIN_SECONDS - CAUSYN_VIDEO_LIMITS.tolerance:
        raise MediaPolicyError("each reference audio clip must last 2-15 seconds")
    return duration


def _validate_data_url(ref: MediaRef) -> float:
    declared, body = _split_data_url(ref.url)
    if not declared.startswith(ref.kind + "/"):
        raise MediaPolicyError(f"declared media type '{declared or 'empty'}' is not {_article(ref.kind)}")
    limit = {"image": IMAGE_MAX_BYTES, "video": VIDEO_MAX_BYTES, "audio": AUDIO_MAX_BYTES}[ref.kind]
    raw = _decode(body, limit, ref.kind)
    if ref.kind == "image":
        _image_bytes(raw)
        return 0.0
    return _video_bytes(raw) if ref.kind == "video" else _audio_bytes(raw)


def _article(kind: str) -> str:
    return "an image" if kind == "image" else f"a {kind}"


# ------------------------------------------------------------------ SSRF


async def resolve_host(host: str, port: int) -> list[str]:
    """Every A/AAAA answer for ``host``. Patched in tests; asynchronous so a slow resolver never blocks the loop."""
    lookup = functools.partial(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    infos = await _gate("dns").run(lookup, DNS_TIMEOUT_S)
    return [str(info[4][0]) for info in infos]


def _non_global(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or any(ip in net for net in _EXTRA_BLOCKED if ip.version == net.version)


_NUMERIC_LABEL = re.compile(r"0x[0-9a-f]*|[0-9]+", re.IGNORECASE)


def _is_numeric_ipv4(labels: list[str]) -> bool:
    """inet_aton semantics: 1-4 dot-separated labels that are ALL numeric (decimal, octal or 0x hex).

    A domain that merely consists of hex letters ("bad.cafe", "face.be") has a non-numeric label and is not matched.
    """
    return 1 <= len(labels) <= 4 and all(_NUMERIC_LABEL.fullmatch(label) for label in labels)


def _check_url_syntax(url: str) -> tuple[str, int]:
    try:
        parts = urlsplit(url)
    except ValueError:
        raise MediaPolicyError(PUBLIC_HOST_MESSAGE) from None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise MediaPolicyError("media must be a public HTTP(S) URL or Base64 data URL")
    try:
        port = parts.port
    except ValueError:
        raise MediaPolicyError(PUBLIC_HOST_MESSAGE) from None
    port = port if port is not None else (443 if parts.scheme == "https" else 80)
    host = parts.hostname.lower().rstrip(".")
    labels = host.split(".")
    if (
        not host
        or not host.isascii()  # fullwidth digits / ideographic dots normalise to IPv4 in some resolvers
        or "%" in host  # IPv6 zone ids and percent-encoding tricks
        or ":" in host  # IPv6 literal in any spelling
        or "" in labels  # empty label ("a..b")
        or len(labels) < 2  # single label: internal service names
        or host == "localhost"
        or host.endswith(_BLOCKED_SUFFIXES)
        or _is_numeric_ipv4(labels)
        or not (port in (80, 443) or port >= 1024)
    ):
        raise MediaPolicyError(PUBLIC_HOST_MESSAGE)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host, port
    raise MediaPolicyError(PUBLIC_HOST_MESSAGE)


async def _check_public_url(url: str, lookups: dict[tuple[str, int], asyncio.Future]) -> None:
    host, port = _check_url_syntax(url)
    try:
        # One lookup per distinct host within a request.
        if (host, port) not in lookups:
            lookups[(host, port)] = asyncio.ensure_future(asyncio.wait_for(resolve_host(host, port), DNS_TIMEOUT_S))
        answers = await asyncio.shield(lookups[(host, port)])
    except (OSError, TimeoutError, asyncio.TimeoutError, UnicodeError):
        raise MediaPolicyError("media URL host could not be resolved") from None
    if not answers:
        raise MediaPolicyError("media URL host could not be resolved")
    try:
        if any(_non_global(answer) for answer in answers):
            raise MediaPolicyError(PUBLIC_HOST_MESSAGE)
    except ValueError:
        raise MediaPolicyError(PUBLIC_HOST_MESSAGE) from None


# ------------------------------------------------------------------ entry points


async def _check_one(ref: MediaRef, lookups: dict[tuple[str, int], asyncio.Future]) -> float:
    try:
        if ref.url.startswith("data:"):
            try:
                return await _gate("decode").run(functools.partial(_validate_data_url, ref), DECODE_TIMEOUT_S)
            except (TimeoutError, asyncio.TimeoutError):
                raise MediaPolicyError(f"{ref.kind} could not be validated in time") from None
        await _check_public_url(ref.url, lookups)
        return 0.0
    except MediaPolicyError as exc:
        raise MediaPolicyError(f"{ref.label}: {exc}") from None


async def validate_public_media(refs: Sequence[MediaRef]) -> None:
    """Raise :class:`MediaPolicyError` for the first offending reference in request order."""
    if len(refs) > MAX_REFERENCES:
        raise MediaPolicyError(f"too many media references (at most {MAX_REFERENCES})")
    lookups: dict[tuple[str, int], asyncio.Future] = {}
    outcomes = await asyncio.gather(*(_check_one(ref, lookups) for ref in refs), return_exceptions=True)
    durations: dict[Kind, float] = {"image": 0.0, "video": 0.0, "audio": 0.0}
    for ref, outcome in zip(refs, outcomes):
        if isinstance(outcome, BaseException):
            raise outcome
        durations[ref.kind] += outcome
    if durations["video"] > CAUSYN_VIDEO_REF_TOTAL_MAX_SECONDS + CAUSYN_VIDEO_LIMITS.tolerance:
        raise MediaPolicyError(
            f"combined reference video duration exceeds {CAUSYN_VIDEO_REF_TOTAL_MAX_SECONDS:g} seconds"
        )
    if durations["audio"] > AUDIO_TOTAL_MAX_SECONDS:
        raise MediaPolicyError("combined reference audio duration exceeds 15 seconds")


async def validate_public_payload(payload: Mapping[str, object]) -> None:
    await validate_public_media(extract_media_refs(payload))
