from __future__ import annotations

import base64
import io
import struct
import zlib

import pytest
from PIL import Image

from litellm.llms.causyn import public_media_policy as policy
from litellm.llms.causyn.public_media_policy import MediaPolicyError, MediaRef, extract_media_refs

MB = 1024 * 1024


def data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def encoded(fmt: str, size: tuple[int, int] = (300, 300), color=(200, 30, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format=fmt)
    return out.getvalue()


def noisy_png(size: tuple[int, int] = (300, 300)) -> bytes:
    import os

    out = io.BytesIO()
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(out, format="PNG")
    return out.getvalue()


def png_declaring(width: int, height: int) -> bytes:
    """A structurally valid PNG header declaring absurd dimensions; no pixel data is ever decoded."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\0")) + chunk(b"IEND", b"")


def image(url: str, role: str = "reference") -> MediaRef:
    return MediaRef(kind="image", url=url, role=role, label="reference image 1")


async def check(*refs: MediaRef) -> None:
    await policy.validate_public_media(list(refs))


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    table: dict[str, list[str]] = {}

    async def resolve(host: str, port: int) -> list[str]:
        if host in table:
            return table[host]
        if host.endswith(".gone.example"):
            raise OSError("no such host")
        return ["93.184.216.34"]

    monkeypatch.setattr(policy, "resolve_host", resolve)
    return table


# ---------------------------------------------------------------- images

@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
async def test_allowed_formats_pass(fmt):
    await check(image(data_url(encoded(fmt), "image/" + fmt.lower())))


@pytest.mark.parametrize("size", [(256, 256), (640, 256), (256, 640), (2560, 1024), (5760, 2304)])
async def test_boundary_dimensions_pass(size):
    await check(image(data_url(encoded("PNG", size))))


async def test_declared_type_does_not_have_to_match_actual_allowed_format():
    # A declared png holding real JPEG bytes is accepted as JPEG: the bytes decide the format.
    await check(image(data_url(encoded("JPEG"), "image/png")))


def gif() -> bytes:
    return encoded("GIF")


def bmp() -> bytes:
    return encoded("BMP")


def tiff() -> bytes:
    return encoded("TIFF")


def ico() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64)).save(out, format="ICO")
    return out.getvalue()


@pytest.mark.parametrize(
    "raw,mime,name",
    [
        (gif(), "image/gif", "GIF"),
        (bmp(), "image/bmp", "BMP"),
        (tiff(), "image/tiff", "TIFF"),
        (ico(), "image/x-icon", "ICO"),
        (b'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300"/>', "image/svg+xml", "SVG"),
        (b"%PDF-1.7\n" + b"0" * 400, "image/png", "PDF"),
        (b"<!DOCTYPE html><html><body>x</body></html>", "image/png", "HTML"),
        (b"\x7fELF" + b"\0" * 400, "image/png", "ELF"),
        (b"PK\x03\x04" + b"\0" * 400, "image/png", "ZIP"),
        (b"\x00\x01\x02 nonsense bytes" * 50, "image/png", None),
    ],
)
async def test_unsupported_formats_are_named_and_rejected(raw, mime, name):
    with pytest.raises(MediaPolicyError) as exc:
        await check(image(data_url(raw, mime)))
    message = str(exc.value)
    assert message.startswith("reference image 1: unsupported format")
    assert "allowed: JPEG, PNG, WEBP, HEIC, HEIF" in message
    if name:
        assert f"({name})" in message


async def test_non_image_declared_type_is_rejected():
    with pytest.raises(MediaPolicyError, match="declared media type 'application/pdf' is not an image"):
        await check(image(data_url(encoded("PNG"), "application/pdf")))
    with pytest.raises(MediaPolicyError, match="declared media type 'video/mp4' is not an image"):
        await check(image(data_url(encoded("PNG"), "video/mp4")))


async def test_truncated_and_corrupt_images_are_rejected():
    raw = noisy_png((300, 300))
    with pytest.raises(MediaPolicyError, match="reference image 1: .*(corrupt|truncated)"):
        await check(image(data_url(raw[: len(raw) // 2])))
    with pytest.raises(MediaPolicyError, match="corrupt|truncated"):
        await check(image(data_url(raw[:40])))
    broken = bytearray(raw)
    for index in range(len(broken) // 2, len(broken) // 2 + 64):
        broken[index] ^= 0xFF
    with pytest.raises(MediaPolicyError, match="corrupt|truncated"):
        await check(image(data_url(bytes(broken))))
    jpeg = encoded("JPEG")
    with pytest.raises(MediaPolicyError, match="corrupt|truncated"):
        await check(image(data_url(jpeg[: len(jpeg) // 2], "image/jpeg")))


@pytest.mark.parametrize(
    "size",
    [(100, 100), (256, 255), (255, 256), (6000, 3000), (5761, 512), (768, 256), (256, 768), (2000, 700), (700, 2000)],
)
async def test_dimension_and_aspect_limits(size):
    with pytest.raises(MediaPolicyError, match="reference image 1: .*(dimensions|aspect)"):
        await check(image(data_url(encoded("PNG", size))))


async def test_decompression_bomb_is_rejected_without_decoding(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pixels must not be decoded for out-of-range dimensions")

    monkeypatch.setattr(Image.Image, "load", forbidden)
    with pytest.raises(MediaPolicyError, match="dimensions"):
        await check(image(data_url(png_declaring(30000, 30000))))
    with pytest.raises(MediaPolicyError, match="dimensions"):
        await check(image(data_url(png_declaring(6000, 6000))))


async def test_oversized_image_is_rejected_before_decoding():
    raw = b"\xff\xd8\xff\xe0" + b"0" * (30 * MB + 10)
    with pytest.raises(MediaPolicyError, match="reference image 1: .*exceeds 30 MB"):
        await check(image(data_url(raw, "image/jpeg")))


async def test_invalid_base64_payload_is_rejected():
    with pytest.raises(MediaPolicyError, match="Base64"):
        await check(image("data:image/png;base64,@@@not-base64@@@"))


async def test_heic_brand_is_recognised_and_avif_is_rejected():
    pytest.importorskip("pillow_heif")
    avif = b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf" + b"\0" * 100
    with pytest.raises(MediaPolicyError, match=r"\(AVIF\)"):
        await check(image(data_url(avif, "image/avif")))


def test_heif_sniff_accepts_heic_brands_without_needing_a_decoder():
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\0" * 64
    assert policy.sniff_image(heic) == "HEIF"
    assert policy.sniff_image(b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf") == "AVIF"


# ---------------------------------------------------------------- video / audio

def mp4_bytes(brand: bytes = b"isom") -> bytes:
    return b"\x00\x00\x00\x18ftyp" + brand + b"\x00\x00\x02\x00isomiso2" + b"\0" * 200


async def test_video_is_sniffed_size_limited_and_inspected(monkeypatch):
    seen = []

    def inspect(raw: bytes, limits) -> float:
        seen.append(len(raw))
        return 6.0

    monkeypatch.setattr(policy, "inspect_video", inspect)
    ok = MediaRef(kind="video", url=data_url(mp4_bytes(), "video/mp4"), role="reference", label="reference video 1")
    await check(ok, ok)
    assert seen == [len(mp4_bytes())] * 2
    mov = MediaRef(kind="video", url=data_url(mp4_bytes(b"qt  "), "video/quicktime"), role="reference", label="reference video 1")
    await check(mov)
    with pytest.raises(MediaPolicyError, match="reference video 1: unsupported format"):
        await check(MediaRef(kind="video", url=data_url(gif(), "video/mp4"), role="reference", label="reference video 1"))
    with pytest.raises(MediaPolicyError, match="declared media type"):
        await check(MediaRef(kind="video", url=data_url(mp4_bytes(), "image/png"), role="reference", label="reference video 1"))
    big = mp4_bytes() + b"0" * (50 * MB)
    with pytest.raises(MediaPolicyError, match="reference video 1: .*exceeds 50 MB"):
        await check(MediaRef(kind="video", url=data_url(big, "video/mp4"), role="reference", label="reference video 1"))


async def test_video_inspection_errors_and_total_duration(monkeypatch):
    from litellm.llms.causyn.h3_prompt import RewriteError

    def reject(raw, limits):
        raise RewriteError("Reference dimensions must be 256-5760 pixels and aspect ratio 0.4-2.5", 400)

    monkeypatch.setattr(policy, "inspect_video", reject)
    ref = MediaRef(kind="video", url=data_url(mp4_bytes(), "video/mp4"), role="reference", label="reference video 2")
    with pytest.raises(MediaPolicyError, match="reference video 2: Reference dimensions"):
        await check(ref)
    monkeypatch.setattr(policy, "inspect_video", lambda raw, limits: 9.0)
    with pytest.raises(MediaPolicyError, match="combined reference video duration"):
        await check(ref, ref)


def wav_bytes() -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + 24) + b"WAVEfmt " + b"\0" * 16


async def test_audio_is_wav_or_mp3_and_size_limited(monkeypatch):
    monkeypatch.setattr(policy, "inspect_audio", lambda raw: 3.0)

    def audio(raw: bytes, mime="audio/wav") -> MediaRef:
        return MediaRef(kind="audio", url=data_url(raw, mime), role="reference", label="reference audio 1")

    await check(audio(wav_bytes()))
    await check(audio(b"ID3\x03\x00" + b"\0" * 64, "audio/mpeg"))
    await check(audio(b"\xff\xfb\x90\x00" + b"\0" * 64, "audio/mpeg"))
    with pytest.raises(MediaPolicyError, match=r"reference audio 1: unsupported format \(.*\); allowed: WAV, MP3"):
        await check(audio(b"OggS" + b"\0" * 64, "audio/ogg"))
    with pytest.raises(MediaPolicyError, match="unsupported format"):
        await check(audio(b"\xff\xf1\x50\x80" + b"\0" * 64, "audio/aac"))
    with pytest.raises(MediaPolicyError, match="exceeds 15 MB"):
        await check(audio(wav_bytes() + b"0" * (15 * MB)))


# ---------------------------------------------------------------- SSRF

@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a.png",
        "http://100.100.100.200/latest/meta-data",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/a.png",
        "http://172.16.3.4/a.png",
        "http://192.168.1.1/a.png",
        "http://[::1]/a.png",
        "http://[::ffff:127.0.0.1]/a.png",
        "http://[::ffff:7f00:1]/a.png",
        "http://[fe80::1]/a.png",
        "http://[2606:4700:4700::1111]/a.png",
        "http://8.8.8.8/a.png",
        "http://0.0.0.0/a.png",
        "http://2130706433/a.png",
        "http://0x7f000001/a.png",
        "http://0x7f.0.0.1/a.png",
        "http://0177.0.0.1/a.png",
        "http://127.1/a.png",
        "http://drama-litellm:4000/a.png",
        "http://drama-platform:3000/a.png",
        "http://localhost/a.png",
        "http://LOCALHOST./a.png",
        "http://foo.localhost/a.png",
        "http://db.internal/a.png",
        "http://x.svc/a.png",
        "http://x.ns.svc.cluster.local/a.png",
        "https://printer.local/a.png",
        "http://minio.cluster.local/a.png",
        "http://example.com:22/a.png",
        "http://example.com:0/a.png",
        "http://example.com:99999/a.png",
        "gopher://example.com/a.png",
        "file:///etc/passwd",
    ],
)
async def test_ssrf_targets_are_rejected_at_submit(url):
    with pytest.raises(MediaPolicyError) as exc:
        await check(image(url))
    assert str(exc.value).startswith("reference image 1: ")


@pytest.mark.parametrize("url", ["http://127.0.0.1/a.png", "http://db.internal/", "http://drama-litellm:4000/"])
async def test_ssrf_message_is_the_documented_one(url):
    with pytest.raises(MediaPolicyError, match="media URL must point to a public internet host"):
        await check(image(url))


@pytest.mark.parametrize(
    "answers",
    [["10.1.2.3"], ["93.184.216.34", "10.1.2.3"], ["100.64.0.1"], ["100.100.100.200"], ["169.254.169.254"],
     ["::1"], ["::ffff:10.0.0.1"], ["fd00::1"], ["64:ff9b::7f00:1"], ["198.18.0.1"], ["0.0.0.0"], ["168.63.129.16"]],
)
async def test_hostnames_resolving_to_non_global_addresses_are_rejected(public_dns, answers):
    public_dns["cdn.example.com"] = answers
    with pytest.raises(MediaPolicyError, match="media URL must point to a public internet host"):
        await check(image("https://cdn.example.com/a.png"))


async def test_public_hosts_and_allowed_ports_pass(public_dns):
    public_dns["cdn.example.com"] = ["93.184.216.34", "2606:2800:220:1::1"]
    for url in (
        "https://cdn.example.com/a.png",
        "http://cdn.example.com/a.png",
        "https://cdn.example.com:443/a.png",
        "http://cdn.example.com:80/a.png",
        "https://cdn.example.com:8443/a.png",
        "https://CDN.Example.com./a.png?sig=1",
    ):
        await check(image(url))


async def test_dns_failure_and_timeout_are_client_errors(monkeypatch):
    with pytest.raises(MediaPolicyError, match="could not be resolved"):
        await check(image("https://missing.gone.example/a.png"))

    async def hang(host, port):
        import asyncio

        await asyncio.sleep(30)
        return []

    monkeypatch.setattr(policy, "resolve_host", hang)
    monkeypatch.setattr(policy, "DNS_TIMEOUT_S", 0.05)
    with pytest.raises(MediaPolicyError, match="could not be resolved"):
        await check(image("https://slow.example.com/a.png"))


async def test_empty_dns_answer_is_rejected(public_dns):
    public_dns["empty.example.com"] = []
    with pytest.raises(MediaPolicyError, match="could not be resolved"):
        await check(image("https://empty.example.com/a.png"))


async def test_real_resolver_uses_the_event_loop(monkeypatch):
    # Not patched: resolve_host must accept a literal-free name via loop.getaddrinfo.
    monkeypatch.undo()
    answers = await policy.resolve_host("localhost", 80)
    assert answers and all(isinstance(a, str) for a in answers)


# ---------------------------------------------------------------- extraction

def test_extract_from_v1_videos_references_and_frames():
    payload = {
        "model": "causyn-1.1",
        "image": "https://cdn.example.com/first.png",
        "last_image": "https://cdn.example.com/last.png",
    }
    refs = extract_media_refs(payload)
    assert [(r.kind, r.role) for r in refs] == [("image", "first_frame"), ("image", "last_frame")]
    payload = {
        "model": "causyn-1.1",
        "references": [
            {"role": "reference", "media_type": "image", "url": "https://cdn.example.com/1.png"},
            {"role": "reference", "media_type": "video", "url": "https://cdn.example.com/1.mp4"},
            {"role": "reference", "media_type": "image", "url": "https://cdn.example.com/2.png"},
            {"role": "reference", "media_type": "audio", "url": "https://cdn.example.com/1.wav"},
        ],
    }
    refs = extract_media_refs(payload)
    assert [r.label for r in refs] == ["reference image 1", "reference video 1", "reference image 2", "reference audio 1"]


def test_extract_from_h3_content_shape_and_labels_keyframes():
    payload = {
        "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": "https://cdn.example.com/f.png"}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": "https://cdn.example.com/l.png"}, "role": "last_frame"},
            {"type": "video_url", "video_url": {"url": "https://cdn.example.com/v.mp4"}},
            {"type": "audio_url", "audio_url": {"url": "https://cdn.example.com/a.mp3"}},
        ]
    }
    assert [r.label for r in extract_media_refs(payload)] == [
        "first frame image",
        "last frame image",
        "reference video 1",
        "reference audio 1",
    ]


async def test_first_failure_in_request_order_wins_and_indexes_are_one_based():
    refs = extract_media_refs(
        {
            "references": [
                {"role": "reference", "media_type": "image", "url": data_url(encoded("PNG"))},
                {"role": "reference", "media_type": "image", "url": data_url(gif(), "image/png")},
                {"role": "reference", "media_type": "image", "url": "http://127.0.0.1/x.png"},
            ]
        }
    )
    with pytest.raises(MediaPolicyError, match=r"^reference image 2: unsupported format \(GIF\)"):
        await policy.validate_public_media(refs)


async def test_unsupported_url_scheme_is_rejected():
    with pytest.raises(MediaPolicyError, match="public HTTP"):
        await check(image("ftp://cdn.example.com/a.png"))
