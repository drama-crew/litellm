from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.causyn import h3_media, h3_prompt
from litellm.llms.causyn.h3_prompt import (
    ContextIRRequest,
    H3PromptRewriter,
    RefFacts,
    RewriteError,
    validate_prompt,
)
from litellm.llms.causyn.ref_media_facts import AudioFacts, Keyframe, VideoFacts

DATA = Path(__file__).parent / "data"
PROMPTS = json.loads((DATA / "ref2va_prompts.json").read_text(encoding="utf-8"))
GOLD, OLD = PROMPTS["gold"], PROMPTS["old"]

IMG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/"
VID = "data:video/mp4;base64,AAAA"
AUD = "https://media.example/a.wav"

CUT_MSG = "The rewritten H3 prompt does not mirror the source video's shot cuts"


def item(kind: str, url: str | None = None) -> dict:
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": url or IMG}, "role": "reference_image"}
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": url or VID}, "role": "reference_video"}
    return {"type": "audio_url", "audio_url": {"url": url or AUD}, "role": "reference_audio"}


def ref_spec(*kinds: str, duration: int = 15) -> ContextIRRequest:
    return ContextIRRequest.model_validate(
        {
            "model": "causyn-1.1",
            "content": [{"type": "text", "text": "Replace the runner."}, *[item(kind) for kind in kinds]],
            "duration": duration,
            "ratio": "16:9",
        }
    )


def vfacts(cut: float | None, duration: float = 5.0, with_frames: bool = True) -> VideoFacts:
    frames: tuple[Keyframe, ...] = ()
    if with_frames:
        bounds = [0.0, *([cut] if cut is not None else []), duration]
        frames = tuple(
            Keyframe(t=round(bounds[i] + 0.15, 3), shot=i + 1, jpeg=b"\xff\xd8\xff-jpeg-%d" % i)
            for i in range(len(bounds) - 1)
        )
    return VideoFacts(duration, 24.0, 1344, 768, None if cut is None else (cut,), frames)


def afacts() -> AudioFacts:
    return AudioFacts(15.0, 44100, 2, (-30.0,) * 30, (6.0, 11.3), 3.2, "percussive/rhythmic")


GOLD_CASES = {
    "R1": (("image", "video"), 5, RefFacts(videos=(vfacts(3.583),))),
    "R2": (("image", "video"), 15, RefFacts(videos=(vfacts(3.542, 15.0),))),
    "R3": (
        ("image", "video", "video", "video"),
        15,
        RefFacts(videos=(vfacts(3.542), vfacts(None), vfacts(None))),
    ),
    "R4": (
        ("image", "video", "audio"),
        15,
        RefFacts(videos=(vfacts(3.542, 15.0),), audios=(afacts(),), audio_raw=(b"RIFFxxxxWAVE",)),
    ),
}


def case(name: str) -> tuple[ContextIRRequest, RefFacts]:
    kinds, duration, facts = GOLD_CASES[name]
    return ref_spec(*kinds, duration=duration), facts


# ----------------------------------------------------------------------------- golden prompts


@pytest.mark.parametrize("name", ["R1", "R2", "R3", "R4"])
def test_hand_written_official_prompts_pass(name):
    spec, facts = case(name)
    validate_prompt(GOLD[name], spec, facts)


@pytest.mark.parametrize("name", ["R1", "R2", "R4"])
def test_old_single_shot_prompts_fail_cut_alignment(name):
    spec, facts = case(name)
    with pytest.raises(RewriteError) as caught:
        validate_prompt(OLD[name], spec, facts)
    cut = [v for v in caught.value.violations if v.startswith(CUT_MSG)]
    assert len(cut) == 1 and "00:03." in cut[0]


def test_old_three_video_prompt_is_exempt_from_cut_alignment():
    spec, facts = case("R3")
    try:
        validate_prompt(OLD["R3"], spec, facts)
    except RewriteError as exc:
        assert not any(v.startswith(CUT_MSG) for v in exc.violations)


# ----------------------------------------------------------------------------- validator matrix


def mutate(name: str, old: str, new: str) -> str:
    text = GOLD[name]
    assert old in text
    return text.replace(old, new, 1)


def violations(prompt: str, spec: ContextIRRequest, facts: RefFacts | None) -> tuple[str, ...]:
    with pytest.raises(RewriteError) as caught:
        validate_prompt(prompt, spec, facts)
    return caught.value.violations or (str(caught.value),)


def test_missing_cut_shot_is_reported_with_fixed_message():
    spec, facts = case("R1")
    prompt = mutate("R1", "[Shot 2] At 00:03.583, the shot cuts", "[Shot 2] The shot cuts")
    with pytest.raises(RewriteError) as caught:
        validate_prompt(prompt, spec, facts)
    assert str(caught.value) == CUT_MSG
    assert any("00:03.583" in v for v in caught.value.violations)


def test_cut_within_tolerance_passes_and_beyond_fails():
    spec, facts = case("R1")
    validate_prompt(mutate("R1", "At 00:03.583, the", "At 00:03.900, the"), spec, facts)  # 0.317 s off
    with pytest.raises(RewriteError) as caught:
        validate_prompt(mutate("R1", "At 00:03.583, the", "At 00:04.000, the"), spec, facts)  # 0.417 s off
    assert str(caught.value) == CUT_MSG


def cut_findings(prompt: str, spec: ContextIRRequest, facts: RefFacts | None) -> list[str]:
    try:
        validate_prompt(prompt, spec, facts)
    except RewriteError as exc:
        return [v for v in exc.violations if v.startswith(CUT_MSG)]
    return []


def test_cut_rule_needs_video_editing_summary_single_video_and_cuts():
    spec, facts = case("R1")
    no_edit = mutate("R1", "[video editing + reference generation]", "[reference generation]")
    no_edit = no_edit.replace("At 00:03.583, the", "Then the", 1)
    validate_prompt(no_edit, spec, facts)
    assert cut_findings(OLD["R1"], spec, facts)  # the rule does fire for the weak prompt
    # no detected cuts / detection unavailable
    assert not cut_findings(OLD["R1"], spec, RefFacts(videos=(vfacts(None),)))
    assert not cut_findings(OLD["R1"], spec, None)
    # cut beyond the target duration is not a cut of the target
    short = ref_spec("image", "video", duration=5)
    assert not cut_findings(OLD["R1"], short, RefFacts(videos=(vfacts(5.4, 6.0),)))


def test_unknown_or_missing_media_labels_fail():
    spec, facts = case("R1")
    found = violations(mutate("R1", "<Video 1> is the source", "<Video 2> is the source"), spec, facts)
    assert found[0].startswith("The rewritten H3 prompt must use exactly the provided media labels")
    # provided label used elsewhere but never defined in subject_definitions
    prompt = GOLD["R4"]
    spec4, facts4 = case("R4")
    start = prompt.index("<Audio 1> is a complete")
    end = prompt.index("\n", start)
    found = violations(prompt[:start] + prompt[end + 1 :], spec4, facts4)
    assert any(v.startswith("The rewritten H3 prompt must define every provided media label in subject_definitions") for v in found)


def test_provided_label_that_is_never_used_fails():
    spec, facts = case("R4")
    prompt = GOLD["R4"].replace("<Audio 1>", "the audio")
    assert violations(prompt, spec, facts)


def test_shot_numbering_must_be_sequential_from_one():
    spec, facts = case("R1")
    with pytest.raises(RewriteError) as caught:
        validate_prompt(mutate("R1", "[Shot 2] At", "[Shot 3] At"), spec, facts)
    assert str(caught.value) == "The rewritten H3 prompt has invalid shot numbering"


def test_timestamps_must_increase_and_stay_below_duration():
    spec, facts = case("R2")
    extra = mutate("R2", "At about 00:11.000", "At 00:02.000")  # goes backwards after 00:03.542
    with pytest.raises(RewriteError) as caught:
        validate_prompt(extra, spec, facts)
    assert str(caught.value) == "The rewritten H3 prompt has invalid shot timestamps"
    beyond = mutate("R2", "At about 00:11.000", "At 00:15.000")  # == duration, not below
    with pytest.raises(RewriteError) as caught2:
        validate_prompt(beyond, spec, facts)
    assert str(caught2.value) == "The rewritten H3 prompt has invalid shot timestamps"
    spec5, facts5 = case("R1")
    with pytest.raises(RewriteError):
        validate_prompt(mutate("R1", "At 00:03.583", "At 00:05.000"), spec5, facts5)


def test_equal_timestamps_are_not_strictly_increasing():
    spec, facts = case("R2")
    equal = mutate("R2", "At about 00:11.000", "At 00:03.542")
    with pytest.raises(RewriteError) as caught:
        validate_prompt(equal, spec, facts)
    assert str(caught.value) == "The rewritten H3 prompt has invalid shot timestamps"


def test_detailed_description_length_bounds():
    spec, facts = case("R1")
    text = GOLD["R1"]
    head, rest = text.split("detailed_description:\n", 1)
    _, tail = rest.split("\n\noverall_soundscape:", 1)
    short = head + "detailed_description:\n[Shot 1] A runner sprints.\n[Shot 2] At 00:03.583, a low shot." + (
        "\n\noverall_soundscape:" + tail
    )
    with pytest.raises(RewriteError) as caught:
        validate_prompt(short, spec, facts)
    assert str(caught.value) == "The rewritten H3 prompt detailed_description must have 200 to 750 words"
    long_body = "[Shot 1] " + "word " * 760 + "\n[Shot 2] At 00:03.583, " + "word " * 5
    long = head + "detailed_description:\n" + long_body + "\n\noverall_soundscape:" + tail
    with pytest.raises(RewriteError):
        validate_prompt(long, spec, facts)


def test_retention_lines_must_start_with_a_defined_label():
    spec, facts = case("R1")
    prompt = mutate("R1", "<Subject 2> (appears in [Shot 1], [Shot 2]): fully", "The plaza (appears): fully")
    with pytest.raises(RewriteError) as caught:
        validate_prompt(prompt, spec, facts)
    assert str(caught.value) == "The rewritten H3 prompt retention_analysis lines must start with a defined label"
    undefined = mutate("R1", "<Subject 2> (appears in [Shot 1], [Shot 2]): fully", "<Subject 9> (appears): fully")
    with pytest.raises(RewriteError):
        validate_prompt(undefined, spec, facts)


def test_all_violations_are_collected_for_repair():
    spec, facts = case("R1")
    found = violations(OLD["R1"], spec, facts)
    assert len(found) >= 2  # cut + length at least


def test_image_only_ref2va_keeps_working_without_facts():
    spec = ref_spec("image", "image", duration=8)
    desc = " ".join(["She walks along the quiet street while the camera slowly follows her pace."] * 20)
    prompt = (
        "subject_definitions:\n<Subject 1> is the woman from <Picture 1> in the coat of <Picture 2>.\n\n"
        "summary:\n[reference generation] A woman walks.\n\n"
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - face and coat.\n"
        "<Picture 2>: fully_preserved - the coat.\n\n"
        f"detailed_description:\nA calm look.\n[Shot 1] {desc}\n\n"
        "overall_soundscape:\nSoft footsteps.\n\nnon_diegetic_music:\nN/A"
    )
    validate_prompt(prompt, spec, RefFacts())
    validate_prompt(prompt, spec, None)


def test_non_ref2va_validation_is_unchanged():
    spec = ContextIRRequest.model_validate(
        {"model": "MiniMax-H3", "content": [{"type": "text", "text": "A cat."}], "duration": 5, "ratio": "16:9"}
    )
    prompt = (
        "integrated_multimodal_description:\n[Shot 1] A cat walks.\n\noverall_soundscape:\nPurring.\n\n"
        "non_diegetic_music:\nN/A"
    )
    validate_prompt(prompt, spec)


# ----------------------------------------------------------------------------- rewriter


class Provider:
    def __init__(self, *answers: str, model: str = "qwen/qwen3.8-flash") -> None:
        self.answers = list(answers)
        self.model = model
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        answer = self.answers[len(self.bodies) - 1]
        return httpx.Response(
            200,
            json={
                "model": self.model,
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": 0.001},
            },
        )


@pytest.fixture
def patched(monkeypatch):
    def install(spec: ContextIRRequest, facts: RefFacts, raws=None):
        async def prepare(client, request):
            return request, h3_media.PreparedRaw(videos=(b"v",) * len(facts.videos), audios=tuple(facts.audio_raw))

        async def perceive(request, raw):
            return facts

        monkeypatch.setattr(h3_media, "prepare_media_with_raw", prepare)
        monkeypatch.setattr(h3_prompt, "perceive_media", perceive)

    return install


async def run(provider: Provider, spec: ContextIRRequest):
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        return await H3PromptRewriter(client, "key").rewrite(spec)


@pytest.mark.asyncio
async def test_valid_first_answer_makes_one_call(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(GOLD["R1"])
    result = await run(provider, spec)
    assert len(provider.bodies) == 1
    assert result.prompt == GOLD["R1"].strip()


@pytest.mark.asyncio
async def test_invalid_then_valid_makes_exactly_one_repair_call(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(OLD["R1"], GOLD["R1"])
    result = await run(provider, spec)
    assert len(provider.bodies) == 2
    assert result.prompt == GOLD["R1"].strip()
    assert result.usage.prompt_tokens == 200 and result.usage.completion_tokens == 20
    messages = provider.bodies[1]["messages"]
    assert messages[:2] == provider.bodies[0]["messages"][:2]
    assert messages[2] == {"role": "assistant", "content": OLD["R1"].strip()}
    assert messages[3]["role"] == "user"
    assert "00:03.583" in messages[3]["content"] and "200" in messages[3]["content"]
    assert "Bearer" not in json.dumps(messages[3])


@pytest.mark.asyncio
async def test_two_invalid_answers_raise_the_validator_error(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(OLD["R1"], OLD["R1"], GOLD["R1"])
    with pytest.raises(RewriteError) as caught:
        await run(provider, spec)
    assert len(provider.bodies) == 2
    assert str(caught.value) == CUT_MSG or str(caught.value).startswith("The rewritten H3 prompt")
    assert "http" not in str(caught.value)


@pytest.mark.asyncio
async def test_non_ref2va_failure_is_never_repaired():
    spec = ContextIRRequest.model_validate(
        {"model": "MiniMax-H3", "content": [{"type": "text", "text": "A cat."}], "duration": 5, "ratio": "16:9"}
    )
    provider = Provider("not a valid prompt", "also invalid")
    with pytest.raises(RewriteError):
        await run(provider, spec)
    assert len(provider.bodies) == 1


@pytest.mark.asyncio
async def test_repair_is_skipped_when_the_attempt_deadline_is_nearly_spent(patched, monkeypatch):
    spec, facts = case("R1")
    patched(spec, facts)
    import time

    token = h3_prompt.ATTEMPT_DEADLINE.set(time.monotonic() + 1.0)
    try:
        provider = Provider(OLD["R1"], GOLD["R1"])
        with pytest.raises(RewriteError):
            await run(provider, spec)
    finally:
        h3_prompt.ATTEMPT_DEADLINE.reset(token)
    assert len(provider.bodies) == 1


def parts(body: dict) -> list[dict]:
    return body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_ref2va_user_content_has_facts_and_labelled_keyframes_without_raw_video(patched, monkeypatch):
    monkeypatch.delenv("CAUSYN_H3_REWRITE_SEND_VIDEO", raising=False)
    monkeypatch.delenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", raising=False)
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(GOLD["R1"])
    await run(provider, spec)
    content = parts(provider.bodies[0])
    types = [p["type"] for p in content]
    assert "video_url" not in types
    texts = [p["text"] for p in content if p["type"] == "text"]
    assert any(t.startswith("<Video 1>: 5.00 s") and "[Shot 2] 00:03.583" in t for t in texts)
    labels = [t for t in texts if "frame at" in t]
    assert labels == ["<Video 1> frame at 00:00.150 (Shot 1)", "<Video 1> frame at 00:03.733 (Shot 2)"]
    index = {t: i for i, p in enumerate(content) for t in [p.get("text")] if t}
    jpeg_parts = [p for p in content if p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg")]
    assert any(base64.b64encode(k.jpeg).decode() in p["image_url"]["url"] for k in facts.videos[0].keyframes for p in jpeg_parts)
    # each keyframe image directly follows its label, in playing order
    for label in labels:
        assert content[index[label] + 1]["type"] == "image_url"
    assert index[labels[0]] < index[labels[1]]
    assert provider.bodies[0]["model"] == "qwen/qwen3.8-flash"


@pytest.mark.asyncio
async def test_send_video_env_keeps_raw_video_part(patched, monkeypatch):
    monkeypatch.setenv("CAUSYN_H3_REWRITE_SEND_VIDEO", "1")
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(GOLD["R1"])
    await run(provider, spec)
    assert "video_url" in [p["type"] for p in parts(provider.bodies[0])]


@pytest.mark.asyncio
async def test_video_without_keyframes_degrades_to_the_raw_video(patched, monkeypatch):
    monkeypatch.delenv("CAUSYN_H3_REWRITE_SEND_VIDEO", raising=False)
    spec = ref_spec("image", "video", duration=5)
    facts = RefFacts(videos=(vfacts(None, with_frames=False),))
    patched(spec, facts)
    provider = Provider(OLD["R1"], OLD["R1"])
    with pytest.raises(RewriteError):
        await run(provider, spec)
    assert "video_url" in [p["type"] for p in parts(provider.bodies[0])]


@pytest.mark.asyncio
async def test_audio_part_is_sent_only_to_the_omni_model(patched, monkeypatch):
    spec, facts = case("R4")
    patched(spec, facts)
    monkeypatch.delenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", raising=False)
    flash = Provider(GOLD["R4"])
    await run(flash, spec)
    assert "input_audio" not in [p["type"] for p in parts(flash.bodies[0])]
    assert any("<Audio 1>: 15.00 s" in p.get("text", "") for p in parts(flash.bodies[0]))

    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "qwen/qwen3.8-omni-flash")
    omni = Provider(GOLD["R4"], model="qwen/qwen3.8-omni-flash")
    result = await run(omni, spec)
    assert omni.bodies[0]["model"] == "qwen/qwen3.8-omni-flash" and result.model == "qwen/qwen3.8-omni-flash"
    audio = [p for p in parts(omni.bodies[0]) if p["type"] == "input_audio"]
    assert audio == [{"type": "input_audio", "input_audio": {"data": base64.b64encode(b"RIFFxxxxWAVE").decode(), "format": "wav"}}]


@pytest.mark.asyncio
async def test_model_env_is_allow_listed_and_ref2va_only(patched, monkeypatch):
    spec, facts = case("R1")
    patched(spec, facts)
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "evil/other-model")
    provider = Provider(GOLD["R1"])
    await run(provider, spec)
    assert provider.bodies[0]["model"] == "qwen/qwen3.8-flash"
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "qwen/qwen3.8-max-0902")
    provider = Provider(GOLD["R1"], model="qwen/qwen3.8-max-0902")
    await run(provider, spec)
    assert provider.bodies[0]["model"] == "qwen/qwen3.8-max-0902"
    # a non-ref2va request keeps flash even when the env is set
    t2va = ContextIRRequest.model_validate(
        {"model": "MiniMax-H3", "content": [{"type": "text", "text": "A cat."}], "duration": 5, "ratio": "16:9"}
    )
    plain = Provider("bad")
    with pytest.raises(RewriteError):
        await run(plain, t2va)
    assert plain.bodies[0]["model"] == "qwen/qwen3.8-flash"


@pytest.mark.asyncio
async def test_ref2va_system_prompt_carries_the_perception_rules(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(GOLD["R1"])
    await run(provider, spec)
    system = provider.bodies[0]["messages"][0]["content"]
    for phrase in (
        "one `[Shot N]` per detected source shot",
        "At MM:SS.mmm",
        "its own `<Subject N>`",
        "(appears in [Shot",
        "350–500 words",
        "Every provided media label appears in `subject_definitions`",
    ):
        assert phrase in system


def test_completion_model_is_an_allow_list_check():
    from litellm.llms.causyn.h3_prompt import _Completion

    base = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}}
    for model in ("qwen/qwen3.8-flash", "qwen/qwen3.8-omni-flash", "qwen/qwen3.8-max-0902"):
        assert _Completion.model_validate({**base, "model": model}).model == model
    with pytest.raises(ValidationError):
        _Completion.model_validate({**base, "model": "other/model"})


# ----------------------------------------------------------------------------- non-ref2va snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "l2va"])
async def test_non_ref2va_payload_is_byte_identical_to_the_previous_code_path(mode, monkeypatch):
    golden = json.loads((DATA / "h3_non_ref2va_payload_golden.json").read_text())[mode]
    media = {
        "t2va": [],
        "i2va": [("first_frame",)],
        "fl2va": [("first_frame",), ("last_frame",)],
        "l2va": [("last_frame",)],
    }[mode]
    content = [{"type": "text", "text": "A cat walks."}] + [
        {"type": "image_url", "image_url": {"url": IMG}, "role": role} for (role,) in media
    ]
    spec = ContextIRRequest.model_validate({"model": "MiniMax-H3", "content": content, "duration": 5, "ratio": "16:9"})
    assert spec.mode == golden["mode"]

    async def prepare(client, request):
        return request, h3_media.PreparedRaw(videos=(), audios=())

    monkeypatch.setattr(h3_media, "prepare_media_with_raw", prepare)
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "qwen/qwen3.8-omni-flash")
    monkeypatch.setenv("CAUSYN_H3_REWRITE_SEND_VIDEO", "1")
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(
            200,
            json={
                "model": h3_prompt.MODEL,
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RewriteError):
            await H3PromptRewriter(client, "k").rewrite(spec)
    assert len(seen) == 1
    assert hashlib.sha256(seen[0]).hexdigest() == golden["sha256"]
