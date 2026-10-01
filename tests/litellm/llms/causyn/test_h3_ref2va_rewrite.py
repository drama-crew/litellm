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
from litellm.proxy.video_endpoints.minimax_h3_models import MediaURL
from litellm.llms.causyn.ref_media_facts import AudioFacts, Keyframe, VideoFacts

DATA = Path(__file__).parent / "data"
PROMPTS = json.loads((DATA / "ref2va_prompts.json").read_text(encoding="utf-8"))
GOLD, OLD = PROMPTS["gold"], PROMPTS["old"]

IMG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/"
VID = "data:video/mp4;base64,AAAA"
AUD = "https://media.example/a.wav"

CUT_MSG = "The rewritten H3 prompt does not mirror the source video's shot cuts"


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    h3_prompt._FACTS_MEMO.clear()
    h3_prompt._FIRST_ANSWERS.clear()
    h3_prompt._warned_models.clear()
    monkeypatch.delenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", raising=False)
    monkeypatch.delenv("CAUSYN_H3_REWRITE_SEND_VIDEO", raising=False)


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
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - face from <Picture 1>\n"
        "and the coat of <Picture 2>.\n"
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

        async def perceive(request, raw, key=None):
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
async def test_video_without_keyframes_sends_facts_only_never_the_raw_video(patched, monkeypatch):
    spec = ref_spec("image", "video", duration=5)
    facts = RefFacts(videos=(vfacts(None, with_frames=False),))
    patched(spec, facts)
    provider = Provider(OLD["R1"], OLD["R1"])
    await run(provider, spec)  # no detected cuts: the old prompt only has soft findings and is accepted
    content = parts(provider.bodies[0])
    assert "video_url" not in [p["type"] for p in content]
    assert any(p.get("text", "").startswith("<Video 1>: ") for p in content)


def test_raw_video_needs_the_env_and_a_size_under_20_mb():
    from types import SimpleNamespace

    def video(n):
        return SimpleNamespace(video_url=SimpleNamespace(url="data:video/mp4;base64," + "A" * n))

    send = ContextIRRequest._perceived_video_parts
    facts = RefFacts(videos=(vfacts(3.0),))
    assert "video_url" in [p["type"] for p in send(video(1000), 1, facts, True)]
    assert "video_url" not in [p["type"] for p in send(video(1000), 1, facts, False)]
    assert "video_url" not in [p["type"] for p in send(video(28 * 1024 * 1024), 1, facts, True)]
    assert "video_url" in [p["type"] for p in send(video(26 * 1024 * 1024), 1, facts, True)]
    nokey = RefFacts(videos=(vfacts(None, with_frames=False),))
    assert "video_url" not in [p["type"] for p in send(video(1000), 1, nokey, False)]


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
async def test_unknown_model_env_fails_closed_for_ref2va_with_a_warning(patched, monkeypatch, caplog):
    spec, facts = case("R1")
    patched(spec, facts)
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "qwen/qwen3.8-omni")
    provider = Provider(GOLD["R1"])
    with caplog.at_level("WARNING"):
        for _ in range(2):
            with pytest.raises(RewriteError) as caught:
                await run(provider, spec)
    assert str(caught.value) == "H3 prompt rewrite model is not configured"
    assert caught.value.status_code == 503 and caught.value.retryable is False
    assert provider.bodies == []
    assert sum("not an allow-listed" in r.message for r in caplog.records) == 1  # warned once


@pytest.mark.asyncio
async def test_model_env_selects_allow_listed_models_and_ref2va_only(patched, monkeypatch):
    spec, facts = case("R1")
    patched(spec, facts)
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "qwen/qwen3.8-max-0902")
    provider = Provider(GOLD["R1"], model="qwen/qwen3.8-max-0902")
    await run(provider, spec)
    assert provider.bodies[0]["model"] == "qwen/qwen3.8-max-0902"
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "typo/model")
    t2va = ContextIRRequest.model_validate(
        {"model": "MiniMax-H3", "content": [{"type": "text", "text": "A cat."}], "duration": 5, "ratio": "16:9"}
    )
    plain = Provider("bad")
    with pytest.raises(RewriteError):
        await run(plain, t2va)
    assert plain.bodies[0]["model"] == "qwen/qwen3.8-flash"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,reasoning,max_tokens",
    [
        ("qwen/qwen3.8-flash", {"enabled": False}, 8192),
        ("qwen/qwen3.8-omni-flash", {"enabled": False}, 8192),
        ("qwen/qwen3.8-max-0902", {"enabled": True, "effort": "low"}, 12000),
    ],
)
async def test_reasoning_and_token_budget_are_per_model(patched, monkeypatch, model, reasoning, max_tokens):
    spec, facts = case("R1")
    patched(spec, facts)
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", model)
    provider = Provider(GOLD["R1"], model=model)
    await run(provider, spec)
    body = provider.bodies[0]
    assert body["reasoning"] == reasoning and body["max_tokens"] == max_tokens
    assert list(body) == ["model", "messages", "max_tokens", "stream", "temperature", "reasoning", "provider"]


@pytest.mark.asyncio
async def test_answer_with_reasoning_fields_is_accepted(patched, monkeypatch):
    spec, facts = case("R1")
    patched(spec, facts)
    monkeypatch.setenv("CAUSYN_H3_REF2VA_REWRITE_MODEL", "qwen/qwen3.8-max-0902")

    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "qwen/qwen3.8-max-0902",
                "choices": [
                    {
                        "message": {"content": GOLD["R1"], "reasoning": "thinking...", "reasoning_details": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await H3PromptRewriter(client, "k").rewrite(spec)
    assert result.prompt == GOLD["R1"].strip()


VIDEO_ONLY = ("one `[Shot N]` per detected source shot", "At MM:SS.mmm", "its own `<Subject N>`", "350–500 words")
COMMON = ("(appears in [Shot", "Every provided media label appears in `subject_definitions`")


@pytest.mark.asyncio
async def test_ref2va_system_prompt_carries_the_perception_rules(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(GOLD["R1"])
    await run(provider, spec)
    system = provider.bodies[0]["messages"][0]["content"]
    for phrase in (*VIDEO_ONLY, *COMMON):
        assert phrase in system


@pytest.mark.asyncio
@pytest.mark.parametrize("kinds", [("image", "image"), ("image", "audio")])
async def test_video_only_rules_are_absent_without_a_reference_video(patched, kinds):
    spec = ref_spec(*kinds, duration=8)
    facts = RefFacts(audios=(afacts(),), audio_raw=(b"RIFFxxxxWAVE",)) if kinds[-1] == "audio" else RefFacts()
    patched(spec, facts)
    provider = Provider("not valid", "not valid")
    with pytest.raises(RewriteError):
        await run(provider, spec)
    system = provider.bodies[0]["messages"][0]["content"]
    added = system[len(h3_prompt.system_prompt()) :]  # the official skill text itself mentions timestamps
    for phrase in VIDEO_ONLY:
        assert phrase not in added
    for phrase in (*COMMON, "`[Shot N]` markers numbered from 1", "200–750 words"):
        assert phrase in added
    assert "official six-section reference format" in system


def test_repair_message_names_the_exact_missing_items():
    message = h3_prompt._repair_message(("A: missing <Audio 1>", "B: 12 words"), ("<Picture 1>", "<Audio 1>"))
    assert "- A: missing <Audio 1>" in message and "- B: 12 words" in message
    assert "<Picture 1>, <Audio 1>" in message


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


# ----------------------------------------------------------------------------- real perception, no patches


def two_shot_clip() -> bytes:
    import io

    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    out = io.BytesIO()
    rng = np.random.default_rng(1)
    with av.open(out, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = stream.height = 256
        stream.pix_fmt = "yuv420p"
        for i in range(72):
            base = 30 if i < 36 else 200
            frame = np.clip(rng.integers(0, 40, (256, 256, 3)) + base, 0, 255).astype("uint8")
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out.getvalue()


@pytest.mark.asyncio
async def test_real_perception_reaches_the_provider_payload():
    clip = "data:video/mp4;base64," + base64.b64encode(two_shot_clip()).decode()
    spec = ContextIRRequest.model_validate(
        {
            "model": "causyn-1.1",
            "content": [
                {"type": "text", "text": "Replace the runner."},
                item("image", "data:image/png;base64," + base64.b64encode(_png()).decode()),
                item("video", clip),
            ],
            "duration": 5,
            "ratio": "16:9",
        }
    )
    provider = Provider("not valid", "still not valid")
    with pytest.raises(RewriteError):
        await run(provider, spec)
    assert len(provider.bodies) == 2
    content = parts(provider.bodies[0])
    facts = next(p["text"] for p in content if p["type"] == "text" and p["text"].startswith("<Video 1>: "))
    assert "[Shot 2] 00:01.5" in facts or "[Shot 2] 00:01.4" in facts
    assert "video_url" not in [p["type"] for p in content]
    assert sum(p["type"] == "image_url" for p in content) >= 3  # reference picture + keyframes


def _png() -> bytes:
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(out, format="PNG")
    return out.getvalue()


def test_completion_accepts_suffixed_ids_of_the_requested_model_and_rejects_mismatch():
    from litellm.llms.causyn.h3_prompt import _Completion

    base = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}}
    ctx = {"requested": "qwen/qwen3.8-omni-flash"}
    assert _Completion.model_validate({**base, "model": "qwen/qwen3.8-omni-flash:20261001"}, context=ctx)
    with pytest.raises(ValidationError):
        _Completion.model_validate({**base, "model": "qwen/qwen3.8-flash"}, context=ctx)


@pytest.mark.asyncio
async def test_returned_model_mismatch_is_a_retryable_fixed_phrase_error(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(GOLD["R1"], model="qwen/qwen3.8-max-0902")  # requested flash
    with pytest.raises(RewriteError) as caught:
        await run(provider, spec)
    assert caught.value.status_code == 502 and caught.value.retryable is True
    assert str(caught.value) == "H3 prompt rewrite provider returned an invalid response"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b"RIFFxxxxWAVE", "wav"),
        (b"ID3\x04\x00", "mp3"),
        (b"\xff\xfb\x90\x00", "mp3"),  # MPEG-1 Layer III
        (b"\xff\xf3\x90\x00", "mp3"),  # MPEG-2 Layer III
        (b"\xff\xf1\x50\x80", None),  # AAC ADTS
        (b"\xff\xf9\x50\x80", None),  # AAC ADTS
        (b"\xff\xfd\x90\x00", None),  # Layer II
        (b"OggS", None),
    ],
)
def test_audio_format_detection(raw, expected):
    assert h3_prompt._audio_format(raw) == expected


def test_audio_over_the_budget_is_dropped_but_facts_remain():
    spec = ref_spec("image", "audio", duration=8)
    big = RefFacts(audios=(afacts(),), audio_raw=(b"RIFFxxxxWAVE" + b"0" * (12 * 1024 * 1024),))
    content = spec.user_content(big, "qwen/qwen3.8-omni-flash")
    assert "input_audio" not in [p["type"] for p in content]
    assert any(p.get("text", "").startswith("<Audio 1>: ") for p in content)
    small = RefFacts(audios=(afacts(),), audio_raw=(b"RIFFxxxxWAVE" + b"0" * 1000,))
    assert "input_audio" in [p["type"] for p in spec.user_content(small, "qwen/qwen3.8-omni-flash")]


# ----------------------------------------------------------------------------- cut / retention rules


def test_cut_rule_matches_lowercase_at_and_second_precision():
    spec, facts = case("R1")
    for variant in ("at 00:03.583, the", "at 00:03.700, the"):
        assert not cut_findings(mutate("R1", "At 00:03.583, the", variant), spec, facts), variant
    assert cut_findings(mutate("R1", "At 00:03.583, the", "at 00:05, the"), spec, facts)
    whole = RefFacts(videos=(vfacts(3.0),))  # cut at a whole second: MM:SS is enough
    assert not cut_findings(mutate("R1", "At 00:03.583, the", "At 00:03, the"), spec, whole)
    assert cut_findings(mutate("R1", "At 00:03.583, the", "At 00:04, the"), spec, whole)


def test_retention_allows_wrapped_lines_but_needs_every_label():
    spec, facts = case("R1")
    validate_prompt(mutate("R1", "are retained; the blade", "are retained;\nthe blade"), spec, facts)
    subject2 = next(line for line in GOLD["R1"].splitlines() if line.startswith("<Subject 2> (appears in"))
    dropped = GOLD["R1"].replace(subject2 + "\n", "")
    assert dropped != GOLD["R1"]
    assert any(v.startswith(h3_prompt._V_RETENTION) and "<Subject 2>" in v for v in violations(dropped, spec, facts))
    stray = GOLD["R1"].replace("retention_analysis:\n", "retention_analysis:\nstray text\n")
    assert any(v.startswith(h3_prompt._V_RETENTION) for v in violations(stray, spec, facts))


# ----------------------------------------------------------------------------- perception robustness


@pytest.mark.asyncio
async def test_perceive_degrades_on_corrupt_media_without_mocks():
    raw = h3_media.PreparedRaw(videos=(b"not a video",), audios=(b"not audio",))
    facts = await h3_prompt.perceive_media(ref_spec("image", "video", "audio"), raw)
    assert facts.videos[0].cuts is None and facts.videos[0].keyframes == ()
    assert facts.audios == (None,)
    assert facts.audio_raw == (b"not audio",)


@pytest.mark.asyncio
async def test_perceive_degrades_when_the_inner_analysis_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("decoder crash")

    monkeypatch.setattr(h3_prompt, "analyze_video", boom)
    raw = h3_media.PreparedRaw(videos=(b"x", b"y"), audios=(b"z",))
    facts = await h3_prompt.perceive_media(ref_spec("image", "video", "video", "audio"), raw)
    assert facts.videos == (None, None) and facts.audios == (None,)


@pytest.mark.asyncio
async def test_perceive_with_an_exhausted_budget_returns_facts_without_cuts(monkeypatch):
    monkeypatch.setattr(h3_prompt, "PERCEPTION_BUDGET_S", -1.0)
    raw = h3_media.PreparedRaw(videos=(two_shot_clip(),), audios=())
    facts = await h3_prompt.perceive_media(ref_spec("image", "video", duration=5), raw)
    assert facts.videos[0] is not None and facts.videos[0].cuts is None and facts.videos[0].keyframes == ()


@pytest.mark.asyncio
async def test_perception_is_memoised_across_retries(monkeypatch):
    calls = []
    real = h3_prompt.analyze_video

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(h3_prompt, "analyze_video", counting)
    spec = ref_spec("image", "video", duration=5)
    clip = two_shot_clip()
    first = await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(clip,), audios=()))
    second = await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(bytes(clip),), audios=()))
    assert first.videos == second.videos and len(calls) == 1
    other = await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"different",), audios=()))
    assert other is not first and len(calls) == 2


@pytest.mark.asyncio
async def test_memo_is_bounded_to_32_entries():
    spec = ref_spec("image", "video", duration=5)
    for n in range(40):
        await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"v%d" % n,), audios=()))
    assert len(h3_prompt._FACTS_MEMO) == 32


@pytest.mark.asyncio
async def test_retry_after_a_failed_repair_call_reuses_the_first_answer(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    calls: list[dict] = []
    plan = iter([OLD["R1"], 503, GOLD["R1"]])

    def handler(request):
        calls.append(json.loads(request.content))
        step = next(plan)
        if step == 503:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "model": "qwen/qwen3.8-flash",
                "choices": [{"message": {"content": step}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RewriteError) as caught:
            await H3PromptRewriter(client, "k").rewrite(spec)
        assert caught.value.retryable
        result = await H3PromptRewriter(client, "k").rewrite(spec)
    assert len(calls) == 3  # first, failed repair, retried repair: the first answer was not paid for twice
    assert result.usage.prompt_tokens == 10  # the first answer's 5 tokens are still reported on the retry
    assert calls[2]["messages"][2] == {"role": "assistant", "content": OLD["R1"].strip()}
    assert result.prompt == GOLD["R1"].strip()


@pytest.mark.asyncio
async def test_cached_facts_hold_no_audio_bytes_and_reattach_current_ones():
    spec = ref_spec("image", "video", "audio", duration=5)
    audio = b"RIFFxxxxWAVE" + b"0" * 5000
    first = await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"v",), audios=(audio,)))
    assert first.audio_raw == (audio,)
    assert all(entry.audio_raw == () for entry in h3_prompt._FACTS_MEMO.values())
    again = await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"v",), audios=(audio,)))
    assert again.audio_raw == (audio,)


@pytest.mark.asyncio
async def test_facts_memo_is_bounded_by_total_keyframe_bytes(monkeypatch):
    monkeypatch.setattr(h3_prompt, "FACTS_MEMO_MAX_BYTES", 2500)

    def fat(raw, **kwargs):
        return VideoFacts(5.0, 24.0, 64, 64, (2.0,), (Keyframe(0.15, 1, b"j" * 1000),))

    monkeypatch.setattr(h3_prompt, "analyze_video", fat)
    spec = ref_spec("image", "video", duration=5)
    for n in range(5):
        await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"v%d" % n,), audios=()))
    assert len(h3_prompt._FACTS_MEMO) == 2  # 3 x 1000 B would exceed 2500 B
    kept = {h3_prompt.media_key(spec, (b"v%d" % n,), ()) for n in (3, 4)}
    assert set(h3_prompt._FACTS_MEMO) == kept


def test_media_key_covers_text_duration_ratio_roles_and_fetched_bytes():
    base = ref_spec("image", "video", duration=5)
    key = h3_prompt.media_key(base, (b"v",), ())
    assert key == h3_prompt.media_key(ref_spec("image", "video", duration=5), (b"v",), ())
    assert key != h3_prompt.media_key(base, (b"w",), ())
    assert key != h3_prompt.media_key(ref_spec("image", "video", duration=6), (b"v",), ())
    assert key != h3_prompt.media_key(base.model_copy(update={"ratio": "9:16"}), (b"v",), ())
    changed = base.model_copy(update={"content": (base.content[0].model_copy(update={"text": "Other"}), *base.content[1:])})
    assert key != h3_prompt.media_key(changed, (b"v",), ())
    assert key != h3_prompt.media_key(ref_spec("video", "image", duration=5), (b"v",), ())
    # the video URL itself is not part of the key; only its fetched bytes are
    other_url = base.model_copy(update={"content": (*base.content[:2], base.content[2].model_copy(update={"video_url": {"url": "https://x.example/v.mp4"}}))})
    assert key == h3_prompt.media_key(other_url, (b"v",), ())


@pytest.mark.asyncio
async def test_permanent_repair_failure_leaves_no_cached_first_answer(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(OLD["R1"], OLD["R1"])
    with pytest.raises(RewriteError):
        await run(provider, spec)
    assert len(h3_prompt._FIRST_ANSWERS) == 0
    provider2 = Provider(GOLD["R1"])
    await run(provider2, spec)
    assert len(provider2.bodies) == 1  # the next identical request does not replay a stale answer


# ----------------------------------------------------------------------------- hard / soft classes


def soft_only_prompt() -> str:
    """Gold R1 with a too-short detailed_description: every hard rule holds, only the word range is broken."""
    head, rest = GOLD["R1"].split("detailed_description:\n", 1)
    _, tail = rest.split("\n\noverall_soundscape:", 1)
    return head + "detailed_description:\n[Shot 1] A runner sprints.\n[Shot 2] At 00:03.583, a low shot.\n\noverall_soundscape:" + tail


def classes(prompt, spec, facts):
    hard, soft = h3_prompt.ref2va_violations(prompt, spec, facts)
    return {v.split(": ", 1)[0] for v in hard}, {v.split(": ", 1)[0] for v in soft}


def test_hard_soft_matrix():
    spec, facts = case("R1")
    assert classes(soft_only_prompt(), spec, facts) == (set(), {h3_prompt._V_WORDS})
    hard, soft = classes(mutate("R1", "<Video 1> is the source", "<Video 2> is the source"), spec, facts)
    assert h3_prompt._V_LABELS in hard
    assert h3_prompt._V_SHOTS in classes(mutate("R1", "[Shot 2] At", "[Shot 3] At"), spec, facts)[0]
    assert h3_prompt._V_TIMES in classes(mutate("R1", "At 00:03.583", "At 00:05.000"), spec, facts)[0]
    assert h3_prompt._V_CUTS in classes(OLD["R1"], spec, facts)[0]
    # a ref2va answer without [Shot 1] is hard
    assert h3_prompt._V_SHOTS in classes(GOLD["R1"].replace("[Shot 1]", "Shot one").replace("[Shot 2]", "Shot two"), spec, facts)[0]
    # undefined provided label, retention gaps and word range are soft only
    spec4, facts4 = case("R4")
    start = GOLD["R4"].index("<Audio 1> is a complete")
    end = GOLD["R4"].index("\n", start)
    assert classes(GOLD["R4"][:start] + GOLD["R4"][end + 1 :], spec4, facts4)[0] == set()
    subject2 = next(l for l in GOLD["R1"].splitlines() if l.startswith("<Subject 2> (appears in"))
    hard, soft = classes(GOLD["R1"].replace(subject2 + "\n", ""), spec, facts)
    assert hard == set() and h3_prompt._V_RETENTION in soft


def test_validate_with_a_soft_sink_accepts_soft_only_and_raises_hard():
    spec, facts = case("R1")
    sink: list[str] = []
    validate_prompt(soft_only_prompt(), spec, facts, soft=sink)
    assert sink and sink[0].startswith(h3_prompt._V_WORDS)
    with pytest.raises(RewriteError):
        validate_prompt(OLD["R1"], spec, facts, soft=[])


@pytest.mark.asyncio
async def test_soft_only_first_answer_is_repaired_and_a_clean_repair_wins(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(soft_only_prompt(), GOLD["R1"])
    result = await run(provider, spec)
    assert len(provider.bodies) == 2 and result.prompt == GOLD["R1"].strip() and result.soft_violations == ()


@pytest.mark.asyncio
async def test_soft_only_after_repair_is_accepted_and_recorded(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(soft_only_prompt(), soft_only_prompt())
    result = await run(provider, spec)
    assert len(provider.bodies) == 2
    assert result.prompt == soft_only_prompt().strip()
    assert result.soft_violations and result.soft_violations[0].startswith(h3_prompt._V_WORDS)


@pytest.mark.asyncio
async def test_hard_violation_after_repair_raises(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    provider = Provider(soft_only_prompt(), OLD["R1"])
    # first had only soft findings: the worse repair is dropped, the first answer returned
    result = await run(provider, spec)
    assert result.prompt == soft_only_prompt().strip() and result.soft_violations
    h3_prompt._FIRST_ANSWERS.clear()
    provider = Provider(OLD["R1"], OLD["R1"])
    with pytest.raises(RewriteError) as caught:
        await run(provider, spec)
    assert str(caught.value) == CUT_MSG


@pytest.mark.asyncio
async def test_failed_repair_call_keeps_a_soft_only_first_answer(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    answers = iter([soft_only_prompt()])

    def handler(request):
        try:
            text = next(answers)
        except StopIteration:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "model": "qwen/qwen3.8-flash",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await H3PromptRewriter(client, "k").rewrite(spec)
    assert result.prompt == soft_only_prompt().strip()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,expected",
    [
        ({"model": "qwen/qwen3.8-flash", "choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}},
         "finish_reason=length model=qwen/qwen3.8-flash content_empty=True"),
        ({"model": "other/model", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}},
         "finish_reason=stop model=other/model content_empty=False"),
    ],
)
async def test_invalid_response_carries_a_redacted_detail_not_a_public_message(patched, reply, expected):
    spec, facts = case("R1")
    patched(spec, facts)

    def handler(request):
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RewriteError) as caught:
            await H3PromptRewriter(client, "k").rewrite(spec)
    assert str(caught.value) == "H3 prompt rewrite provider returned an invalid response"
    assert caught.value.detail == expected
    assert expected.split()[0] not in str(caught.value)


@pytest.mark.asyncio
async def test_unparseable_reply_has_a_detail_too(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"{broken"))) as client:
        with pytest.raises(RewriteError) as caught:
            await H3PromptRewriter(client, "k").rewrite(spec)
    assert caught.value.detail == "unreadable reply, 7 bytes"


# ----------------------------------------------------------------------------- truncated answers


def scripted(plan):
    """Handler replaying `plan`: a str is a stop answer, 'length' a truncated one (looping text)."""
    bodies: list[dict] = []
    steps = iter(plan)

    def handler(request):
        bodies.append(json.loads(request.content))
        step = next(steps)
        if step == "length":
            message = {"content": "The runner sprints. " * 50}
            finish = "length"
        else:
            message, finish = {"content": step}, "stop"
        return httpx.Response(
            200,
            json={
                "model": "qwen/qwen3.8-flash",
                "choices": [{"message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

    return handler, bodies


@pytest.mark.asyncio
async def test_truncated_first_answer_is_repaired_once_without_echoing_it(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    handler, bodies = scripted(["length", GOLD["R1"]])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await H3PromptRewriter(client, "k").rewrite(spec)
    assert result.prompt == GOLD["R1"].strip()
    assert len(bodies) == 2
    messages = bodies[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert messages[2]["content"] == h3_prompt.TRUNCATION_REPAIR_MESSAGE
    assert "The runner sprints" not in json.dumps(messages)
    assert "350–500 words" in messages[2]["content"] and "do not repeat sentences" in messages[2]["content"]


@pytest.mark.asyncio
async def test_truncated_twice_raises_the_permanent_error(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    handler, bodies = scripted(["length", "length", GOLD["R1"]])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RewriteError) as caught:
            await H3PromptRewriter(client, "k").rewrite(spec)
    assert len(bodies) == 2
    assert str(caught.value) == "H3 prompt rewrite provider returned an invalid response"
    assert caught.value.status_code == 502 and caught.value.retryable is False
    assert "finish_reason=length" in caught.value.detail


@pytest.mark.asyncio
async def test_non_ref2va_truncation_is_still_permanent_without_repair():
    spec = ContextIRRequest.model_validate(
        {"model": "MiniMax-H3", "content": [{"type": "text", "text": "A cat."}], "duration": 5, "ratio": "16:9"}
    )
    handler, bodies = scripted(["length", "x"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RewriteError) as caught:
            await H3PromptRewriter(client, "k").rewrite(spec)
    assert len(bodies) == 1 and caught.value.retryable is False
    assert bodies[0]["max_tokens"] == 4096 and bodies[0]["reasoning"] == {"enabled": False}


# ----------------------------------------------------------------------------- final-review fixes


@pytest.mark.asyncio
async def test_missing_imaging_library_logs_one_warning_per_process(monkeypatch, caplog):
    monkeypatch.setattr(h3_prompt, "_warned_import", False)
    monkeypatch.setitem(__import__("sys").modules, "numpy", None)  # `import numpy` now raises ImportError
    spec = ref_spec("image", "video", duration=5)
    with caplog.at_level("WARNING"):
        for n in range(3):
            facts = await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"v%d" % n,), audios=()))
            assert facts.videos == (None,)
    assert sum("imaging library is missing" in r.message for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_empty_perception_of_a_fetched_video_logs_a_warning(caplog):
    spec = ref_spec("image", "video", duration=5)
    with caplog.at_level("WARNING"):
        await h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"corrupt",), audios=()))
    assert any("no shot cuts and no keyframes" in r.message for r in caplog.records)
    assert all("corrupt" not in r.message and "http" not in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_perception_concurrency_is_capped_at_two(monkeypatch):
    import asyncio

    running = peak = 0

    async def slow(videos, audios):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return RefFacts(videos=(None,) * len(videos))

    monkeypatch.setattr(h3_prompt, "_perceive", slow)
    spec = ref_spec("image", "video", duration=5)
    await asyncio.gather(
        *[h3_prompt.perceive_media(spec, h3_media.PreparedRaw(videos=(b"c%d" % n,), audios=())) for n in range(6)]
    )
    assert peak == 2


@pytest.mark.asyncio
async def test_cancelled_attempt_still_fills_the_memo(monkeypatch):
    import asyncio

    async def slow(videos, audios):
        await asyncio.sleep(0.1)
        return RefFacts(videos=(vfacts(2.0),))

    monkeypatch.setattr(h3_prompt, "_perceive", slow)
    spec = ref_spec("image", "video", duration=5)
    raw = h3_media.PreparedRaw(videos=(b"late",), audios=())
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02):
            await h3_prompt.perceive_media(spec, raw)
    await asyncio.sleep(0.2)
    assert len(h3_prompt._FACTS_MEMO) == 1


@pytest.mark.asyncio
async def test_ref2va_does_not_build_a_video_data_url_unless_send_video(monkeypatch):
    import asyncio

    clip = base64.b64encode(two_shot_clip()).decode()
    original = "data:video/quicktime;base64," + clip  # a rebuilt URL would say video/mp4
    spec = ref_spec("image", "video", duration=5)
    image = spec.content[1].model_copy(update={"image_url": MediaURL(url="data:image/png;base64," + base64.b64encode(_png()).decode())})
    video = spec.content[2].model_copy(update={"video_url": MediaURL(url=original)})
    spec = spec.model_copy(update={"content": (spec.content[0], image, video)})
    async with httpx.AsyncClient() as client:
        prepared, raw = await h3_media.prepare_media_with_raw(client, spec)
        assert prepared.content[2].video_url.url == original and raw.videos[0]
        monkeypatch.setenv("CAUSYN_H3_REWRITE_SEND_VIDEO", "1")
        prepared, _ = await h3_media.prepare_media_with_raw(client, spec)
        assert prepared.content[2].video_url.url.startswith("data:video/mp4;base64,")


def test_timestamp_rule_is_hard_only_with_a_reference_video():
    bad = "[Shot 1] A. At 00:04.000 x At 00:02.000 y"
    desc = " ".join(["She walks along the quiet street while the camera follows."] * 25)
    prompt = (
        "subject_definitions:\n<Subject 1> is the woman from <Picture 1>.\n\n"
        "summary:\n[reference generation] A woman walks.\n\n"
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): kept from <Picture 1>.\n\n"
        f"detailed_description:\n[Shot 1] {desc} At 00:04.000 then At 00:02.000.\n\n"
        "overall_soundscape:\nSteps.\n\nnon_diegetic_music:\nN/A"
    )
    image_only = ref_spec("image", duration=8)
    hard, soft = h3_prompt.ref2va_violations(prompt, image_only, None)
    assert hard == [] and any(v.startswith(h3_prompt._V_TIMES) for v in soft)
    validate_prompt(prompt, image_only, None, soft=[])
    with_video = ref_spec("image", "video", duration=8)
    hard, _ = h3_prompt.ref2va_violations(prompt, with_video, RefFacts(videos=(vfacts(None),)))
    assert any(v.startswith(h3_prompt._V_TIMES) for v in hard)


@pytest.mark.asyncio
async def test_usage_of_truncated_and_reused_first_answers_is_reported(patched):
    spec, facts = case("R1")
    patched(spec, facts)
    handler, bodies = scripted(["length", GOLD["R1"]])  # each call reports 7 prompt / 3 completion tokens
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await H3PromptRewriter(client, "k").rewrite(spec)
    assert result.usage.prompt_tokens == 14 and result.usage.completion_tokens == 6
