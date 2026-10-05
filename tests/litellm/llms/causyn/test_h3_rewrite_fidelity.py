from __future__ import annotations

import json
import time

import httpx
import pytest

from litellm.llms.causyn import h3_media, h3_prompt
from litellm.llms.causyn.h3_prompt import (
    CRITIC_INSTRUCTION,
    ContextIRRequest,
    H3PromptRewriter,
    RewriteError,
    literal_violations,
)

IMG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/"
AUD = "https://media.example/a.wav"
TAIL = "overall_soundscape: Quiet room tone.\nnon_diegetic_music: None."


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    h3_prompt._FACTS_MEMO.clear()
    h3_prompt._FIRST_ANSWERS.clear()
    h3_prompt._warned_models.clear()
    h3_prompt._PLAN_NOTES.clear()
    monkeypatch.setenv("CAUSYN_H3_REF2VA_PLAN", "0")  # covered in test_h3_ref2va_plan.py; would add provider calls here
    for name in ("CAUSYN_H3_REWRITE_CRITIC", "CAUSYN_H3_REWRITE_CRITIC_MODEL", "CAUSYN_H3_REF2VA_REWRITE_MODEL"):
        monkeypatch.delenv(name, raising=False)

    async def prepare(client, request):
        return request, h3_media.PreparedRaw(videos=(), audios=())

    monkeypatch.setattr(h3_media, "prepare_media_with_raw", prepare)


def make_spec(text: str, *roles: str, duration: int = 5, audio: bool = False) -> ContextIRRequest:
    content: list[dict] = [{"type": "text", "text": text}]
    for role in roles:
        content.append({"type": "image_url", "image_url": {"url": IMG}, "role": role})
    if audio:
        content.append({"type": "audio_url", "audio_url": {"url": AUD}, "role": "reference_audio"})
    model = "causyn-1.1" if audio or any(r.startswith("reference_") for r in roles) else "MiniMax-H3"
    return ContextIRRequest.model_validate({"model": model, "content": content, "duration": duration, "ratio": "16:9"})


def t2va_prompt(desc: str) -> str:
    return f"integrated_multimodal_description: {desc}\n{TAIL}"


def is_critic(body: dict) -> bool:
    content = body["messages"][0]["content"]
    return isinstance(content, list) and content[0].get("text") == CRITIC_INSTRUCTION


class Provider:
    """Rewrite answers in order; critic calls are answered separately and recorded in `critic_bodies`."""

    def __init__(self, *answers: str, critic=None, finish: list[str] | None = None, usage: int = 100) -> None:
        self.answers, self.critic, self.finish, self.usage = list(answers), critic, finish or [], usage
        self.bodies: list[dict] = []
        self.critic_bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if is_critic(body):
            self.critic_bodies.append(body)
            if isinstance(self.critic, Exception):
                raise self.critic
            if isinstance(self.critic, httpx.Response):
                return self.critic
            text = json.dumps({"defects": self.critic or []}) if not isinstance(self.critic, str) else self.critic
            return reply(body["model"], text, 7)
        self.bodies.append(body)
        index = len(self.bodies) - 1
        finish = self.finish[index] if index < len(self.finish) else "stop"
        return reply(body["model"], self.answers[index], self.usage, finish)


def reply(model: str, text: str, tokens: int, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": tokens, "completion_tokens": tokens, "total_tokens": 2 * tokens, "cost": 0.001},
        },
    )


async def run(provider: Provider, spec: ContextIRRequest):
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        return await H3PromptRewriter(client, "key").rewrite(spec)


QUOTE_SPEC = lambda: make_spec('A man says "I bet there is no bullet in your gun."')  # noqa: E731
GOOD_QUOTE = t2va_prompt("[Shot 1] A man says: <d>[English] I bet there is no bullet in your gun.</d>")
BAD_QUOTE = t2va_prompt("[Shot 1] A man says: <d>[English] I bet you won't pull the trigger.</d>")


# 1
@pytest.mark.asyncio
async def test_dropped_quoted_line_triggers_a_repair_naming_the_span():
    provider = Provider(BAD_QUOTE, GOOD_QUOTE)
    result = await run(provider, QUOTE_SPEC())
    assert result.prompt == GOOD_QUOTE
    assert len(provider.bodies) == 2
    last = provider.bodies[1]["messages"][-1]["content"]
    assert "I bet there is no bullet in your gun" in last
    assert "must keep the user's quoted text verbatim" in last
    assert provider.bodies[1]["messages"][-2] == {"role": "assistant", "content": BAD_QUOTE}


# 2
@pytest.mark.asyncio
async def test_invented_dialogue_is_repaired_but_allowed_with_reference_audio():
    spec = make_spec("A man waits by a door.")
    invented = t2va_prompt("[Shot 1] A man waits: <d>[English] Where are you?</d>")
    clean = t2va_prompt("[Shot 1] A man waits by a door in silence.")
    provider = Provider(invented, clean)
    result = await run(provider, spec)
    assert result.prompt == clean and len(provider.bodies) == 2
    assert "must not add dialogue the user did not write" in provider.bodies[1]["messages"][-1]["content"]
    assert "Where are you" in provider.bodies[1]["messages"][-1]["content"]
    audio = make_spec("A man waits by a door.", "reference_image", audio=True)
    assert literal_violations(invented, audio) == []
    assert literal_violations(invented, spec) != []


# 3
@pytest.mark.asyncio
async def test_invented_visible_text_is_repaired_for_t2va_only():
    spec = make_spec("A microwave on top of an oven.")
    invented = t2va_prompt('[Shot 1] A microwave on an oven, its display reads "00:30".')
    clean = t2va_prompt("[Shot 1] A microwave on an oven.")
    provider = Provider(invented, clean)
    result = await run(provider, spec)
    assert result.prompt == clean and len(provider.bodies) == 2
    assert "must not add visible text" in provider.bodies[1]["messages"][-1]["content"]
    i2va = make_spec("A microwave on top of an oven.", "first_frame")
    assert not any("visible text" in v for v in literal_violations(invented, i2va))


# 4
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second",
    ["[Shot 2] At 00:06.000, cut to a wide shot.", "[Shot 2] The camera cuts to a wide shot."],
)
async def test_base_mode_later_shot_timestamps_are_hard_and_repaired(second):
    bad = t2va_prompt(f"[Shot 1] A cat walks. {second}")
    good = t2va_prompt("[Shot 1] A cat walks. [Shot 2] At 00:03.000, cut to a wide shot.")
    provider = Provider(bad, good)
    result = await run(provider, make_spec("A cat walks.", duration=5))
    assert result.prompt == good and len(provider.bodies) == 2
    assert "invalid shot timestamps" in provider.bodies[1]["messages"][-1]["content"]


# 5
@pytest.mark.asyncio
async def test_fl2va_alignment_must_end_on_the_final_shot():
    head = "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {n}) aligns with the 5.00-second mark of the target video.\n"
    body = t2va_prompt("[Shot 1] A cat walks. [Shot 2] At 00:03.000, it stops.")
    bad, good = head.format(n=1) + body, head.format(n=2) + body
    provider = Provider(bad, good)
    result = await run(provider, make_spec("A cat walks.", "first_frame", "last_frame"))
    assert result.prompt == good and len(provider.bodies) == 2
    assert "invalid last-frame alignment" in provider.bodies[1]["messages"][-1]["content"]


# 6
@pytest.mark.asyncio
async def test_critic_defects_drive_the_repair_and_usage_is_summed():
    spec = make_spec("An orange vase on a table.")
    first = t2va_prompt("[Shot 1] A red rose in a glass vase.")
    fixed = t2va_prompt("[Shot 1] An orange vase on a table.")
    provider = Provider(first, fixed, critic=["The vase is not orange"])
    result = await run(provider, spec)
    assert result.prompt == fixed
    assert "Fidelity: The vase is not orange" in provider.bodies[1]["messages"][-1]["content"]
    assert len(provider.critic_bodies) == 1
    # two rewrite calls (100 each side) + critic (7 each side)
    assert result.usage.prompt_tokens == 207 and result.usage.completion_tokens == 207
    assert result.usage.total_tokens == 2 * 207
    critic = provider.critic_bodies[0]
    assert critic["model"] == "qwen/qwen3.8-omni-flash" and critic["max_tokens"] == 1500 and critic["temperature"] == 0
    texts = [p.get("text", "") for p in critic["messages"][0]["content"]]
    assert texts[0] == CRITIC_INSTRUCTION
    assert texts[-1] == f"USER REQUEST:\n{spec.prompt}\n\nREWRITE:\n{first}"


# 7
@pytest.mark.asyncio
async def test_clean_critic_means_two_calls_and_the_first_answer():
    good = t2va_prompt("[Shot 1] An orange vase on a table.")
    provider = Provider(good, critic=[])
    result = await run(provider, make_spec("An orange vase on a table."))
    assert result.prompt == good
    assert len(provider.bodies) == 1 and len(provider.critic_bodies) == 1
    assert result.usage.prompt_tokens == 107


# 8
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "critic",
    [
        httpx.Response(500, json={"error": {"message": "boom"}}),
        "not json at all",
        httpx.ConnectError("down"),
        '{"defects": "x"}',
    ],
)
async def test_critic_failures_are_fail_open(critic):
    good = t2va_prompt("[Shot 1] An orange vase on a table.")
    provider = Provider(good, critic=critic)
    result = await run(provider, make_spec("An orange vase on a table."))
    assert result.prompt == good and len(provider.bodies) == 1


# 9
@pytest.mark.asyncio
async def test_critic_can_be_disabled_and_unknown_model_skips_it(monkeypatch):
    good = t2va_prompt("[Shot 1] An orange vase on a table.")
    spec = make_spec("An orange vase on a table.")
    monkeypatch.setenv("CAUSYN_H3_REWRITE_CRITIC", "0")
    provider = Provider(good)
    assert (await run(provider, spec)).prompt == good
    assert provider.critic_bodies == []
    monkeypatch.delenv("CAUSYN_H3_REWRITE_CRITIC")
    monkeypatch.setenv("CAUSYN_H3_REWRITE_CRITIC_MODEL", "evil/model")
    provider = Provider(good)
    assert (await run(provider, spec)).prompt == good
    assert provider.critic_bodies == []


# 10
@pytest.mark.asyncio
async def test_no_critic_when_the_request_has_a_reference_video(monkeypatch):
    from tests.litellm.llms.causyn import test_h3_ref2va_rewrite as ref

    spec, facts = ref.case("R1")

    async def perceive(request, raw, key=None):
        return facts

    async def prepare(client, request):
        return request, h3_media.PreparedRaw(videos=(b"v",), audios=())

    monkeypatch.setattr(h3_media, "prepare_media_with_raw", prepare)
    monkeypatch.setattr(h3_prompt, "perceive_media", perceive)
    provider = Provider(ref.GOLD["R1"])
    result = await run(provider, spec)
    assert result.prompt == ref.GOLD["R1"].strip()
    assert provider.critic_bodies == [] and len(provider.bodies) == 1


# 11
@pytest.mark.asyncio
async def test_non_ref2va_truncation_goes_through_the_repair_path():
    good = t2va_prompt("[Shot 1] A cat walks.")
    provider = Provider("cut off", good, finish=["length"], critic=[])
    result = await run(provider, make_spec("A cat walks."))
    assert result.prompt == good
    assert provider.bodies[0]["max_tokens"] == 3072 and provider.bodies[0]["model"] == "qwen/qwen3.8-flash"
    assert provider.bodies[0]["reasoning"] == {"enabled": False}
    assert provider.bodies[1]["messages"][-1]["content"] == h3_prompt.TRUNCATION_REPAIR_MESSAGE_BASE


# 12
@pytest.mark.asyncio
async def test_failed_repair_call_keeps_a_soft_only_first_answer():
    class Flaky(Provider):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if not is_critic(body) and self.bodies:
                self.bodies.append(body)
                return httpx.Response(503)
            return super().__call__(request)

    missing = t2va_prompt("[Shot 1] A man stands still.")
    provider = Flaky(missing, critic=[])
    result = await run(provider, QUOTE_SPEC())
    assert result.prompt == missing and len(provider.bodies) == 2
    assert any("quoted text" in v for v in result.soft_violations)
    assert h3_prompt._FIRST_ANSWERS == {}


# 13
@pytest.mark.asyncio
async def test_hard_failure_surviving_the_repair_raises_non_retryable():
    bad = "integrated_multimodal_description: [Shot 1] A cat.\noverall_soundscape: x"  # an older check fails
    provider = Provider(bad, bad)
    with pytest.raises(RewriteError) as caught:
        await run(provider, make_spec("A cat walks.", duration=5))
    assert len(provider.bodies) == 2 and caught.value.retryable is False
    assert provider.critic_bodies == []


# extras
@pytest.mark.asyncio
async def test_critic_is_skipped_without_budget():
    good = t2va_prompt("[Shot 1] A cat walks.")
    token = h3_prompt.ATTEMPT_DEADLINE.set(time.monotonic() + 1.0)
    try:
        provider = Provider(good)
        result = await run(provider, make_spec("A cat walks."))
    finally:
        h3_prompt.ATTEMPT_DEADLINE.reset(token)
    assert result.prompt == good and provider.critic_bodies == []


@pytest.mark.asyncio
async def test_critic_gets_images_and_an_audio_note_but_never_audio_bytes():
    spec = make_spec("A girl sings.", "reference_image", audio=True)
    provider = Provider("x", critic=[])
    h3_prompt._warned_models.clear()
    rewriter = H3PromptRewriter(httpx.AsyncClient(transport=httpx.MockTransport(provider)), "k")
    defects, usage = await rewriter.fidelity_defects(spec, "p", 30.0)
    assert defects == () and usage.prompt_tokens == 7
    parts = provider.critic_bodies[0]["messages"][0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)
    assert any(p.get("text") == h3_prompt.CRITIC_AUDIO_NOTE for p in parts)
    assert "input_audio" not in json.dumps(parts) and "audio_url" not in json.dumps(parts)


def test_literal_checks_ignore_one_trailing_punctuation_and_whitespace():
    spec = make_spec('He says "Hello,   world!" and "你好。"')
    ok = t2va_prompt("[Shot 1] <d>[English] Hello, world</d> <d>[Chinese] 你好</d>")
    assert literal_violations(ok, spec) == []


def test_parse_defects_limits_and_shapes():
    many = json.dumps({"defects": ["x" * 400] + [f"d{i}" for i in range(20)]})
    parsed = h3_prompt.parse_defects("sure ```json\n" + many + "\n```")
    assert parsed is not None and len(parsed) == 8 and len(parsed[0]) == 300
    assert h3_prompt.parse_defects('{"defects": [1]}') is None
    assert h3_prompt.parse_defects("nothing") is None


# ----------------------------------------------------------------------------- fix round 1


@pytest.mark.asyncio
async def test_critic_never_fails_the_rewrite_on_unexpected_errors(monkeypatch):
    good = t2va_prompt("[Shot 1] An orange vase on a table.")
    spec = make_spec("An orange vase on a table.")
    provider = Provider(good, critic=httpx.DecodingError("bad gzip"))
    assert (await run(provider, spec)).prompt == good and len(provider.bodies) == 1

    async def boom(self, *args, **kwargs):
        raise RuntimeError("secret-token-123 https://x.example")

    monkeypatch.setattr(H3PromptRewriter, "_complete", H3PromptRewriter._complete)
    provider = Provider(good)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        rewriter = H3PromptRewriter(client, "k")
        real = H3PromptRewriter._complete

        async def selective(self, model, messages, budget, ref2va=False, settings=None):
            if settings is not None:
                raise RuntimeError("secret-token-123 https://x.example")
            return await real(self, model, messages, budget, ref2va, settings)

        monkeypatch.setattr(H3PromptRewriter, "_complete", selective)
        result = await rewriter.rewrite(spec)
    assert result.prompt == good and len(provider.bodies) == 1


@pytest.mark.parametrize(
    "user,rewrite",
    [
        ("He says “I don’t know”", "<d>[English] I don't know</d>"),
        ('He says "I don\'t know"', "<d>[English] I don’t know</d>"),
        ("a man says hello to her", "<d>[English] Hello</d>"),
        ("她说“你好，世界”", "<d>[Chinese] 你好,世界</d>"),
        ('He says "wait..."', "<d>[English] wait…</d>"),
    ],
)
def test_comparison_is_normalised_on_both_sides(user, rewrite):
    assert literal_violations(t2va_prompt("[Shot 1] " + rewrite), make_spec(user)) == []


def test_inch_marks_and_unbalanced_quotes_trigger_nothing():
    plain = t2va_prompt("[Shot 1] A 12 inch pizza and a 14 inch pan.")
    assert literal_violations(plain, make_spec('a 12" pizza and a 14" pan')) == []
    assert literal_violations(plain, make_spec('he says "hello there')) == []
    odd = t2va_prompt('[Shot 1] A sign reading "OPEN" and "CLOSED" hangs.')
    found = literal_violations(odd, make_spec("A sign hangs on a door."))
    assert len(found) == 1 and "OPEN" in found[0]
    unbalanced = t2va_prompt('[Shot 1] A 5" nail and a "sign.')
    assert literal_violations(unbalanced, make_spec("A nail.")) == []


def test_pasted_caption_in_dialogue_is_flagged_when_the_user_marked_lines():
    spec = make_spec('A cop says "Drop it now." The street is empty at night.')
    ok = t2va_prompt("[Shot 1] <d>[English] Drop it now.</d>")
    assert literal_violations(ok, spec) == []
    caption = t2va_prompt(
        "[Shot 1] <d>[English] A cop says Drop it now. The street is empty at night.</d> Drop it now."
    )
    found = literal_violations(caption, spec)
    assert any("must not add dialogue" in v for v in found)
    short = t2va_prompt("[Shot 1] <d>[English] The street is empty</d> Drop it now.")
    assert literal_violations(short, spec) == []  # short phrase from the prompt is tolerated
    # without quotes plain containment applies
    assert (
        literal_violations(t2va_prompt("[Shot 1] <d>[English] A cop says drop it</d>"), make_spec("A cop says drop it"))
        == []
    )


def test_language_tag_is_stripped_from_every_piece_and_markers_split():
    spec = make_spec("He says hello and goodbye.")
    prompt = t2va_prompt("[Shot 1] <d>[English] hello<cutoff>[English] goodbye</d>")
    assert literal_violations(prompt, spec) == []
    prompt = t2va_prompt("[Shot 1] <d>[English] hello<scenetrans>[English] never said</d>")
    found = literal_violations(prompt, spec)
    assert len(found) == 1 and "never said" in found[0] and "[english]" not in found[0].lower()


@pytest.mark.asyncio
async def test_fl2va_alignment_without_a_from_shot_marker_is_accepted():
    head = "How the reference pictures align with the target video — the last picture aligns with the 5.00-second mark of the target video.\n"
    prompt = head + t2va_prompt("[Shot 1] A cat walks. [Shot 2] At 00:03.000, it stops.")
    provider = Provider(prompt, critic=[])
    result = await run(provider, make_spec("A cat walks.", "first_frame", "last_frame"))
    assert result.prompt == prompt and len(provider.bodies) == 1


@pytest.mark.asyncio
async def test_retryable_repair_failure_after_hard_failure_reuses_the_first_answer():
    bad = t2va_prompt("[Shot 1] A cat. [Shot 2] At 00:09.000, cut.")
    good = t2va_prompt("[Shot 1] A cat walks. [Shot 2] At 00:03.000, cut.")
    spec = make_spec("A cat walks.", duration=5)
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return reply(body["model"], bad, 10)
        if len(calls) == 2:
            return httpx.Response(503)
        return reply(body["model"], good, 10)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RewriteError) as caught:
            await H3PromptRewriter(client, "k").rewrite(spec)
        assert caught.value.retryable is True and len(calls) == 2
        result = await H3PromptRewriter(client, "k").rewrite(spec)
    assert result.prompt == good and len(calls) == 3  # exactly one provider call on the retry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,reasoning,max_tokens",
    [
        ("qwen/qwen3.8-omni-flash", {"enabled": False}, 3072),
        ("qwen/qwen3.8-max-0902", {"enabled": True, "effort": "low"}, 12000),
    ],
)
async def test_base_rewrite_model_is_configurable(monkeypatch, model, reasoning, max_tokens):
    monkeypatch.setenv("CAUSYN_H3_REWRITE_MODEL", model)
    monkeypatch.setenv("CAUSYN_H3_REWRITE_CRITIC", "0")
    provider = Provider(t2va_prompt("[Shot 1] A cat walks."))
    result = await run(provider, make_spec("A cat walks."))
    body = provider.bodies[0]
    assert body["model"] == model and body["reasoning"] == reasoning and body["max_tokens"] == max_tokens
    assert result.model == model


@pytest.mark.asyncio
async def test_unknown_base_rewrite_model_fails_before_any_http_call(monkeypatch):
    monkeypatch.setenv("CAUSYN_H3_REWRITE_MODEL", "typo/model")
    called = []

    async def prepare(client, request):
        called.append("media")
        raise AssertionError("media must not be fetched")

    monkeypatch.setattr(h3_media, "prepare_media_with_raw", prepare)
    provider = Provider("x")
    with pytest.raises(RewriteError) as caught:
        await run(provider, make_spec("A cat walks."))
    assert caught.value.status_code == 503 and caught.value.retryable is False
    assert provider.bodies == [] and called == []


# ----------------------------------------------------------------------------- final fix round


@pytest.mark.asyncio
async def test_critic_sends_the_prepared_images_never_the_original_urls(monkeypatch):
    original = "https://private.example/first.png?sig=SECRET"
    prepared_url = "data:image/jpeg;base64,PREPARED"
    spec = ContextIRRequest.model_validate(
        {
            "model": "MiniMax-H3",
            "content": [
                {"type": "text", "text": "A cat walks."},
                {"type": "image_url", "image_url": {"url": original}, "role": "first_frame"},
            ],
            "duration": 5,
            "ratio": "16:9",
        }
    )

    async def prepare(client, request):
        swapped = ContextIRRequest.model_validate(json.loads(request.model_dump_json().replace(original, prepared_url)))
        return swapped, h3_media.PreparedRaw(videos=(), audios=())

    monkeypatch.setattr(h3_media, "prepare_media_with_raw", prepare)
    good = (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
        + t2va_prompt("[Shot 1] A cat walks.")
    )
    provider = Provider(good, critic=[])
    await run(provider, spec)
    rewrite_urls = [
        p["image_url"]["url"] for p in provider.bodies[0]["messages"][1]["content"] if p["type"] == "image_url"
    ]
    critic_urls = [
        p["image_url"]["url"] for p in provider.critic_bodies[0]["messages"][0]["content"] if p["type"] == "image_url"
    ]
    assert critic_urls == rewrite_urls == [prepared_url]
    assert "private.example" not in json.dumps(provider.critic_bodies[0]) and "SECRET" not in json.dumps(
        provider.critic_bodies[0]
    )


BAD_TIMES = t2va_prompt("[Shot 1] A cat. [Shot 2] At 00:09.000, cut.")


@pytest.mark.asyncio
async def test_repair_failing_only_new_checks_keeps_the_first_answer():
    provider = Provider(BAD_TIMES, BAD_TIMES)
    result = await run(provider, make_spec("A cat walks.", duration=5))
    assert result.prompt == BAD_TIMES and len(provider.bodies) == 2
    assert any("invalid shot timestamps" in v for v in result.soft_violations)


@pytest.mark.asyncio
async def test_repair_failing_an_older_check_still_raises():
    broken = "integrated_multimodal_description: [Shot 1] A cat walks.\noverall_soundscape: x"  # missing field
    provider = Provider(BAD_TIMES, broken)
    with pytest.raises(RewriteError):
        await run(provider, make_spec("A cat walks.", duration=5))
    # an older-check failure in the first answer is never excused, even if the repair has a new-check failure
    provider = Provider(broken, BAD_TIMES)
    with pytest.raises(RewriteError):
        await run(provider, make_spec("A cat walks.", duration=5))


# ----------------------------------------------------------------------------- never-worse repair


def ref_prompt(repeats: int) -> str:
    desc = " ".join(["She walks along the quiet street while the camera slowly follows her pace."] * repeats)
    return (
        "subject_definitions:\n<Subject 1> is the woman from <Picture 1> in the coat of <Picture 2>.\n\n"
        "summary:\n[reference generation] A woman walks.\n\n"
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - face from <Picture 1>\n"
        "and the coat of <Picture 2>.\n"
        "<Picture 2>: fully_preserved - the coat.\n\n"
        f"detailed_description:\nA calm look.\n[Shot 1] {desc}\n\n"
        "overall_soundscape:\nSoft footsteps.\n\nnon_diegetic_music:\nN/A"
    )


def ref_image_spec() -> ContextIRRequest:
    return make_spec("A woman walks.", "reference_image", "reference_image", duration=8)


@pytest.mark.asyncio
async def test_a_repair_that_adds_deterministic_findings_loses_to_the_first_answer():
    first = ref_prompt(2)  # one finding: below the 200-word soft minimum
    worse = first.replace("[Shot 1] ", "[Shot 1] <d>[English] hello there</d> ")  # plus invented dialogue
    provider = Provider(first, worse)
    result = await run(provider, ref_image_spec())
    assert result.prompt == first
    assert len(result.soft_violations) == 1 and "200 to 750 words" in result.soft_violations[0]
    assert result.usage.prompt_tokens == 200  # both calls are still paid


@pytest.mark.asyncio
async def test_a_repair_with_equal_or_fewer_deterministic_findings_wins():
    provider = Provider(ref_prompt(2), ref_prompt(20))
    result = await run(provider, ref_image_spec())
    assert result.prompt == ref_prompt(20) and result.soft_violations == ()
    # equal count (one finding each) is not "strictly more": the repair is kept
    provider = Provider(ref_prompt(2), ref_prompt(3))
    result = await run(provider, ref_image_spec())
    assert result.prompt == ref_prompt(3)


@pytest.mark.asyncio
async def test_ref2va_never_calls_the_critic_but_base_modes_do():
    audio = make_spec("A girl sings.", "reference_image", "reference_image", audio=True)
    for spec in (ref_image_spec(), audio):
        provider = Provider(ref_prompt(20), ref_prompt(20), critic=["would be ignored"])
        result = await run(provider, spec)
        assert provider.critic_bodies == []
        assert all(not v.startswith("Fidelity:") for v in result.soft_violations)
    provider = Provider(t2va_prompt("[Shot 1] A cat walks."), critic=[])
    await run(provider, make_spec("A cat walks."))
    assert len(provider.critic_bodies) == 1


@pytest.mark.asyncio
async def test_a_repair_that_drops_a_quoted_line_loses_to_the_first_answer():
    provider = Provider(
        GOOD_QUOTE.replace("</d>", "</d> A fidelity-flagged flourish"), BAD_QUOTE, critic=["flourish is odd"]
    )
    result = await run(provider, QUOTE_SPEC())
    assert "flourish" in result.prompt and any(v.startswith("Fidelity:") for v in result.soft_violations)


@pytest.mark.parametrize("base", [True, False])
def test_every_repair_message_keeps_the_detail_sentence(base):
    message = h3_prompt._repair_message(("A: x",), ("<Picture 1>",), base=base)
    assert message.endswith(
        "Keep everything that was already correct, including the level of detail and length of the descriptive "
        "section; change only what is needed to fix the listed points."
    )
    assert base or "The provided media labels are exactly: <Picture 1>." in message


# ----------------------------------------------------------------------------- length-aware repair sentence

SHORTEN = (
    "Regenerate from the original request and references, targeting at most 5500 characters for the whole prompt. "
    "Use at most 200 characters per subject definition and 90 per reference retention line; "
    "summary at most 250 characters, the description at most 1900, and audio fields at most 300 together. "
    "Describe appearance once and use subject IDs in the action. Never enumerate synonyms or repeat sentences; "
    "keep every required section, label, spoken line, subject appearance, clothing layer, action and spatial relation."
)


@pytest.mark.asyncio
async def test_repair_after_an_over_7000_character_answer_asks_to_shorten():
    too_long = t2va_prompt("[Shot 1] " + "A cat walks slowly. " * 400)
    assert len(too_long) > 7000
    provider = Provider(too_long, t2va_prompt("[Shot 1] A cat walks."))
    await run(provider, make_spec("A cat walks."))
    last = provider.bodies[1]["messages"][-1]["content"]
    assert last.endswith(SHORTEN) and h3_prompt.KEEP_DETAIL_SENTENCE not in last
    assert "it has" in last and "must contain 1 to 7000 characters" in last
    assert all(message["role"] != "assistant" for message in provider.bodies[1]["messages"])


@pytest.mark.asyncio
async def test_overlong_reference_recovery_uses_a_compact_system_with_all_original_references():
    too_long = ref_prompt(20).replace("Soft footsteps.", "room tone " * 1000)
    provider = Provider(too_long, ref_prompt(20))
    result = await run(provider, ref_image_spec())
    recovery = provider.bodies[1]["messages"]
    system = recovery[0]["content"]
    assert len(system) < 2000 and all(field in system for field in h3_prompt.REFERENCE_FIELDS)
    assert all(message["role"] != "assistant" for message in recovery)
    assert recovery[1] == provider.bodies[0]["messages"][1]
    assert "identity-defining appearance" in system and "Still subjects remain still" in system
    assert result.prompt == ref_prompt(20)
    assert "subject_definitions:\nsummary:\nretention_analysis:\n" in system
    assert "No markdown headings" in system


@pytest.mark.asyncio
async def test_many_picture_headroom_triggers_recovery_before_the_hard_limit():
    spec = make_spec("A woman walks.", *("reference_image",) * 5, duration=8)
    pictures = ", ".join(f"<Picture {i}>" for i in range(1, 6))
    clean = ref_prompt(20).replace("the coat of <Picture 2>", "the clothing from " + pictures)
    for i in range(3, 6):
        clean = clean.replace(
            "detailed_description:", f"<Picture {i}>: fully_preserved - appearance.\n\ndetailed_description:"
        )
    roomy = clean.replace("Soft footsteps.", "room tone. " * 380)
    assert 6000 < len(roomy) < 7000
    provider = Provider(roomy, clean)
    result = await run(provider, spec)
    assert result.prompt == clean and len(provider.bodies) == 2
    assert "6000 characters" in provider.bodies[1]["messages"][-1]["content"]
    assert len(provider.bodies[1]["messages"][0]["content"]) < 2000


@pytest.mark.asyncio
async def test_one_picture_with_safe_length_keeps_the_a3_writer_context():
    spec = ref_image_spec()
    prompt = ref_prompt(20).replace("Soft footsteps.", "room tone. " * 380)
    assert 6000 < len(prompt) < 7000
    provider = Provider(prompt)
    result = await run(provider, spec)
    assert result.prompt == prompt and len(provider.bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_truncated", [False, True])
async def test_exhausted_truncation_gets_one_fresh_recovery_and_keeps_paid_usage(monkeypatch, recovery_truncated):
    provider = Provider("runaway", ref_prompt(20), finish=["length", "length" if recovery_truncated else "stop"])
    monkeypatch.setattr(h3_prompt, "_remaining", lambda started: 5.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        rewriter = H3PromptRewriter(client, "key")
        with pytest.raises(h3_prompt.TruncatedRewriteError) as first:
            await rewriter.rewrite(ref_image_spec())
        assert first.value.retryable and len(provider.bodies) == 1
        monkeypatch.setattr(h3_prompt, "_remaining", lambda started: 30.0)
        if recovery_truncated:
            with pytest.raises(h3_prompt.TruncatedRewriteError) as second:
                await rewriter.rewrite(ref_image_spec())
            assert not second.value.retryable
        else:
            result = await rewriter.rewrite(ref_image_spec())
            assert result.prompt == ref_prompt(20) and result.usage.cost == 0.002
            assert result.usage.prompt_tokens == 200
    assert len(provider.bodies) == 2 and h3_prompt._FIRST_ANSWERS == {}
    assert provider.bodies[1]["messages"][0]["content"] == h3_prompt._compact_ref2va_system()
    assert all(message["role"] != "assistant" for message in provider.bodies[1]["messages"])


@pytest.mark.asyncio
async def test_fresh_truncation_recovery_counts_new_plan_usage(monkeypatch):
    plan_costs = iter((0.0003, 0.0004))

    async def plan(self, prepared, key, started, answer_cached=False):
        return None, h3_prompt.RewriteUsage(cost=next(plan_costs), prompt_tokens=5)

    monkeypatch.setattr(H3PromptRewriter, "ref2va_plan_notes", plan)
    provider = Provider("runaway", ref_prompt(20), finish=["length", "stop"])
    monkeypatch.setattr(h3_prompt, "_remaining", lambda started: 5.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        rewriter = H3PromptRewriter(client, "key")
        with pytest.raises(h3_prompt.TruncatedRewriteError):
            await rewriter.rewrite(ref_image_spec())
        monkeypatch.setattr(h3_prompt, "_remaining", lambda started: 30.0)
        result = await rewriter.rewrite(ref_image_spec())
    assert result.usage.cost == pytest.approx(0.0027)
    assert result.usage.prompt_tokens == 210
    assert len(provider.bodies) == 2


@pytest.mark.asyncio
async def test_overlong_audio_reference_keeps_the_original_media_guidance():
    provider = Provider(ref_prompt(20).replace("Soft footsteps.", "room tone " * 1000), ref_prompt(20))
    await run(provider, make_spec("A woman walks.", "reference_image", "reference_image", audio=True))
    assert provider.bodies[1]["messages"][0] == provider.bodies[0]["messages"][0]
    assert provider.bodies[1]["messages"][0]["content"] != h3_prompt._compact_ref2va_system()


@pytest.mark.asyncio
@pytest.mark.parametrize("audio", [False, True])
async def test_late_truncation_does_not_add_a_retry_to_other_media_paths(monkeypatch, audio):
    provider = Provider("runaway", finish=["length"])
    monkeypatch.setattr(h3_prompt, "_remaining", lambda started: 5.0)
    spec = make_spec("A woman walks.", "reference_image", audio=True) if audio else make_spec("A woman walks.")
    with pytest.raises(h3_prompt.TruncatedRewriteError) as failure:
        await run(provider, spec)
    assert not failure.value.retryable and len(provider.bodies) == 1
    assert h3_prompt._FIRST_ANSWERS == {}


@pytest.mark.asyncio
async def test_over_maximum_word_count_asks_to_shorten_and_too_short_keeps_detail():
    long_desc = " ".join(
        ["She walks along the quiet street while the camera slowly follows her pace."] * 70
    )  # >750 words
    assert len(long_desc.split()) > 750
    spec = ref_image_spec()
    provider = Provider(ref_prompt(70), ref_prompt(20))
    await run(provider, spec)
    last = provider.bodies[1]["messages"][-1]["content"]
    assert last.endswith(SHORTEN) and h3_prompt.KEEP_DETAIL_SENTENCE not in last
    provider = Provider(ref_prompt(2), ref_prompt(20))
    await run(provider, spec)
    last = provider.bodies[1]["messages"][-1]["content"]
    assert last.endswith(h3_prompt.KEEP_DETAIL_SENTENCE) and SHORTEN not in last


def test_ordinary_repairs_and_empty_answers_keep_the_detail_sentence():
    for violations in (("A: x",), (h3_prompt._V_LENGTH,)):
        assert h3_prompt._repair_message(violations, (), base=True).endswith(h3_prompt.KEEP_DETAIL_SENTENCE)


@pytest.mark.parametrize(
    "user,line",
    [
        ("A child waves in the kitchen.", "Hi"),
        ("The high shelf is blue.", "hi"),
        ("The person nods, then waits.", "then"),
        ("看着明天的日历，点点头。", "明天"),
    ],
)
def test_invented_short_dialogue_cannot_hide_in_user_prose(user, line):
    spec = make_spec(user, "reference_image")
    found = literal_violations(t2va_prompt(f"<d>[English] {line}</d>"), spec)
    assert any("must not add dialogue" in v for v in found)


@pytest.mark.parametrize("user,line", [('She says "Hi!"', "Hi!"), ("她说：“走！”", "走！")])
def test_short_explicit_dialogue_is_preserved(user, line):
    spec = make_spec(user, "reference_image")
    assert not literal_violations(t2va_prompt(f"<d>[English] {line}</d>"), spec)


def test_dropped_one_character_quoted_line_is_reported():
    spec = make_spec("她说：“走”", "reference_image")
    found = literal_violations(t2va_prompt("[Shot 1] She silently nods."), spec)
    assert any("quoted text verbatim" in v and "走" in v for v in found)


@pytest.mark.parametrize("user", ['She says "Hi!"', "She says 'Hi!'", "She says ‘Hi!’"])
def test_dropped_short_line_cannot_hide_inside_child(user):
    spec = make_spec(user, "reference_image")
    found = literal_violations(t2va_prompt("[Shot 1] A child silently nods."), spec)
    assert any("quoted text verbatim" in v and "Hi" in v for v in found)


def test_contractions_and_possessives_do_not_create_speech_requirements():
    spec = make_spec("A child's toy doesn't move.", "reference_image")
    assert not literal_violations(t2va_prompt("[Shot 1] A child's toy is still."), spec)


@pytest.mark.asyncio
async def test_invented_line_surviving_repair_cannot_return_as_a_soft_violation():
    invented = ref_prompt(20).replace("[Shot 1] ", "[Shot 1] <d>[English] Hi!</d> ")
    provider = Provider(invented, invented)
    with pytest.raises(RewriteError, match="must not add dialogue") as caught:
        await run(provider, ref_image_spec())
    assert not caught.value.retryable and len(provider.bodies) == 2


@pytest.mark.asyncio
async def test_invented_line_cannot_fall_back_after_failed_repair():
    invented = ref_prompt(20).replace("[Shot 1] ", "[Shot 1] <d>[English] Hi!</d> ")
    provider = Provider(invented, "invalid fields")
    with pytest.raises(RewriteError):
        await run(provider, ref_image_spec())
    assert len(provider.bodies) == 2


@pytest.mark.asyncio
async def test_invented_line_cannot_return_without_a_repair_budget(monkeypatch):
    invented = ref_prompt(20).replace("[Shot 1] ", "[Shot 1] <d>[English] Hi!</d> ")
    monkeypatch.setattr(h3_prompt, "_remaining", lambda started: 0.0)
    provider = Provider(invented)
    with pytest.raises(RewriteError, match="must not add dialogue"):
        await run(provider, ref_image_spec())
    assert len(provider.bodies) == 1


@pytest.mark.asyncio
async def test_clean_dialogue_repair_wins_even_with_more_minor_findings():
    invented = ref_prompt(20).replace("[Shot 1] ", "[Shot 1] <d>[English] Hi!</d> ")
    clean = ref_prompt(2)
    provider = Provider(invented, clean)
    result = await run(provider, ref_image_spec())
    assert result.prompt == clean
    assert not any(v.startswith(h3_prompt._V_SPEECH) for v in result.soft_violations)
