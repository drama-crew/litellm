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
GOOD_QUOTE = t2va_prompt('[Shot 1] A man says: <d>[English] I bet there is no bullet in your gun.</d>')
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
    [httpx.Response(500, json={"error": {"message": "boom"}}), "not json at all", httpx.ConnectError("down"), '{"defects": "x"}'],
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
    assert provider.bodies[0]["max_tokens"] == 8192 and provider.bodies[0]["model"] == "qwen/qwen3.8-flash"
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

    provider = Flaky(BAD_QUOTE, critic=[])
    result = await run(provider, QUOTE_SPEC())
    assert result.prompt == BAD_QUOTE and len(provider.bodies) == 2
    assert any("quoted text" in v for v in result.soft_violations)
    assert h3_prompt._FIRST_ANSWERS == {}


# 13
@pytest.mark.asyncio
async def test_hard_failure_surviving_the_repair_raises_non_retryable():
    bad = t2va_prompt("[Shot 1] A cat. [Shot 2] At 00:09.000, cut.")
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
