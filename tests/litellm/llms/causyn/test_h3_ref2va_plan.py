from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from litellm.llms.causyn import h3_media, h3_prompt, ref2va_plan
from litellm.llms.causyn.h3_prompt import ContextIRRequest, H3PromptRewriter, RewriteError

IMG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/"


def variant(n: int) -> str:
    """A distinct but still valid base64 image URL (the 40th payload char is a letter, not padding)."""
    i = IMG.index("AAMCAgIC")
    return IMG[:i] + "ABCDEFGH"[n] + IMG[i + 1 :]


IMG2 = variant(1)
VID = "data:video/mp4;base64,AAAA"
AUD = "https://media.example/a.wav"

PLAN = {
    "look": "soft daylight, painterly",
    "entities": [
        {
            "id": "e1",
            "name": "a young woman",
            "kind": "character",
            "pictures": [1, 2],
            "appearance_facts": ["short black hair", "red wool coat"],
            "retention": "fully_preserved",
            "changed": ["wears a blue scarf"],
        },
        {
            "id": "e2",
            "name": "a teal kettle",
            "kind": "object",
            "pictures": [3],
            "appearance_facts": ["teal enamel"],
            "retention": "weak_reference",
            "changed": [],
        },
    ],
    "excluded_pictures": [{"picture": 4, "reason": "duplicate of picture 1"}],
    "camera": "Static Shot",
    "beats": ["{e1} stands (start state)", "{e1} lifts {e2}", "{e1} pours tea (end state)"],
    "dialogue": [{"speaker": "e1", "text": "Hello, 你好", "language": "English"}],
    "music": {"status": "absent", "description": ""},
    "ambience": "Quiet kitchen with a ticking clock.",
    "requirements": ["woman", "kettle"],
}
# Output of the evaluated prototype's render_plan(PLAN, 4).
GOLDEN_RENDER = (
    "Look: soft daylight, painterly\n"
    "Entity e1 = a young woman (character; from picture 1, picture 2; retention fully_preserved): "
    "short black hair; red wool coat. Requested changes: wears a blue scarf\n"
    "Entity e2 = a teal kettle (object; from picture 3; retention weak_reference): teal enamel\n"
    "Excluded picture 4: duplicate of picture 1\n"
    "Camera: Static Shot\n"
    "Beats (in order, one continuous shot):\n"
    "  1. {e1} stands (start state)\n"
    "  2. {e1} lifts {e2}\n"
    "  3. {e1} pours tea (end state)\n"
    'Dialogue (verbatim): [{"speaker": "e1", "text": "Hello, 你好", "language": "English"}]\n'
    "Music: absent — \n"
    "Ambience: Quiet kitchen with a ticking clock.\n"
    "User requirements to cover: woman; kettle"
)

DESC = " ".join(["She walks along the quiet street while the camera slowly follows her pace."] * 20)
WRITER_ANSWER = (
    "subject_definitions:\n<Subject 1> is the woman from <Picture 1> in the coat of <Picture 2>.\n\n"
    "summary:\n[reference generation] A woman walks.\n\n"
    "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - face from <Picture 1>\n"
    "and the coat of <Picture 2>.\n"
    "<Picture 2>: fully_preserved - the coat.\n\n"
    f"detailed_description:\nA calm look.\n[Shot 1] {DESC}\n\n"
    "overall_soundscape:\nSoft footsteps.\n\nnon_diegetic_music:\nN/A"
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    h3_prompt._FACTS_MEMO.clear()
    h3_prompt._FIRST_ANSWERS.clear()
    h3_prompt._PLAN_NOTES.clear()
    h3_prompt._warned_models.clear()
    for name in ("CAUSYN_H3_REF2VA_PLAN", "CAUSYN_H3_REF2VA_PLAN_MODEL", "CAUSYN_H3_REF2VA_REWRITE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAUSYN_H3_REWRITE_CRITIC", "0")

    async def prepare(client, request):
        return request, h3_media.PreparedRaw(videos=(), audios=())

    monkeypatch.setattr(h3_media, "prepare_media_with_raw", prepare)

    async def perceive(request, raw, key=None):
        return h3_prompt.RefFacts()

    monkeypatch.setattr(h3_prompt, "perceive_media", perceive)


def spec_of(*kinds: str, duration: int = 8) -> ContextIRRequest:
    urls = {"image": IMG, "image2": IMG2}
    content: list[dict] = [{"type": "text", "text": "A woman walks."}]
    for kind in kinds:
        if kind.startswith("image"):
            content.append({"type": "image_url", "image_url": {"url": urls[kind]}, "role": "reference_image"})
        elif kind == "video":
            content.append({"type": "video_url", "video_url": {"url": VID}, "role": "reference_video"})
        else:
            content.append({"type": "audio_url", "audio_url": {"url": AUD}, "role": "reference_audio"})
    return ContextIRRequest.model_validate({"model": "causyn-1.1", "content": content, "duration": duration, "ratio": "16:9"})


def stage_of(body: dict) -> str:
    system = body["messages"][0]["content"]
    if system == ref2va_plan.OBSERVE:
        return "observe"
    if isinstance(system, str) and system.startswith("You are a video DIRECTOR"):
        return "plan"
    return "writer"


def completion(body: dict, text: str, cost: float = 0.001) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": cost},
        },
    )


class Provider:
    def __init__(self, *, observe=None, plan=None, writer=None) -> None:
        self.observe = observe or (lambda body: completion(body, "KIND: character\nA woman."))
        self.plan = plan or (lambda body: completion(body, "Plan:\n" + json.dumps(PLAN)))
        self.writer = writer or (lambda body: completion(body, WRITER_ANSWER))
        self.bodies: list[dict] = []

    def stages(self) -> list[str]:
        return [stage_of(b) for b in self.bodies]

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        handler = {"observe": self.observe, "plan": self.plan, "writer": self.writer}[stage_of(body)]
        result = handler(body)
        return await result if asyncio.iscoroutine(result) else result


async def run(provider: Provider, spec: ContextIRRequest):
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        return await H3PromptRewriter(client, "key").rewrite(spec)


def writer_bodies(provider: Provider) -> list[dict]:
    return [b for b in provider.bodies if stage_of(b) == "writer"]


@pytest.mark.asyncio
async def test_image_only_ref2va_observes_each_picture_plans_then_writes_with_the_plan():
    provider = Provider()
    result = await run(provider, spec_of("image", "image2"))
    assert sorted(provider.stages()) == ["observe", "observe", "plan", "writer"]
    assert provider.stages()[-2:] == ["plan", "writer"]
    observes = [b for b in provider.bodies if stage_of(b) == "observe"]
    for body in observes:
        assert body["messages"][0] == {"role": "system", "content": ref2va_plan.OBSERVE}
        assert [p["type"] for p in body["messages"][1]["content"]] == ["image_url"]
        assert body["max_tokens"] == 600 and body["temperature"] == 0
        assert body["reasoning"] == {"enabled": False}
        assert body["provider"] == {"allow_fallbacks": False, "require_parameters": True}
        assert body["model"] == "qwen/qwen3.8-omni-flash"
    assert {b["messages"][1]["content"][0]["image_url"]["url"] for b in observes} == {IMG, IMG2}
    plan = next(b for b in provider.bodies if stage_of(b) == "plan")
    assert plan["messages"][0]["content"] == ref2va_plan.PLAN.format(duration=8, n=2)
    assert plan["max_tokens"] == 1500 + 300 * 2
    content = plan["messages"][1]["content"]
    assert [p["type"] for p in content] == ["text", "image_url", "text"] * 2 + ["text"]
    assert content[0]["text"] == "Picture 1:" and content[2]["text"].startswith("Observation of picture 1:\nKIND: character")
    assert content[-1]["text"] == "USER REQUEST (8 s):\nA woman walks."
    (writer,) = writer_bodies(provider)
    last = writer["messages"][1]["content"][-1]
    assert last["type"] == "text" and last["text"].startswith("\n\nDIRECTING PLAN")
    assert "  2. {e1} lifts {e2}" in last["text"]
    assert result.usage.total_tokens == 4 * 110
    assert result.usage.cost == pytest.approx(0.004)


@pytest.mark.asyncio
@pytest.mark.parametrize("kinds", [("image", "video"), ("image", "audio"), ("image", "video", "audio")])
async def test_reference_video_or_audio_gets_no_plan_stage(kinds):
    provider = Provider()
    spec = spec_of(*kinds)
    try:
        await run(provider, spec)
    except RewriteError:
        pass  # the canned answer need not satisfy the video/audio rules; only the calls matter
    assert set(provider.stages()) <= {"writer"}
    assert not any("DIRECTING PLAN" in json.dumps(b) for b in provider.bodies)


@pytest.mark.asyncio
async def test_non_ref2va_modes_get_no_plan_stage():
    provider = Provider(writer=lambda body: completion(body, "integrated_multimodal_description: [Shot 1] A cat.\noverall_soundscape: Quiet.\nnon_diegetic_music: None."))
    spec = ContextIRRequest.model_validate(
        {"model": "MiniMax-H3", "content": [{"type": "text", "text": "A cat."}], "duration": 5, "ratio": "16:9"}
    )
    await run(provider, spec)
    assert provider.stages() == ["writer"]
    assert "DIRECTING PLAN" not in json.dumps(provider.bodies)


@pytest.mark.asyncio
async def test_switch_off_skips_the_plan_stage_and_leaves_the_writer_payload_alone(monkeypatch):
    monkeypatch.setenv("CAUSYN_H3_REF2VA_PLAN", "0")
    provider = Provider()
    spec = spec_of("image", "image2")
    await run(provider, spec)
    assert provider.stages() == ["writer"]
    user = provider.bodies[0]["messages"][1]["content"]
    assert user == spec.user_content(h3_prompt.RefFacts(), "qwen/qwen3.8-omni-flash", False)


@pytest.mark.asyncio
async def test_unknown_plan_model_skips_the_stage_with_one_warning(monkeypatch, caplog):
    monkeypatch.setenv("CAUSYN_H3_REF2VA_PLAN_MODEL", "evil/model")
    provider = Provider()
    with caplog.at_level("WARNING"):
        result = await run(provider, spec_of("image", "image2"))
        await run(provider, spec_of("image2", "image"))
    assert result.prompt == WRITER_ANSWER.strip()
    assert set(provider.stages()) == {"writer"}
    assert sum("CAUSYN_H3_REF2VA_PLAN_MODEL" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_plan_model_env_selects_the_other_reasoning_off_model(monkeypatch):
    monkeypatch.setenv("CAUSYN_H3_REF2VA_PLAN_MODEL", "qwen/qwen3.8-flash")
    provider = Provider()
    await run(provider, spec_of("image", "image2"))
    for stage in ("observe", "plan"):
        body = next(b for b in provider.bodies if stage_of(b) == stage)
        assert body["model"] == "qwen/qwen3.8-flash" and body["reasoning"] == {"enabled": False}


@pytest.mark.asyncio
async def test_reasoning_plan_model_is_not_allowed_and_skips_the_stage(monkeypatch):
    monkeypatch.setenv("CAUSYN_H3_REF2VA_PLAN_MODEL", "qwen/qwen3.8-max-0902")
    provider = Provider()
    await run(provider, spec_of("image", "image2"))
    assert provider.stages() == ["writer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken",
    [
        {"observe": lambda body: httpx.Response(500)},
        {"plan": lambda body: completion(body, "not json at all")},
        {"plan": lambda body: completion(body, '{"entities": [], "beats": []}')},
        {"plan": lambda body: httpx.Response(429)},
    ],
)
async def test_plan_stage_failures_fall_back_to_the_plain_rewrite(broken):
    provider = Provider(**broken)
    result = await run(provider, spec_of("image", "image2"))
    assert result.prompt == WRITER_ANSWER.strip()
    (writer,) = writer_bodies(provider)
    assert "DIRECTING PLAN" not in json.dumps(writer)


@pytest.mark.asyncio
async def test_plan_timeout_falls_back_to_the_plain_rewrite(monkeypatch):
    monkeypatch.setattr(ref2va_plan, "PLAN_STAGE_MAX_S", 0.05)
    monkeypatch.setattr(ref2va_plan, "PLAN_STAGE_MIN_S", 0.0)

    async def slow(body):
        await asyncio.sleep(1)
        return completion(body, json.dumps(PLAN))

    provider = Provider(plan=slow)
    result = await run(provider, spec_of("image", "image2"))
    assert result.prompt == WRITER_ANSWER.strip()
    assert "DIRECTING PLAN" not in json.dumps(writer_bodies(provider))


@pytest.mark.asyncio
async def test_plan_stage_is_skipped_without_time_budget(monkeypatch):
    monkeypatch.setattr(h3_prompt, "_remaining", lambda started: 49.0)  # 49 - 10 - 30 < 10
    provider = Provider()
    await run(provider, spec_of("image", "image2"))
    assert provider.stages() == ["writer"]


@pytest.mark.asyncio
async def test_retry_after_a_retryable_writer_failure_reuses_the_plan():
    calls = {"n": 0}

    def writer(body):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else completion(body, WRITER_ANSWER)

    provider = Provider(writer=writer)
    spec = spec_of("image", "image2")
    with pytest.raises(RewriteError) as caught:
        await run(provider, spec)
    assert caught.value.retryable
    assert sorted(provider.stages()) == ["observe", "observe", "plan", "writer"]
    result = await run(provider, spec)
    assert result.prompt == WRITER_ANSWER.strip()
    assert sorted(provider.stages()) == ["observe", "observe", "plan", "writer", "writer"]
    assert "DIRECTING PLAN" in json.dumps(writer_bodies(provider)[1])
    assert result.usage.total_tokens == 110  # the memo hit spent nothing on observe/plan


def test_render_plan_matches_the_prototype_golden_output():
    assert ref2va_plan.render_plan(PLAN, 4) == GOLDEN_RENDER


def test_notes_wrap_the_rendered_plan():
    notes = ref2va_plan.plan_notes(PLAN, 4)
    assert notes.startswith("DIRECTING PLAN (prepared by the director for this request")
    assert GOLDEN_RENDER in notes
    assert notes.endswith("Do not add dialogue beyond the plan.")


@pytest.mark.asyncio
async def test_repeating_a_delivered_request_reuses_the_notes_without_billing_them_again():
    provider = Provider()
    spec = spec_of("image", "image2")
    first = await run(provider, spec)
    assert first.usage.total_tokens == 4 * 110
    second = await run(provider, spec)
    assert provider.stages().count("observe") == 2 and provider.stages().count("plan") == 1
    assert second.usage.total_tokens == 110
    assert "DIRECTING PLAN" in json.dumps(writer_bodies(provider)[1])


@pytest.mark.asyncio
async def test_a_cached_first_answer_without_memoized_notes_skips_the_stage():
    spec = spec_of("image", "image2")
    key = h3_prompt.media_key(spec, (), ()) + "qwen/qwen3.8-omni-flash"
    h3_prompt._FIRST_ANSWERS[key] = (WRITER_ANSWER.strip(), h3_prompt.RewriteUsage())
    provider = Provider()
    result = await run(provider, spec)
    assert provider.stages() == []  # valid cached answer: no observe, plan or writer call
    assert result.prompt == WRITER_ANSWER.strip()


@pytest.mark.asyncio
async def test_nine_pictures_observe_concurrently_and_plan_with_text_only():
    active = {"now": 0, "peak": 0}

    async def observe(body):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return completion(body, "KIND: character\nA woman.")

    provider = Provider(observe=observe)
    spec = ContextIRRequest.model_validate(
        {
            "model": "causyn-1.1",
            "content": [{"type": "text", "text": "A woman walks."}]
            + [{"type": "image_url", "image_url": {"url": variant(i)}, "role": "reference_image"} for i in range(8)]
            + [{"type": "image_url", "image_url": {"url": IMG + ""[:0]}, "role": "reference_image"}],
            "duration": 8,
            "ratio": "16:9",
        }
    )
    try:
        await run(provider, spec)
    except RewriteError:
        pass  # the canned writer answer only knows two pictures
    assert provider.stages().count("observe") == 9 and active["peak"] == 9
    plan = next(b for b in provider.bodies if stage_of(b) == "plan")
    assert plan["max_tokens"] == 4000
    kinds = [p["type"] for p in plan["messages"][1]["content"]]
    assert "image_url" not in kinds and kinds.count("text") == 9 * 2 + 1
    assert plan["messages"][1]["content"][0]["text"] == "Picture 1:"
    assert plan["messages"][1]["content"][1]["text"].startswith("Observation of picture 1:")


@pytest.mark.asyncio
async def test_four_pictures_keep_images_in_the_plan_call_and_five_drop_them():
    for count, expect_images in ((4, 4), (5, 0)):
        provider = Provider()
        spec = ContextIRRequest.model_validate(
            {
                "model": "causyn-1.1",
                "content": [{"type": "text", "text": "A woman walks."}]
                + [{"type": "image_url", "image_url": {"url": variant(i + 2)}, "role": "reference_image"} for i in range(count)],
                "duration": 8,
                "ratio": "16:9",
            }
        )
        try:
            await run(provider, spec)
        except RewriteError:
            pass
        plan = next(b for b in provider.bodies if stage_of(b) == "plan")
        kinds = [p["type"] for p in plan["messages"][1]["content"]]
        assert kinds.count("image_url") == expect_images
        assert plan["max_tokens"] == min(4000, 1500 + 300 * count)


@pytest.mark.asyncio
async def test_a_429_is_retried_once_and_the_stage_still_succeeds(monkeypatch):
    monkeypatch.setattr(ref2va_plan, "PLAN_RETRY_AFTER_MAX_S", 0.0)
    state = {"n": 0}

    def observe(body):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return completion(body, "KIND: character\nA woman.")

    provider = Provider(observe=observe)
    await run(provider, spec_of("image", "image2"))
    assert "DIRECTING PLAN" in json.dumps(writer_bodies(provider))
    assert provider.stages().count("observe") == 3


@pytest.mark.asyncio
async def test_a_persistent_failure_is_retried_only_once_and_partial_spend_is_billed(monkeypatch):
    monkeypatch.setattr(ref2va_plan, "PLAN_RETRY_AFTER_MAX_S", 0.0)
    provider = Provider(plan=lambda body: httpx.Response(500))
    result = await run(provider, spec_of("image", "image2"))
    assert provider.stages().count("plan") == 2
    assert result.usage.total_tokens == 3 * 110  # two observes + the writer


@pytest.mark.asyncio
async def test_cancellation_propagates_out_of_the_stage():
    started = asyncio.Event()

    async def hang(body):
        started.set()
        await asyncio.sleep(30)

    provider = Provider(observe=hang)
    task = asyncio.create_task(run(provider, spec_of("image", "image2")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_failure_logs_leaf_classes_and_status_but_no_text(monkeypatch, caplog):
    monkeypatch.setattr(ref2va_plan, "PLAN_RETRY_AFTER_MAX_S", 0.0)
    with caplog.at_level("INFO"):
        await run(Provider(observe=lambda body: httpx.Response(500)), spec_of("image", "image2"))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "RewriteError(500)" in text
    assert "A woman" not in text and "walks" not in text


@pytest.mark.asyncio
async def test_timeout_is_logged_with_its_own_reason(monkeypatch, caplog):
    monkeypatch.setattr(ref2va_plan, "PLAN_STAGE_MAX_S", 0.05)
    monkeypatch.setattr(ref2va_plan, "PLAN_STAGE_MIN_S", 0.0)

    async def slow(body):
        await asyncio.sleep(1)

    with caplog.at_level("INFO"):
        await run(Provider(observe=slow), spec_of("image", "image2"))
    assert any("timed out" in r.getMessage() for r in caplog.records)
