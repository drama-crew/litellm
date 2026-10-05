"""Directing plan for image-only Ref2VA rewrites: observe each picture, plan the shot, hand the plan to the writer.

Pure text and parsing helpers; the provider calls live in `H3PromptRewriter.ref2va_plan_notes`.
"""

from __future__ import annotations

import json
import re
import unicodedata

PLAN_ENV = "CAUSYN_H3_REF2VA_PLAN"
PLAN_MODEL_ENV = "CAUSYN_H3_REF2VA_PLAN_MODEL"
OBSERVE_MAX_TOKENS = 600
OBSERVE_MAX_TOKENS_MANY = 350  # more than PLAN_IMAGES_MAX pictures
PLAN_MAX_TOKENS_CAP = 4000
PLAN_IMAGES_MAX = 4  # above this the plan call gets the observations only (cost cap)
PLAN_REASONING = {"enabled": False}
PLAN_MODELS = frozenset({"qwen/qwen3.8-flash", "qwen/qwen3.8-omni-flash"})  # reasoning-off models only
PLAN_RETRY_AFTER_MAX_S = 5.0
PLAN_STAGE_MAX_S = 45.0
PLAN_STAGE_MIN_S = 10.0
PLAN_STAGE_RESERVE_S = 30.0

OBSERVE = """You are preparing to direct a short video. You are looking at ONE reference picture the user supplied.
Describe only what this picture actually shows, in compact English (80-150 words): the visible subject or subjects and
their identity-defining traits (face, age, hair, build, skin, clothing with colours and materials, accessories,
distinctive marks), objects and props (shape, colour, material, parts), the setting, lighting, composition and
rendering style, and any visible text verbatim. Be exact: name colours precisely (e.g. seal-brown with dark
points, teal, purple clay), facial hair (clean-shaven / stubble / short beard), eye colour only if clearly visible,
shapes (curved vs rectangular), materials (metal vs wood) and the count of small items and toppings. Start with one line "KIND: character | creature | product/object |
scene/place | composite" for what the picture mainly supplies.
This picture is a REFERENCE. It is not a frame of the video. Do not describe motion, camera, sound or what happens
next. Do not guess names, relationships, places or brands that are not visibly established. Say "unclear" when unsure."""

PLAN = """You are a video DIRECTOR writing a production plan for a {duration}-second reference-to-video clip. The user gave a
request and {n} reference picture(s) (shown, with an observation of each). Nothing has been filmed yet; you decide what
will be filmed. Where the user specified something, honour it exactly; where they left it open, decide it yourself.

Rules:
1. Pictures supply APPEARANCE (identity, clothing, objects, place, style), never events. Take the action from the
   request. A picture is not the first frame: do not open the video on the picture's pose/framing unless asked; adapt
   pose, framing and setting to the request.
2. Bind each picture to exactly one entity (person, creature, object, or a scene/place). Several pictures of the same
   thing form one entity. A place picture becomes a scene entity. Every picture the user supplied is used — one the
   request does not mention still appears (as a prop, a background element or the setting); exclude a picture only when
   using it would contradict the request, and give that reason.
3. When the request changes an aspect of a referenced subject (outfit, hairstyle, setting...), keep the identity from
   the picture and apply the requested change; record it under "changed". Never contradict the request.
4. Appearance facts: 5-10 concrete traits copied exactly from the picture/observation (colours, materials, shapes,
   facial hair, marks, counts). Never alter or add parts of an object (keep toppings, logos, patterns as shown); omit a
   trait the observation calls unclear.
5. ONE continuous shot unless the user explicitly asked for cuts. No fades.
6. Beats: 5-8 short beats in time order covering the whole {duration} s: a clear visible START state, the requested
   action(s) as concrete physical steps, and a clear visible END state. Only what the user asked for plus natural
   micro-movements (breathing, blinking, small weight shifts) — do NOT add extra events, people, animals, objects, exits
   or flights the user did not ask for. Everything must be achievable in {duration} s. Keep left/right positions exactly
   as the user stated. Refer to entities as {{e1}}.
7. Camera: "Static Shot" unless the user asked for a movement or the action leaves the frame; never choose a move that
   crops or hides a requested subject, garment or prop. Keep every requested subject in frame when they must interact.
8. Speech: only lines the user wrote, verbatim and in their original language (never translate, paraphrase or
   shorten). Give each line to the entity that speaks it with a concrete voice (gender, age, timbre, pace, emotion), the
   beat where it starts and whether the speaker is on screen; on-screen speakers' lips move in sync. Pace: a {duration}-s
   clip fits about {words} English words or {chars} Chinese characters, so start the line early enough to finish it. If
   the user says people talk but gives no words, they talk naturally and the speech is indistinct background
   conversation (no line) — never silent mouthing.
9. Sound effects: every sound the user asked for must appear. Add at most 3 other effects, only for clearly audible
   physical events the beats show (footsteps while walking, impacts, objects handled or set down, animal calls,
   vehicles, water, an instrument being played), each tied to its beat. No effect for quiet movements (gestures,
   smiles, glances, blinking, breathing, a raised hand, standing or sitting still, clothing, hair) and no sound without
   a visible source.
   For each required or justified effect, describe its audible character: name the source and physical event,
   supported material/contact surface, attack and decay (sharp click, dull thud, short ring, sustained hiss), and
   recurrence tied to the action (one strike, each footfall, continuous flow). Use only qualities justified by the
   request or visible source; do not invent material, an exact rate, louder intensity, or a new event. Prioritise
   the requested sound over incidental room tone. Silent requests override all incidental effects.
   Ambience: ONE concise sentence containing only sound sources established by the request or shot. A quiet
   scene may have only room tone; never invent extra sources to fill a quota.
   Music: "absent" unless the user asked for music or the request is explicitly a commercial, trailer, montage or
   music video; then follow the user's style, instrumentation, tempo and dynamics (non_diegetic unless it is played on
   screen). Music played or heard on screen (an instrument, a performance, a radio, a party) is diegetic and synced to
   the visible playing. Never add music the user did not ask for under speech.
10. requirements: list every explicit user requirement (subject, attribute, action, place, camera, style, sound).

Return ONLY JSON:
{{"look":"render medium, lighting, palette",
 "entities":[{{"id":"e1","name":"noun phrase","kind":"character|creature|object|scene","pictures":[1],
   "appearance_facts":["..."],"retention":"fully_preserved|partially_preserved|attribute_transfer|weak_reference",
   "changed":["requested change applied to this subject"]}}],
 "excluded_pictures":[{{"picture":2,"reason":"..."}}],
 "camera":"official value","beats":["{{e1}} ... (start state)","...","... (end state)"],
 "speech":[{{"speaker":"e1","voice":"gender, age, timbre, pace, emotion","text":"verbatim","language":"English",
   "beat":2,"visibility":"onscreen"}}],
 "sound_effects":[{{"beat":3,"source":"{{e1}} walking","sound":"crisp footsteps on gravel"}}],
 "ambience":"one sentence",
 "music":{{"status":"absent|non_diegetic|diegetic","description":"instrumentation, tempo, rhythm, dynamics, level"}},
 "requirements":["..."]}}"""

NOTES = """DIRECTING PLAN (prepared by the director for this request from the pictures and the user's words; follow it):
{plan}

How to use it: define one <Subject N> per entity, bound to its picture(s), with its appearance facts. Write exactly
one retention_analysis line per subject (level, what is kept, what changes). Write the detailed_description as one
continuous shot following the beats in order with the planned camera; add nothing the plan does not contain (no
extra events, people, sounds or music). Audio: introduce each speaker with a stable ID such as (S1) and the
planned voice, and write each line as <d>[Language] exact text</d> at its beat with the speaker's lips moving; describe
each sound effect inside the beat where its action happens; overall_soundscape = the ambience sentence plus the
recurring effects (no dialogue, no music); non_diegetic_music = the music decision (N/A when absent or diegetic);
diegetic music is described in the action, synced to the visible playing.
The audio lines come in addition to the visual description, never instead of it: keep every subject definition and
the detailed_description as visually detailed as they would be without them.
Pictures are references, not frames. Do not add dialogue beyond the plan."""


_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_LINE_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_SHORT_QUOTED = re.compile(
    r'"([^"\n]{1,300})"|“([^”\n]{1,300})”|「([^」\n]{1,300})」|『([^』\n]{1,300})』'
    r"|(?<![\w'])'([^\n]{1,300}?)'(?!\w)|‘([^\n]{1,300}?)’"
)
_SPEECH_CUE = re.compile(
    r"(?:\b(?:say(?:s|ing)?|said|ask(?:s|ing)?|repl(?:y|ies|ying)|shout(?:s|ing)?|whisper(?:s|ing)?|"
    r"exclaim(?:s|ing)?|call(?:s|ing)?|yell(?:s|ing)?|utter(?:s|ing)?)\b|(?:说|喊|问|回答|台词)[：:]?)"
    r"\s*[:：]?\s*(?:(?:just|exactly|the words?|the phrase|the line)\s+)?"
    r"(?:[\w’ -]+?\s+(?:and|then)\s+)?(?:[\"'“‘「『]\s*)?$",
    re.I,
)


def _norm_line(text: str) -> str:
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", text).casefold())


def user_line_written(text: str, prompt: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = tuple(_LINE_WORDS.findall(normalized))
    if not words:
        return False
    source = unicodedata.normalize("NFKC", prompt).casefold()
    if any(
        _norm_line(text) == _norm_line(next(g for g in m.groups() if g is not None))
        for m in _SHORT_QUOTED.finditer(source)
    ):
        return True
    start_boundary = "" if re.search(r"[㐀-鿿]", normalized) else r"(?<!\w)"
    pattern = start_boundary + r"[\W_]+".join(re.escape(w) for w in words) + r"(?!\w)"
    matches = tuple(re.finditer(pattern, source))
    if len(words) >= 3 or len(_norm_line(text)) >= 12:
        return bool(matches)
    return any(_SPEECH_CUE.search(source[: m.start()]) is not None for m in matches)


def user_lines_only(speech, prompt):
    if not isinstance(speech, list):
        return []
    return [
        sp
        for sp in speech
        if isinstance(sp, dict) and isinstance(sp.get("text"), str) and user_line_written(sp["text"], prompt)
    ]


def _items(value):
    """A plan list field, or nothing when the model returned a string/dict instead (the stage is fail-open)."""
    return value if isinstance(value, list) else []


def render_plan(plan, n_pictures):
    lines = [f"Look: {plan.get('look', '')}"]
    for e in plan.get("entities", []):
        pics = ", ".join(f"picture {p}" for p in e.get("pictures", []))
        lines.append(
            f"Entity {e['id']} = {e.get('name')} ({e.get('kind')}; from {pics}; retention {e.get('retention')}): "
            + "; ".join(e.get("appearance_facts", []))
            + (f". Requested changes: {'; '.join(e['changed'])}" if e.get("changed") else "")
        )
    for x in plan.get("excluded_pictures", []):
        lines.append(f"Excluded picture {x.get('picture')}: {x.get('reason')}")
    lines.append(f"Camera: {plan.get('camera')}")
    lines.append("Beats (in order, one continuous shot):")
    lines += [f"  {i}. {b}" for i, b in enumerate(plan.get("beats", []), 1)]
    names = {e.get("id"): e.get("name") for e in plan.get("entities", []) if isinstance(e, dict)}
    for sp in _items(plan.get("speech") or plan.get("dialogue")):
        if not isinstance(sp, dict):
            continue
        who = sp.get("speaker")
        who = f"{{{who}}} ({names[who]})" if isinstance(who, str) and who in names else str(who or "a speaker")
        lines.append(
            f"Speech at beat {sp.get('beat')}: {who}, voice {sp.get('voice', '')}, {sp.get('visibility', 'onscreen')}, "
            f"says in {sp.get('language', '')}: {json.dumps(sp.get('text', ''), ensure_ascii=False)}"
        )
    for fx in _items(plan.get("sound_effects")):
        if not isinstance(fx, dict):
            continue
        lines.append(f"Sound effect at beat {fx.get('beat')}: {fx.get('source')} — {fx.get('sound')}")
    m = plan.get("music") or {}
    lines.append(f"Music: {m.get('status')} — {m.get('description', '')}")
    lines.append(f"Ambience: {plan.get('ambience', '')}")
    if plan.get("requirements"):
        lines.append("User requirements to cover: " + "; ".join(plan["requirements"]))
    return "\n".join(lines)


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_plan(text: str) -> dict:
    """The first-to-last `{...}` span of a plan reply as a dict with usable entities and beats; ValueError otherwise."""
    found = _JSON_OBJECT.search(text)
    if found is None:
        raise ValueError("no JSON object")
    plan = json.loads(found.group(0))
    if not isinstance(plan, dict):
        raise ValueError("plan is not an object")
    entities, beats = plan.get("entities"), plan.get("beats")
    if not (isinstance(entities, list) and entities and all(isinstance(e, dict) and "id" in e for e in entities)):
        raise ValueError("plan has no entities")
    if not (isinstance(beats, list) and beats):
        raise ValueError("plan has no beats")
    return plan


def plan_max_tokens(n_pictures: int) -> int:
    return min(PLAN_MAX_TOKENS_CAP, 1500 + 300 * n_pictures)


MANY_RULE = "11. There are more than 4 pictures: give each entity 3-5 appearance facts and use at most 7 beats.\n\n"
MANY_LENGTH = (
    "Length: the complete six-section prompt must fit within 5500 characters, including spaces and labels. "
    "With this many subjects, use one brief sentence per subject definition and one brief retention line per "
    "reference. Keep the detailed_description near 350 words, with no repeated appearance or ambience paragraphs. "
    "Keep all references, requested actions and complete spoken lines; compress wording rather than omit them."
)


def observe_max_tokens(n_pictures: int) -> int:
    return OBSERVE_MAX_TOKENS_MANY if n_pictures > PLAN_IMAGES_MAX else OBSERVE_MAX_TOKENS


def plan_system(duration: int, n_pictures: int) -> str:
    text = PLAN
    if n_pictures > PLAN_IMAGES_MAX:
        text = text.replace("Return ONLY JSON:", MANY_RULE + "Return ONLY JSON:", 1)
    return text.format(duration=duration, n=n_pictures, words=round(duration * 2.5), chars=round(duration * 4))


def plan_notes(plan: dict, n_pictures: int) -> str:
    notes = NOTES.format(plan=render_plan(plan, n_pictures))
    return notes + "\n" + MANY_LENGTH if n_pictures > PLAN_IMAGES_MAX else notes
