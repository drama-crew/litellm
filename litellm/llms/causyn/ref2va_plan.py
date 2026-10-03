"""Directing plan for image-only Ref2VA rewrites: observe each picture, plan the shot, hand the plan to the writer.

Pure text and parsing helpers; the provider calls live in `H3PromptRewriter.ref2va_plan_notes`.
"""
from __future__ import annotations

import json
import re

PLAN_ENV = "CAUSYN_H3_REF2VA_PLAN"
PLAN_MODEL_ENV = "CAUSYN_H3_REF2VA_PLAN_MODEL"
OBSERVE_MAX_TOKENS = 600
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
2. Bind each picture to exactly one entity (person, creature, object, or a scene/place) or exclude it with a reason.
   Several pictures of the same thing form one entity. A place picture becomes a scene entity.
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
8. Speech: only lines the user wrote, verbatim. If the user says people talk/converse but gives no words, they talk
   naturally (visible lip movement) and the speech is indistinct background conversation — never silent mouthing.
9. Music: "absent" unless the user asked for music or the request is explicitly a commercial, trailer, montage or
   music video; then describe instrumentation and tempo. Ambience: ONE sentence, 28-46 words, 3-6 sound sources that
   are physically present in the shot (no unrelated rooms, appliances or crowds).
10. requirements: list every explicit user requirement (subject, attribute, action, place, camera, style, sound).

Return ONLY JSON:
{{"look":"render medium, lighting, palette",
 "entities":[{{"id":"e1","name":"noun phrase","kind":"character|creature|object|scene","pictures":[1],
   "appearance_facts":["..."],"retention":"fully_preserved|partially_preserved|attribute_transfer|weak_reference",
   "changed":["requested change applied to this subject"]}}],
 "excluded_pictures":[{{"picture":2,"reason":"..."}}],
 "camera":"official value","beats":["{{e1}} ... (start state)","...","... (end state)"],
 "dialogue":[{{"speaker":"e1","text":"verbatim","language":"English"}}],
 "music":{{"status":"absent|non_diegetic|diegetic","description":"..."}},
 "ambience":"one sentence",
 "requirements":["..."]}}"""

NOTES = """DIRECTING PLAN (prepared by the director for this request from the pictures and the user's words; follow it):
{plan}

How to use it: define one <Subject N> per entity, bound to its picture(s), with its appearance facts. Write exactly
one retention_analysis line per subject (level, what is kept, what changes). Write the detailed_description as one
continuous shot following the beats in order with the planned camera; add nothing the plan does not contain (no
extra events, people, sounds or music). Use the music decision (absent -> N/A) and the ambience sentence.
Pictures are references, not frames. Do not add dialogue beyond the plan."""


def render_plan(plan, n_pictures):
    lines = [f"Look: {plan.get('look', '')}"]
    for e in plan.get("entities", []):
        pics = ", ".join(f"picture {p}" for p in e.get("pictures", []))
        lines.append(f"Entity {e['id']} = {e.get('name')} ({e.get('kind')}; from {pics}; retention {e.get('retention')}): "
                     + "; ".join(e.get("appearance_facts", []))
                     + (f". Requested changes: {'; '.join(e['changed'])}" if e.get("changed") else ""))
    for x in plan.get("excluded_pictures", []):
        lines.append(f"Excluded picture {x.get('picture')}: {x.get('reason')}")
    lines.append(f"Camera: {plan.get('camera')}")
    lines.append("Beats (in order, one continuous shot):")
    lines += [f"  {i}. {b}" for i, b in enumerate(plan.get("beats", []), 1)]
    if plan.get("dialogue"):
        lines.append("Dialogue (verbatim): " + json.dumps(plan["dialogue"], ensure_ascii=False))
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


def plan_notes(plan: dict, n_pictures: int) -> str:
    return NOTES.format(plan=render_plan(plan, n_pictures))
