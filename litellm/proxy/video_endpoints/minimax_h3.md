# MiniMax H3 video API

API base: `https://api.causyn.cn`.

| Processing | Prefix | Public model |
| --- | --- | --- |
| Built-in Context IR then generation | `/video/minimax-h3` | `MiniMax-H3` |
| Caller prompt unchanged | `/video/minimax-h3/direct` | `MiniMax-H3` |

Swap the base URL of the [MiniMax official V2 API](https://platform.minimax.io) for one of these prefixes and every path equals the official one (`/v2/video_generation`, `/v2/query/video_generation[/{id}]`, `/v2/video_generation/{id}`, `/v2/h3_context_ir`, the latter on the rewrite prefix only). The model is `MiniMax-H3`; the lower-case `minimax-h3` is kept as a legacy alias. `MiniMax-H3-Max` (an official enum value) is not available here and is rejected with HTTP 400. The `/video/minimax-h3-test` prefixes mirror all of this against the isolated test worker pool.

Each prefix exposes:

| Method | Suffix | Result |
| --- | --- | --- |
| POST | `/v2/video_generation` | `{"task_id":"..."}` |
| GET | `/v2/query/video_generation/{task_id}` | `{"task":{...}}` |
| GET | `/v2/query/video_generation` | `{"items":[...],"total":0}` |
| DELETE | `/v2/video_generation/{task_id}` | `{"task_id":"...","action":"cancelled","status":"cancelled"}` or `deleted` |

Use a normal Causyn API key. The public model is normalized before authorization to the existing internal `causyn-1.1` identity, which remains the allowlist, policy and billing key. Custom route allowlists must include the new paths. The old root `/v2` H3 paths are removed, not redirected.

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "The kite rises over a calm beach. Gentle surf is audible."},
    {"type": "image_url", "image_url": {"url": "https://your-media-host/first.png"}, "role": "first_frame"},
    {"type": "image_url", "image_url": {"url": "https://your-media-host/last.png"}, "role": "last_frame"}
  ],
  "resolution": "768P",
  "duration": 5,
  "ratio": "adaptive"
}
```

For reference mode, use 1–9 images with `role: reference_image` and an explicit ratio. Text-only requests also require an explicit ratio: `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, or `9:16`. First-frame or first/last-frame requests use adaptive canvas sizing. Exactly one non-empty text item is required. Duration is an integer from 5 through 15 seconds; the official 4 s is not offered yet and is rejected with HTTP 400 up front on every endpoint (including `h3_context_ir`). Current deployment supports 768P with generated audio.

Capability gaps are explicit, never silently ignored (all reject before submission): resolution `480P`/`2K` (400, only 768P), `mm_file://` URLs (400, there is no file-upload API: use a public HTTP(S) URL or a Base64 data URL), `callback_url` (422, poll the query endpoint), last-frame-only requests (422, a `first_frame` is required), mixed keyframe/reference input and unknown fields (400).

### `extra.prompt_expansion_mode`

The official optional `extra` object is accepted with its single key `prompt_expansion_mode` (`disabled` | `balanced` | `quality`). `"extra": null` is treated as absent; `"prompt_expansion_mode": null`, any other key and any other value are HTTP 400. It is never forwarded: it only has to agree with the base URL, because the prefix decides whether the prompt is rewritten. On the rewrite prefix `quality` currently behaves exactly like `balanced`: both select the built-in Context IR rewrite.

| Prefix | absent | `balanced` / `quality` | `disabled` |
| --- | --- | --- | --- |
| `/video/minimax-h3` (rewrite) | rewrite | rewrite (same for both) | 422 `unsupported_parameter`: use the `/video/minimax-h3/direct` base URL |
| `/video/minimax-h3/direct` | prompt unchanged | 422 `unsupported_parameter`: use the `/video/minimax-h3` base URL | prompt unchanged |

`extra` is not a field of `/v2/h3_context_ir` (as in the official schema).

### Reference video and audio

Both `/video/minimax-h3/v2/video_generation` (built-in rewrite) and `/video/minimax-h3/direct/v2/video_generation` (prompt unchanged) accept these content items, with the same limits, in addition to `text`:

| `type` | field | `role` | limits |
|---|---|---|---|
| `image_url` | `image_url.url` | `reference_image` | up to 9 without reference video; up to 4 with any reference video; JPEG/PNG/WEBP |
| `video_url` | `video_url.url` | `reference_video` | up to 3; MP4/MOV, H.264/H.265, at most 50 MB, audio track (if any) AAC or MP3, sides 256-5760 px, aspect ratio 0.4-2.5, 23.976-60 fps, each 2-15 s, combined at most 15 s |
| `audio_url` | `audio_url.url` | `reference_audio` | up to 3; WAV/MP3, at most 15 MB, each 2-15 s, combined at most 15 s |

At most 12 reference items in total. When any reference video is present, the maximum image/video combination is 3 videos plus 4 images; their combined reference-video duration must still be at most 15 seconds. Over-limit counts return HTTP 400 before media validation or task creation, including on the test-pool mirrors. Reference audio needs at least one reference image or video. Reference video/audio requires an explicit, non-adaptive `ratio` and cannot be combined with `first_frame`/`last_frame`. Each URL is a public HTTP(S) URL or a Base64 data URL. Inline data is bounded by the 64 MB request body: Base64 adds about 33%, so the largest inline video is about 45 MB; use https URLs for large media. Inline media is validated at submit (HTTP 400). URL media is validated when the platform fetches it; a task whose media is rejected then ends `failed` with `error.code=input_media_rejected` and `moderation_status=not_started`, and is not charged.

The same count limits apply to public `/v1/videos` requests for `causyn-1.1`, with violations rejected before moderation admission.

On the direct endpoint the text is passed unchanged, so bind media in your prompt with per-type tags `<Picture N>`, `<Video N>` and `<Audio N>`. Each type is numbered separately from 1, in the order the items appear in `content`.

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "Replace the dancer in <Video 1> with the person in <Picture 1>, keeping the motion."},
    {"type": "image_url", "image_url": {"url": "https://your-media-host/person.png"}, "role": "reference_image"},
    {"type": "video_url", "video_url": {"url": "https://your-media-host/dance.mp4"}, "role": "reference_video"}
  ],
  "resolution": "768P", "duration": 5, "ratio": "9:16"
}
```

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "<Picture 1> sings along to <Audio 1>, camera slowly pushes in, in the style of <Video 1>."},
    {"type": "image_url", "image_url": {"url": "https://your-media-host/singer.png"}, "role": "reference_image"},
    {"type": "video_url", "video_url": {"url": "https://your-media-host/style.mp4"}, "role": "reference_video"},
    {"type": "audio_url", "audio_url": {"url": "https://your-media-host/song.mp3"}, "role": "reference_audio"}
  ],
  "resolution": "768P", "duration": 8, "ratio": "16:9"
}
```

On the Context IR prefix (`/video/minimax-h3/v2/video_generation`) the built-in rewrite measures each reference video (duration, shot cuts, timestamped keyframes) and audio (loudness timeline, onsets) before writing the prompt, and one repair call is made if its first answer fails validation. For a single-source video edit the rewritten prompt starts a new `[Shot N]` at each detected source cut. A reference media file the rewrite cannot read still yields a prompt, with fewer measured facts.

Direct text is passed unchanged: plain language and caller-authored H3 structure are both accepted. Clients cannot send `prompt_processing`, provider routing, queue selection or moderation bypass flags. The IR path performs the existing built-in rewrite after input admission. Standalone rewriting is available only at `/video/minimax-h3/v2/h3_context_ir`; it uses `model: MiniMax-H3` (or the legacy `minimax-h3`) and the existing content/duration/ratio/callback contract without a video resolution field.

Queries and lists consistently return `model: MiniMax-H3`. Task objects carry `id`, `model`, `status`, `created_at`, `updated_at` (unix seconds of the last status change; a platform view that does not report it falls back to `created_at`, the only real timestamp then known), `resolution`, `duration`, `ratio`, `content` (once succeeded) and `usage` (once succeeded). Public task states are `queued`, `running`, `succeeded`, `failed` and `cancelled`. `moderation_status` and `generation_status` explain review and generation separately. Pending or rejected media has no `content` URL. Approved video is at `task.content.url`; completed video usage includes `total_seconds`, `output_seconds`, `input_image_count` (excludes video and audio references) and the extension counters `input_video_count` / `input_audio_count`. Official `input_seconds` (reference-video seconds) is `0` when no reference video was sent and otherwise reported as measured by the platform; `input_audio_seconds` is present only when reference audio was measured. Unmeasured seconds are omitted, never estimated. A `h3_context_ir` task reports `content.prompt` and token usage (`total_tokens`, `prompt_tokens`, `completion_tokens`) once succeeded. Signed URLs expire and may be refreshed by querying with the same key.

List parameters are `page_num` (starting at 1), `page_size` (1–100), `filter.model` (`MiniMax-H3` or the legacy `minimax-h3`), `filter.status`, repeated `filter.task_ids`, and `filter.task_type`. Credential and namespace filters apply before totals and pagination. A task created in the direct namespace cannot be queried or deleted through the IR namespace, or vice versa. Use `Idempotency-Key` to avoid duplicate submission; reusing it for different content or a different namespace is rejected.

Deletion cancels a generation before provider submission. Once submission starts, deletion returns HTTP 409 and does not terminate the GPU process. Terminal tasks may be hidden from public query/list while retaining internal audit and billing records. Other keys and namespaces receive 404.

The facade requires durable moderation admission. Input media is reviewed by the independent CPU moderation queue before GPU dispatch. Output is private until review completes; output review starts after the GPU generation worker has reported its result and released its slot. Review retries and manual decisions do not occupy GPU workers. An explicit policy exemption is reported as `bypassed`, never as `approved`.

Errors use `{"type":"error","error":{"type":"...","message":"...","http_code":"400","code":"...","retryable":false},"request_id":"..."}`; `code` and `retryable` are extensions. `error.type` follows the official set:

| HTTP | `error.type` |
| --- | --- |
| 400 | `bad_request_error` |
| 401 | `authorized_error` |
| 402 | `insufficient_balance_error` |
| 422 | `unprocessable_entity_error` |
| 429 | `rate_limit_error` (a full direct queue keeps `code: insufficient_compute`, with `Retry-After`) |
| 500 | `server_error` |
| 503 | `overloaded_error` (HTTP status stays 503, with `Retry-After`) |
| 403 / 404 / 405 / 409 | `authorized_error` / `not_found_error` / `method_not_allowed_error` / `conflict_error` |

## Changes (2026-10-23)

For integrators already calling these endpoints:

- The model is reported as `MiniMax-H3` in query and list responses. Requests accept both `MiniMax-H3` and the legacy `minimax-h3`, including `filter.model` and `h3_context_ir`. `MiniMax-H3-Max` is rejected (400).
- `error.type` was renamed to the official set: 503 is now `overloaded_error` (was `service_unavailable_error`) and a full-queue 429 is now `rate_limit_error` (was `insufficient_compute_error`). `error.code` (still `insufficient_compute` for a full queue), `retryable` and the HTTP status are unchanged.
- `content` is omitted from a succeeded task when there is no URL or prompt (it was `{"url": null}`).
- `duration: 4` is rejected up front (400) on all prefixes, including `/video/minimax-h3/v2/video_generation` (it used to pass request validation and fail later) and `h3_context_ir`.
- New: `updated_at` on every task, `usage.input_seconds` / `usage.input_audio_seconds` where measured, token usage and `content.prompt` on `h3_context_ir` tasks, and the optional `extra.prompt_expansion_mode` request field.
