# MiniMax H3 video API

API base: `https://api.causyn.cn`.

| Processing | Prefix | Public model |
| --- | --- | --- |
| Built-in Context IR then generation | `/video/minimax-h3` | `minimax-h3` |
| Caller prompt unchanged | `/video/minimax-h3/direct` | `minimax-h3` |

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
  "model": "minimax-h3",
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

For reference mode, use 1–9 images with `role: reference_image` and an explicit ratio. Text-only requests also require an explicit ratio: `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, or `9:16`. First-frame or first/last-frame requests use adaptive canvas sizing. Exactly one non-empty text item is required. Duration is an integer from 4 through 15 seconds. Current deployment supports 768P with generated audio; callbacks, last-frame-only requests, mixed keyframe/reference input, silent output and extra fields are rejected before submission.

### Reference video and audio (direct endpoint only)

`/video/minimax-h3/direct/v2/video_generation` also accepts these content items, in addition to `text` and `image_url`:

| `type` | field | `role` | limits |
|---|---|---|---|
| `image_url` | `image_url.url` | `reference_image` | up to 9; JPEG/PNG/WEBP |
| `video_url` | `video_url.url` | `reference_video` | up to 3; MP4/MOV, H.264/H.265, at most 50 MB, sides 256-5760 px, aspect ratio 0.4-2.5, 23.976-60 fps, each 2-15 s, combined at most 15 s |
| `audio_url` | `audio_url.url` | `reference_audio` | up to 3; WAV/MP3, at most 15 MB, each 2-15 s, combined at most 15 s |

At most 12 reference items in total. Reference audio needs at least one reference image or video. Reference video/audio requires an explicit, non-adaptive `ratio` and cannot be combined with `first_frame`/`last_frame`. Each URL is a public HTTP(S) URL or a Base64 data URL. The same limits are enforced at submit (inline data) and again when the media is fetched; a violation fails the task before generation and is not charged.

Because direct text is passed unchanged, bind media in your prompt with per-type tags `<Picture N>`, `<Video N>` and `<Audio N>`. Each type is numbered separately from 1, in the order the items appear in `content`.

```json
{
  "model": "minimax-h3",
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
  "model": "minimax-h3",
  "content": [
    {"type": "text", "text": "<Picture 1> sings along to <Audio 1>, camera slowly pushes in, in the style of <Video 1>."},
    {"type": "image_url", "image_url": {"url": "https://your-media-host/singer.png"}, "role": "reference_image"},
    {"type": "video_url", "video_url": {"url": "https://your-media-host/style.mp4"}, "role": "reference_video"},
    {"type": "audio_url", "audio_url": {"url": "https://your-media-host/song.mp3"}, "role": "reference_audio"}
  ],
  "resolution": "768P", "duration": 8, "ratio": "16:9"
}
```

The Context IR prefix (`/video/minimax-h3/v2/video_generation`) rejects video and audio items with `reference video and audio are supported on /video/minimax-h3/direct only`.

Direct text is passed unchanged: plain language and caller-authored H3 structure are both accepted. Clients cannot send `prompt_processing`, provider routing, queue selection or moderation bypass flags. The IR path performs the existing built-in rewrite after input admission. Standalone rewriting is available only at `/video/minimax-h3/v2/h3_context_ir`; it uses `model: minimax-h3` and the existing content/duration/ratio/callback contract without a video resolution field.

Queries and lists consistently return `model: minimax-h3`. Public task states are `queued`, `running`, `succeeded`, `failed` and `cancelled`. `moderation_status` and `generation_status` explain review and generation separately. Pending or rejected media has no `content` URL. Approved video is at `task.content.url`; completed video usage includes `total_seconds`, `output_seconds`, `input_image_count`, `input_video_count` and `input_audio_count` (image count excludes video and audio references). Signed URLs expire and may be refreshed by querying with the same key.

List parameters are `page_num` (starting at 1), `page_size` (1–100), `filter.model=minimax-h3`, `filter.status`, repeated `filter.task_ids`, and `filter.task_type`. Credential and namespace filters apply before totals and pagination. A task created in the direct namespace cannot be queried or deleted through the IR namespace, or vice versa. Use `Idempotency-Key` to avoid duplicate submission; reusing it for different content or a different namespace is rejected.

Deletion cancels a generation before provider submission. Once submission starts, deletion returns HTTP 409 and does not terminate the GPU process. Terminal tasks may be hidden from public query/list while retaining internal audit and billing records. Other keys and namespaces receive 404.

The facade requires durable moderation admission. Input media is reviewed by the independent CPU moderation queue before GPU dispatch. Output is private until review completes; output review starts after the GPU generation worker has reported its result and released its slot. Review retries and manual decisions do not occupy GPU workers. An explicit policy exemption is reported as `bypassed`, never as `approved`.

Errors use `{"type":"error","error":{"type":"...","message":"...","http_code":"400"},"request_id":"..."}`.
