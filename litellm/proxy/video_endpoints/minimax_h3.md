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

For reference mode, use 1–9 images with `role: reference_image` and an explicit ratio. Text-only requests also require an explicit ratio: `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, or `9:16`. First-frame or first/last-frame requests use adaptive canvas sizing. Exactly one non-empty text item is required. Duration is an integer from 4 through 15 seconds. Current deployment supports 768P with generated audio; callbacks, last-frame-only requests, video/audio references, mixed keyframe/reference input, silent output and extra fields are rejected before submission.

Direct text is passed unchanged: plain language and caller-authored H3 structure are both accepted. Clients cannot send `prompt_processing`, provider routing, queue selection or moderation bypass flags. The IR path performs the existing built-in rewrite after input admission. Standalone rewriting is available only at `/video/minimax-h3/v2/h3_context_ir`; it uses `model: minimax-h3` and the existing content/duration/ratio/callback contract without a video resolution field.

Queries and lists consistently return `model: minimax-h3`. Public task states are `queued`, `running`, `succeeded`, `failed` and `cancelled`. `moderation_status` and `generation_status` explain review and generation separately. Pending or rejected media has no `content` URL. Approved video is at `task.content.url`; completed video usage includes `total_seconds`, `output_seconds` and `input_image_count`. Signed URLs expire and may be refreshed by querying with the same key.

List parameters are `page_num` (starting at 1), `page_size` (1–100), `filter.model=minimax-h3`, `filter.status`, repeated `filter.task_ids`, and `filter.task_type`. Credential and namespace filters apply before totals and pagination. A task created in the direct namespace cannot be queried or deleted through the IR namespace, or vice versa. Use `Idempotency-Key` to avoid duplicate submission; reusing it for different content or a different namespace is rejected.

Deletion cancels a generation before provider submission. Once submission starts, deletion returns HTTP 409 and does not terminate the GPU process. Terminal tasks may be hidden from public query/list while retaining internal audit and billing records. Other keys and namespaces receive 404.

The facade requires durable moderation admission. Input media is reviewed by the independent CPU moderation queue before GPU dispatch. Output is private until review completes; output review starts after the GPU generation worker has reported its result and released its slot. Review retries and manual decisions do not occupy GPU workers. An explicit policy exemption is reported as `bypassed`, never as `approved`.

Errors use `{"type":"error","error":{"type":"...","message":"...","http_code":"400"},"request_id":"..."}`.
