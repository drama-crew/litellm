# MiniMax H3 create and query API

The proxy accepts the MiniMax H3 JSON protocol at `POST /v2/video_generation` and `GET /v2/query/video_generation/{task_id}`. Send a normal proxy API key as `Authorization: Bearer …`. The key must allow `hailuo-h3` for `MiniMax-H3`, or `hailuo-h3-max` for `MiniMax-H3-Max`. Creation and retrieval use the existing video router, billing hooks and deployment selection

This adapter covers text-to-video, first-frame-to-video and first/last-frame-to-video. `MiniMax-H3` also maps reference image/video/audio content to the backend's reference mode; `MiniMax-H3-Max` rejects reference mode. A request containing only a last frame, or a `callback_url`, currently returns HTTP 422 before submitting a paid task. These are backend capability limits, not silently ignored fields

The request requires `model`, `content`, `resolution` and integer `duration`. H3 accepts `768P`/`2K` and 4–15 seconds; Max accepts `480P`/`768P` and 5–15 seconds. The provider may impose additional admission limits. Exactly one nonempty text item is required, at most 7000 characters. For text-only input, specify a concrete ratio such as `16:9`; image conditioning uses `adaptive` even when another valid ratio is supplied

```json
{
  "model": "MiniMax-H3-Max",
  "content": [
    {"type": "text", "text": "A porcelain teacup beside a brass radio in a moonlit teahouse. Steam rises slowly."},
    {"type": "image_url", "image_url": {"url": "https://media.example/first.png"}, "role": "first_frame"},
    {"type": "image_url", "image_url": {"url": "https://media.example/last.png"}, "role": "last_frame"}
  ],
  "resolution": "768P",
  "duration": 8,
  "ratio": "adaptive"
}
```

For a first-frame request, omit the last-frame item. For text-only generation, omit both image items and set a concrete ratio. Image roles are explicit, so first/last ordering in the JSON array does not change their meaning. Public HTTP(S) URLs and Base64 data URLs are accepted; private MiniMax `mm_file` identifiers are not portable to this backend. The JSON request body is limited to 64 MiB

Creation returns `{"task_id":"h3_task_…"}`. Query with the same API key; IDs are authenticated, encrypted, owner-bound and expire after seven days. Another key cannot use the ID to retrieve the task, even if it can access the same model. Polling an existing task does not create another video

Query returns `{"task":{…}}` with `id`, `model`, `status`, `created_at`, `resolution`, `duration`, `task_type` and `modality`. Status is `queued`, `running`, `succeeded`, `failed` or `cancelled`. A completed task includes `content.url` and usage. Adaptive ratios and completion timestamps are omitted when the backend has not supplied them; requested ratios that were ignored are never reported as measured output geometry

Errors use `{"type":"error","error":{"type":"…","message":"…","http_code":"422"},"request_id":"…"}`. Provider override headers, query parameters and unknown request fields are rejected. Preserve the returned task ID and poll it; do not automatically resubmit after a network timeout because the upstream operation might already have been accepted

The task encryption key is derived from `LITELLM_VIDEO_ID_SECRET`, then `LITELLM_SALT_KEY`, then `LITELLM_MASTER_KEY`. All replicas must use the same value; changing it invalidates previously issued IDs

Protocol references: [MiniMax create](https://platform.minimax.io/docs/api-reference/video-generation-v2-create) and [MiniMax query](https://platform.minimax.io/docs/api-reference/video-generation-v2-query)
