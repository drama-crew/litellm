CAUSYN_REFERENCE_IMAGE_MAX_WITH_VIDEO = 4


def reference_count_limit_violation(image_count: int, video_count: int, audio_count: int) -> str | None:
    if image_count > 9 or video_count > 3 or audio_count > 3:
        return "reference count exceeds 9 images, 3 videos or 3 audio clips"
    if image_count + video_count + audio_count > 12:
        return "at most 12 reference items are allowed"
    if video_count > 0 and image_count > CAUSYN_REFERENCE_IMAGE_MAX_WITH_VIDEO:
        return (
            f"at most {CAUSYN_REFERENCE_IMAGE_MAX_WITH_VIDEO} reference images are allowed "
            "when reference video is present"
        )
    if audio_count > 0 and image_count == 0 and video_count == 0:
        return "reference audio requires at least one reference image or video"
    return None
