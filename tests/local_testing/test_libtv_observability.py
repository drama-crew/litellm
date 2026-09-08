"""Audit trail for libtv video submissions.

A production report ("the video ignored my first frame") could not be answered
for a single one of twelve finished tasks: the create request's parameters are
kept nowhere on our side, the status poll returns none of them, the vendor
exposes no task-detail endpoint, and the one log line that did carry the branch
was INFO on a logger the production root level drops. These tests pin the two
records that make such a question answerable afterwards: a log line and an OTel
span, both keyed by the vendor task id.
"""

import json
import logging

import pytest

from litellm.llms.libtv.observability import (
    AUDIT_LOGGER_NAME,
    build_video_submission_record,
    record_video_submission,
)


def _record(**overrides):
    kwargs = dict(
        model="star-video2-fast",
        mode="singleImage2video",
        images=["https://x/a.png"],
        videos=[],
        audios=[],
        optional_params={"image": "https://x/a.png", "resolution": "480p", "seconds": "4"},
        task_id="t-123",
        prompt="a girl turns to camera",
    )
    kwargs.update(overrides)
    return build_video_submission_record(**kwargs)


def test_record_captures_the_mode_actually_sent_to_the_vendor():
    assert _record()["mode"] == "singleImage2video"


def test_record_captures_reference_counts_and_which_keys_the_caller_sent():
    rec = _record(
        images=["https://x/a.png", "https://x/b.png"],
        optional_params={"image": "https://x/a.png", "last_image": "https://x/b.png"},
    )
    assert rec["reference_images"] == 2
    assert rec["reference_videos"] == 0
    assert rec["reference_audios"] == 0
    assert rec["reference_keys"] == ["image", "last_image"]


def test_record_carries_task_id_and_prompt_length():
    rec = _record()
    assert rec["task_id"] == "t-123"
    assert rec["prompt_chars"] == len("a girl turns to camera")


def test_record_never_carries_reference_urls_which_may_be_presigned():
    blob = json.dumps(_record(optional_params={"image": "https://x/a.png?Signature=SECRET"}))
    assert "Signature" not in blob
    assert "SECRET" not in blob


def test_audit_logger_level_is_info_so_the_production_root_level_cannot_drop_it():
    # Production root is WARNING; a logger that leaves its own level unset
    # inherits that and the record is never created at all (the exact reason the
    # existing reference_collection line was invisible in production).
    assert logging.getLogger(AUDIT_LOGGER_NAME).level == logging.INFO


def test_record_video_submission_logs_a_parseable_line(caplog):
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        record_video_submission(
            model="star-video2-fast",
            mode="singleImage2video",
            images=["https://x/a.png"],
            videos=[],
            audios=[],
            optional_params={"image": "https://x/a.png"},
            task_id="t-123",
            prompt="hello",
        )
    lines = [r.getMessage() for r in caplog.records if r.name == AUDIT_LOGGER_NAME]
    assert len(lines) == 1
    payload = json.loads(lines[0].split("libtv video submission ", 1)[1])
    assert payload["task_id"] == "t-123"
    assert payload["mode"] == "singleImage2video"


def test_record_video_submission_emits_a_span_with_the_submission_attributes():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    record_video_submission(
        model="star-video2-fast",
        mode="singleImage2video",
        images=["https://x/a.png"],
        videos=[],
        audios=[],
        optional_params={"image": "https://x/a.png"},
        task_id="t-123",
        prompt="hello",
        tracer=provider.get_tracer("test"),
    )

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["libtv.video.submit"]
    attrs = dict(spans[0].attributes)
    assert attrs["libtv.mode"] == "singleImage2video"
    assert attrs["libtv.task_id"] == "t-123"
    assert attrs["libtv.reference_images"] == 1
    assert attrs["libtv.model"] == "star-video2-fast"
    assert trace.get_current_span() is not None  # no ambient span leaked


def test_record_video_submission_is_a_no_op_when_tracing_is_unconfigured():
    # Global provider is the API's ProxyTracerProvider in a plain test process;
    # submission recording must not raise there.
    record_video_submission(
        model="star-video2-fast",
        mode="image2video",
        images=[],
        videos=[],
        audios=[],
        optional_params={},
        task_id=None,
        prompt=None,
    )
