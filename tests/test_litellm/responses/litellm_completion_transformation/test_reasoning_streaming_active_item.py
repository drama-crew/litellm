"""
Regression tests for the Responses API streaming contract when bridging a
chat-completions stream that carries reasoning_content (e.g. deepseek-reasoner).

The OpenAI Responses streaming protocol requires that every delta event refers
to an item that was already opened by a `response.output_item.added` event with
a matching item_id, and that an item is closed (`response.output_item.done`)
before a sibling item opens. Strict clients (Codex) reject deltas that arrive
without a matching active item, logging errors like "ReasoningRawContentDelta
without active item" / "OutputTextDelta without active item".

These tests drive the full async stream and assert the emitted event sequence,
which the existing single-chunk transform tests do not cover.
"""

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class _FakeAsyncStreamWrapper:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._idx = 0
        self.logging_obj = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


def _reasoning_chunk(text, cid):
    return ModelResponseStream(
        id=cid,
        created=1,
        model="deepseek-reasoner",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                finish_reason=None,
                delta=Delta(role="assistant", content="", reasoning_content=text),
            )
        ],
    )


def _content_chunk(text, cid):
    return ModelResponseStream(
        id=cid,
        created=1,
        model="deepseek-reasoner",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                finish_reason=None,
                delta=Delta(role="assistant", content=text),
            )
        ],
    )


def _make_iterator(chunks):
    return LiteLLMCompletionStreamingIterator(
        model="deepseek-reasoner",
        litellm_custom_stream_wrapper=_FakeAsyncStreamWrapper(chunks),
        request_input="Test input",
        responses_api_request={},
    )


async def _drain_through_first_text_delta(iterator, max_steps=40):
    """Drive the stream until (and including) the first output_text.delta.

    Stops before the wrapper raises StopAsyncIteration so the end-of-stream
    assembly path (stream_chunk_builder) is never exercised.
    """
    events = []
    for _ in range(max_steps):
        ev = await iterator.__anext__()
        events.append(ev)
        if getattr(ev, "type", None) == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
            break
    return events


def _is_added(ev, item_type):
    return (
        getattr(ev, "type", None) == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        and getattr(getattr(ev, "item", None), "type", None) == item_type
    )


@pytest.mark.asyncio
async def test_reasoning_deltas_share_active_item_id():
    chunks = [
        _reasoning_chunk("First I consider ", "r1"),
        _reasoning_chunk("then I decide.", "r2"),
        _content_chunk("Hello", "c1"),
    ]
    iterator = _make_iterator(chunks)

    events = await _drain_through_first_text_delta(iterator)

    reasoning_added = [e for e in events if _is_added(e, "reasoning")]
    reasoning_deltas = [
        e
        for e in events
        if getattr(e, "type", None)
        == ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA
    ]

    assert reasoning_added, "expected a reasoning response.output_item.added event"
    assert (
        len(reasoning_deltas) >= 2
    ), "expected one reasoning delta per reasoning chunk"

    active_item_id = reasoning_added[0].item.id
    for delta in reasoning_deltas:
        assert delta.item_id == active_item_id, (
            "reasoning delta item_id "
            f"{delta.item_id!r} does not match the active reasoning item "
            f"{active_item_id!r}; strict Responses clients reject this as "
            "'ReasoningRawContentDelta without active item'"
        )


@pytest.mark.asyncio
async def test_message_text_delta_has_preceding_output_item_added():
    chunks = [
        _reasoning_chunk("thinking...", "r1"),
        _content_chunk("Hello", "c1"),
        _content_chunk(" world", "c2"),
    ]
    iterator = _make_iterator(chunks)

    events = await _drain_through_first_text_delta(iterator)

    text_delta_index = next(
        i
        for i, e in enumerate(events)
        if getattr(e, "type", None) == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    )
    text_delta = events[text_delta_index]

    message_added = [e for e in events[:text_delta_index] if _is_added(e, "message")]

    assert message_added, (
        "output_text.delta emitted without a preceding message "
        "response.output_item.added; strict Responses clients reject this as "
        "'OutputTextDelta without active item'"
    )
    assert text_delta.item_id == message_added[0].item.id


@pytest.mark.asyncio
async def test_reasoning_item_closes_before_message_item_opens():
    chunks = [
        _reasoning_chunk("thinking...", "r1"),
        _content_chunk("Hello", "c1"),
    ]
    iterator = _make_iterator(chunks)

    events = await _drain_through_first_text_delta(iterator)

    reasoning_done_idx = next(
        i
        for i, e in enumerate(events)
        if getattr(e, "type", None) == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        and getattr(getattr(e, "item", None), "type", None) == "reasoning"
    )
    message_added_idx = next(
        i for i, e in enumerate(events) if _is_added(e, "message")
    )

    assert reasoning_done_idx < message_added_idx, (
        "reasoning response.output_item.done must precede the message "
        "response.output_item.added; items cannot remain open while a sibling "
        "item is opened"
    )
