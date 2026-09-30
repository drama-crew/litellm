from __future__ import annotations

import pytest

import litellm.llms.causyn.handler as mod

_SPEC = mod._MODEL_SPECS[mod.CAUSYN_H3_MODEL]


@pytest.mark.parametrize(
    ("ordered", "frames"),
    [(5, 124), (10, 243), (15, 345)],
)
def test_observed_frame_counts(ordered: int, frames: int) -> None:
    assert mod._h3_rendered_frames(ordered) == frames
    assert mod._expected_rendered_seconds(ordered) == pytest.approx(frames / 24)


@pytest.mark.parametrize("ordered", range(3, 16))
def test_every_orderable_duration_reconciles_against_its_own_frame_count(ordered: int) -> None:
    frames = mod._h3_rendered_frames(ordered)
    assert frames % 17 == 5 and frames <= 345
    assert mod._duration_reconciles(ordered, frames / 24)
    assert mod._duration_reconciles(ordered, frames / 24 + 1 / 24 * 0.5)


@pytest.mark.parametrize(
    ("ordered", "rendered", "ok"),
    [
        (15, 14.375, True),
        (15, 10.125, False),
        (5, 5.0 + 1 / 6, True),
        (10, 10.125, True),
        (10, 9.0, False),
        (10, 11.0, False),
        (5, 10.125, False),
    ],
)
def test_reconcile_examples(ordered: int, rendered: float, ok: bool) -> None:
    assert mod._duration_reconciles(ordered, rendered) is ok


@pytest.mark.parametrize("ordered", range(3, 14))
def test_a_result_off_by_one_ordered_second_is_rejected(ordered: int) -> None:
    rendered = mod._h3_rendered_frames(ordered) / 24
    assert not mod._duration_reconciles(ordered, rendered + 1)
    assert not mod._duration_reconciles(ordered + 1, rendered)
