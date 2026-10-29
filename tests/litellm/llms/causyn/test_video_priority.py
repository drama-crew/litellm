from copy import deepcopy

import pytest

from litellm.llms.causyn.video_priority import MAX_PRIORITY, PRIORITY_ENV, estimated_priority, priority_route

BASE = "worker:tasks:video_generate_ref2va"


@pytest.mark.parametrize("duration", [5, 5.0])
def test_numeric_duration_and_media_weights_rank_expensive_inputs(duration):
    plain = {"duration_seconds": duration}
    image = {"duration_seconds": duration, "references": [{"media_type": "image"}]}
    audio = {"duration_seconds": duration, "references": [{"media_type": "audio"}]}
    video = {"duration_seconds": duration, "references": [{"media_type": "video"}]}
    assert 0 < estimated_priority(plain) < estimated_priority(image) < estimated_priority(audio)
    assert estimated_priority(audio) < estimated_priority(video) < estimated_priority({"duration_seconds": 15})


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"duration_seconds": True},
        {"duration_seconds": "5"},
        {"duration_seconds": float("nan")},
        {"duration_seconds": float("inf")},
        {"duration_seconds": 0},
        {"duration_seconds": 16},
        {"duration_seconds": 5, "references": [{"media_type": "unknown"}]},
    ],
)
def test_unscorable_input_uses_legacy_routing_without_rejecting_or_mutating(payload, monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "true")
    assert estimated_priority(payload) == 0
    route = priority_route(BASE, model="causyn-1.1", request=payload)
    assert route.stream == BASE and route.priority == 0 and route.catalog_key is None


@pytest.mark.parametrize("enabled", [None, "false", "0", "yes", ""])
def test_priority_requires_explicit_enablement(enabled, monkeypatch):
    if enabled is None:
        monkeypatch.delenv(PRIORITY_ENV, raising=False)
    else:
        monkeypatch.setenv(PRIORITY_ENV, enabled)
    route = priority_route(BASE, model="causyn-1.1", request={"duration_seconds": 15})
    assert route.stream == BASE and route.catalog_key is None


@pytest.mark.parametrize("enabled", ["true", "1", "TRUE"])
def test_enabled_route_uses_only_input_features_and_preserves_the_request(enabled, monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, enabled)
    request = {
        "duration_seconds": 15,
        "prompt": "original",
        "references": [{"media_type": "video", "url": "https://example.invalid/ref.mp4", "role": "reference"}],
    }
    original = deepcopy(request)
    route = priority_route(BASE, model="causyn-1.1", request=request)
    alternate = {**request, "prompt": "different", "task_id": "unrelated-id", "output_hash": "other"}
    assert request == original
    assert route == priority_route(BASE, model="causyn-1.1", request=alternate)
    assert route.priority == estimated_priority(request)
    assert route.stream == f"{BASE}:cost:{route.priority}"
    assert route.catalog_key == "worker:priority-streams:video_generate_ref2va"
    assert 0 < route.priority <= MAX_PRIORITY


@pytest.mark.parametrize(
    "stream,model,pool",
    [
        ("worker:tasks:video_generate_test", "causyn-1.1", "test"),
        (BASE, "causyn-1.1", "test"),
        (BASE, "other-model", None),
        ("worker:tasks:media_transfer", "causyn-1.1", None),
        (BASE + ":cost:100", "causyn-1.1", None),
    ],
)
def test_other_models_pools_and_noncanonical_streams_keep_their_original_route(stream, model, pool, monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "1")
    route = priority_route(stream, model=model, pool=pool, request={"duration_seconds": 15})
    assert route.stream == stream and route.priority == 0 and route.catalog_key is None


def test_valid_duration_grid_scores_increase_and_billed_15_seconds_matches_the_same_grid():
    priorities = [estimated_priority({"duration_seconds": duration}) for duration in (5, 10, 15)]
    assert priorities == sorted(set(priorities))
    assert priorities[-1] == estimated_priority({"duration_seconds": 345 / 24})
