"""Every place that enumerates the H3 routes must also know the test-pool mirror."""

from __future__ import annotations

import pytest

from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.auth import auth_utils
from litellm.proxy.spend_tracking.budget_reservation import estimate_request_max_cost
from litellm.proxy.video_endpoints import moderation_bridge as bridge
from litellm.proxy.video_endpoints.minimax_h3_paths import ALL_PREFIXES, PREFIXES, TEST_PREFIXES, namespace


def _h3_routes(routes, marker):
    return {route for route in routes if route.startswith("/video/" + marker)}


def test_route_allowlist_has_a_test_mirror_of_every_production_h3_route():
    routes = set(LiteLLMRoutes.openai_routes.value)
    production = _h3_routes(routes, "minimax-h3/")
    mirrored = {route.replace("/video/minimax-h3/", "/video/minimax-h3-test/", 1) for route in production}
    assert production and mirrored == _h3_routes(routes, "minimax-h3-test/")


def test_moderation_intake_routes_cover_every_create_route():
    for prefix in ALL_PREFIXES:
        assert prefix + "/v2/video_generation" in bridge.INTAKE_ROUTES
    assert "/video/minimax-h3-test/v2/h3_context_ir" in bridge.INTAKE_ROUTES


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_auth_helpers_classify_every_prefix_alike(prefix):
    assert auth_utils._is_video_mutation_route(prefix + "/v2/video_generation", "POST")
    assert not auth_utils._is_video_mutation_route(prefix + "/v2/video_generation", "GET")
    assert auth_utils._is_video_retrieval_route(prefix + "/v2/query/video_generation/{video_id}")
    assert auth_utils._is_video_retrieval_route(prefix + "/v2/query/video_generation/mod_video_1")
    assert not auth_utils._is_video_retrieval_route(prefix + "/v2/query/video_generation/a/b")


@pytest.mark.parametrize("prefix", ALL_PREFIXES)
def test_status_list_and_cancel_routes_reserve_no_budget(prefix):
    for route in (
        prefix + "/v2/query/video_generation",
        prefix + "/v2/query/video_generation/mod_video_1",
        prefix + "/v2/video_generation/mod_video_1",
    ):
        assert estimate_request_max_cost({"model": "causyn-1.1"}, route, None) is None


def test_context_ir_reservation_matches_on_both_prefixes():
    for prefix in ("/video/minimax-h3", "/video/minimax-h3-test"):
        assert estimate_request_max_cost({}, prefix + "/v2/h3_context_ir", None) == 4.0


def test_namespaces_are_distinct_per_prefix():
    names = [namespace(prefix + "/v2/video_generation") for prefix in ALL_PREFIXES]
    assert len(set(names)) == 4 and None not in names
    assert [namespace(prefix + "/v2/video_generation") for prefix in PREFIXES] == ["minimax-h3", "minimax-h3-direct"]
    assert [namespace(prefix + "/v2/video_generation") for prefix in TEST_PREFIXES] == [
        "minimax-h3-test",
        "minimax-h3-direct-test",
    ]
    assert namespace("/video/minimax-h3-testing/v2/x") is None
