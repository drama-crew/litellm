"""The test worker pool facade needs an explicit grant; production H3 prefixes behave as before."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks

TEST_ROUTES = [
    "/video/minimax-h3-test/v2/video_generation",
    "/video/minimax-h3-test/direct/v2/video_generation",
    "/video/minimax-h3-test/v2/query/video_generation/{video_id}",
    "/video/minimax-h3-test/direct/v2/query/video_generation/mod_video_1",
    "/video/minimax-h3-test/v2/query/video_generation",
    "/video/minimax-h3-test/direct/v2/video_generation/{video_id}",
    "/video/minimax-h3-test/v2/h3_context_ir",
]
PROD_ROUTES = [
    "/video/minimax-h3/v2/video_generation",
    "/video/minimax-h3/direct/v2/video_generation",
    "/video/minimax-h3/v2/query/video_generation/{video_id}",
]


def call(route, allowed_routes, **kwargs):
    return RouteChecks.should_call_route(
        route=route, valid_token=UserAPIKeyAuth(api_key="sk-x", allowed_routes=allowed_routes, **kwargs)
    )


@pytest.mark.parametrize("route", TEST_ROUTES)
@pytest.mark.parametrize("allowed", [None, []])
def test_key_without_allowed_routes_cannot_reach_the_test_prefix(route, allowed):
    with pytest.raises(HTTPException) as error:
        call(route, allowed)
    assert error.value.status_code == 403


@pytest.mark.parametrize("route", TEST_ROUTES)
@pytest.mark.parametrize(
    "allowed",
    [
        PROD_ROUTES,
        ["/video/minimax-h3"],
        ["llm_api_routes"],
        ["openai_routes"],
        ["/video/*"],
        ["/*"],
        ["/video"],
        ["/video/minimax-h3-testing"],
        PROD_ROUTES + ["llm_api_routes"],
    ],
)
def test_production_grants_and_groups_do_not_open_the_test_prefix(route, allowed):
    with pytest.raises(HTTPException) as error:
        call(route, allowed)
    assert error.value.status_code == 403


@pytest.mark.parametrize("route", TEST_ROUTES)
@pytest.mark.parametrize(
    "allowed",
    [
        ["/video/minimax-h3-test"],
        ["/video/minimax-h3-test/*"],
        ["/video/minimax-h3-test/v2/video_generation", "/video/minimax-h3-test/direct/v2/video_generation"]
        + [r for r in TEST_ROUTES],
        PROD_ROUTES + ["/video/minimax-h3-test/*"],
        ["llm_api_routes", "/video/minimax-h3-test"],
    ],
)
def test_explicit_test_grant_opens_the_test_routes(route, allowed):
    assert call(route, allowed) is True


@pytest.mark.parametrize("route", PROD_ROUTES)
@pytest.mark.parametrize("allowed", [None, [], PROD_ROUTES, ["llm_api_routes"], ["openai_routes"], ["/video/*"]])
def test_production_prefixes_are_unchanged_for_every_key_shape(route, allowed):
    assert call(route, allowed) is True


def test_a_key_scoped_to_test_routes_only_does_not_reach_production():
    with pytest.raises(HTTPException):
        call("/video/minimax-h3/v2/video_generation", ["/video/minimax-h3-test/*"])


def test_proxy_admin_is_not_blocked():
    assert call(TEST_ROUTES[0], None, user_role=LitellmUserRoles.PROXY_ADMIN) is True
