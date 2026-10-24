"""Admin-issued H3-only keys: ``models=["causyn-1.1"]`` + the ``h3_only`` route profile.

The platform (OpenHands enterprise ``server.routes.openapi_keys``) provisions these
keys through LiteLLM ``/key/generate`` with ``models`` and ``allowed_routes`` only
-- there is no LiteLLM-side enforcement code for them. These tests pin the
contract that makes that safe, against the real auth chain:

* an H3-only key reaches the H3 create / query / list / Context IR routes
  (authorization passes; whatever the harness returns afterwards is not 401/403);
* the same key is rejected (403) on ``/v1/videos`` / ``/videos``: for
  ``model=seedance-2.0-fast`` by the model check (it runs first), and for the permitted
  ``causyn-1.1`` by the route allowlist -- so neither model nor route can be used to
  reach the generic video API;
* a key limited to ``causyn-1.1`` but granted *broad* routes is still rejected for
  seedance by the model check, so the model restriction is independently enforced.
"""

from __future__ import annotations

import pytest

from litellm.proxy._types import LiteLLM_VerificationTokenView, hash_token
from litellm.proxy.video_endpoints import minimax_h3_endpoints  # noqa: F401  (routes under test)

import litellm.proxy.proxy_server as proxy_server

from tests.test_litellm.proxy.video_endpoints.test_causyn_public_stack import (  # noqa: F401
    _auth_headers,
    stack,
)

# Mirrors OpenHands enterprise ``server.routes.openapi_keys.H3_ONLY_ROUTES``
# (namespaced create / query / list / delete on both prefixes + Context IR).
_H3_SUFFIXES = (
    "/v2/video_generation",
    "/v2/query/video_generation/{video_id}",
    "/v2/query/video_generation",
    "/v2/video_generation/{video_id}",
)
H3_ONLY_ROUTES = [
    prefix + suffix for prefix in ("/video/minimax-h3", "/video/minimax-h3/direct") for suffix in _H3_SUFFIXES
] + ["/video/minimax-h3/v2/h3_context_ir"]

# Mirrors the self-service ("all") default grant for the video routes.
BROAD_ROUTES = [
    "/v1/models",
    "/models",
    "/videos",
    "/v1/videos",
    "/videos/{video_id}",
    "/v1/videos/{video_id}",
    "/videos/{video_id}/content",
    "/v1/videos/{video_id}/content",
    *H3_ONLY_ROUTES,
]

H3_ONLY_KEY = "sk-h3-only"
MODEL_ONLY_KEY = "sk-causyn-model-only-broad-routes"
SEEDANCE = "seedance-2.0-fast"
OTHER_KEY = "sk-other-model"


class _ScopedStore:
    """Virtual-key rows shaped like what ``/key/generate`` stores for admin keys."""

    def __init__(self) -> None:
        self.rows = {
            hash_token(H3_ONLY_KEY): {"models": ["causyn-1.1"], "allowed_routes": list(H3_ONLY_ROUTES)},
            hash_token(MODEL_ONLY_KEY): {"models": ["causyn-1.1"], "allowed_routes": list(BROAD_ROUTES)},
            # Control: an ordinary key for a different model, never allowed on these routes.
            hash_token(OTHER_KEY): {"models": ["other-model"], "allowed_routes": []},
        }

    async def get_data(self, *, token: str, table_name: str, **_: object):
        assert table_name == "combined_view"
        row = self.rows.get(token)
        if row is None:
            return None
        # No team_id: team budget/wallet resolution needs the full proxy cache stack, which the shared
        # harness stubs out; the model and route checks under test do not depend on it.
        return LiteLLM_VerificationTokenView(
            token=token,
            metadata={"type": "admin_openapi", "openapi_key_id": "11111111-1111-1111-1111-111111111111"},
            **row,
        )


@pytest.fixture
def scoped(stack, monkeypatch):  # noqa: F811
    from litellm.proxy.video_endpoints import context_ir_endpoints

    monkeypatch.setattr(proxy_server, "prisma_client", _ScopedStore())
    stack.app.include_router(context_ir_endpoints.router)
    return stack


def _h3_create_body() -> dict[str, object]:
    return {
        "model": "minimax-h3",
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
        "content": [
            {"type": "text", "text": "Gentle waves."},
            {"type": "image_url", "role": "reference_image", "image_url": {"url": "https://source.example/0.png"}},
        ],
    }


# Messages LiteLLM's own key checks produce. A 403 from a later layer (e.g. the moderation bridge not
# finding a platform credential mapping in this harness) is *not* an authorization denial of the key.
_AUTH_DENIAL_MARKERS = ("not allowed to call this route", "not allowed to access model", "invalid_api_key")


def _passed_auth(response) -> bool:
    return not any(marker in response.text for marker in _AUTH_DENIAL_MARKERS)


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize(
    "route",
    [
        "/video/minimax-h3/v2/video_generation",
        "/video/minimax-h3/direct/v2/video_generation",
    ],
)
def test_h3_only_key_gets_through_h3_create(scoped, route):
    denied = scoped.client.post(route, json=_h3_create_body(), headers=_auth_headers(OTHER_KEY))
    assert denied.status_code == 403, denied.text  # control: auth really runs on this route
    created = scoped.client.post(route, json=_h3_create_body(), headers=_auth_headers(H3_ONLY_KEY))
    assert _passed_auth(created), created.text
    assert not scoped.state.enqueued  # harness has no Context IR store; nothing is submitted


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("prefix", ["/video/minimax-h3", "/video/minimax-h3/direct"])
def test_h3_only_key_gets_through_h3_query_list_and_delete(scoped, prefix):
    routes = [
        ("get", prefix + "/v2/query/video_generation/mod_video_1"),
        ("get", prefix + "/v2/query/video_generation"),
        ("delete", prefix + "/v2/video_generation/mod_video_1"),
    ]
    for method, route in routes:
        denied = getattr(scoped.client, method)(route, headers=_auth_headers(OTHER_KEY))
        assert denied.status_code == 403, (route, denied.text)  # control
        allowed = getattr(scoped.client, method)(route, headers=_auth_headers(H3_ONLY_KEY))
        assert _passed_auth(allowed), (route, allowed.text)


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
def test_h3_only_key_gets_through_h3_context_ir(scoped):
    route = "/video/minimax-h3/v2/h3_context_ir"
    body = {"model": "causyn-1.1", "prompt": "kite"}
    denied = scoped.client.post(route, json=body, headers=_auth_headers(OTHER_KEY))
    assert denied.status_code == 403, denied.text  # control
    allowed = scoped.client.post(route, json=body, headers=_auth_headers(H3_ONLY_KEY))
    assert _passed_auth(allowed), allowed.text


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos", "/videos"])
def test_h3_only_key_is_rejected_for_seedance_on_v1_videos(scoped, route):
    body = {"model": SEEDANCE, "prompt": "kite", "seconds": "5", "size": "720p", "aspect_ratio": "16:9"}
    result = scoped.client.post(route, json=body, headers=_auth_headers(H3_ONLY_KEY))
    assert result.status_code == 403, result.text
    assert "key not allowed to access model" in result.text and SEEDANCE in result.text
    assert not scoped.state.enqueued


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos", "/videos"])
def test_h3_only_key_is_rejected_on_v1_videos_even_for_its_own_model(scoped, route):
    """The permitted model cannot use the generic video API: the route allowlist denies it."""
    body = {"model": "causyn-1.1", "prompt": "kite", "seconds": "5", "size": "768p", "aspect_ratio": "16:9"}
    result = scoped.client.post(route, json=body, headers=_auth_headers(H3_ONLY_KEY))
    assert result.status_code == 403, result.text
    assert "not allowed to call this route" in result.text
    assert not scoped.state.enqueued


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos/some_video_id", "/v1/videos/some_video_id/content"])
def test_h3_only_key_is_rejected_on_other_video_routes(scoped, route):
    result = scoped.client.get(route, headers=_auth_headers(H3_ONLY_KEY))
    assert result.status_code == 403, result.text


@pytest.mark.parametrize("stack", ["causyn-1.1"], indirect=True)
@pytest.mark.parametrize("route", ["/v1/videos", "/videos"])
def test_model_restricted_key_with_broad_routes_is_still_rejected_for_seedance(scoped, route):
    """Routes allow /v1/videos, but ``models=["causyn-1.1"]`` blocks seedance on its own."""
    body = {"model": SEEDANCE, "prompt": "kite", "seconds": "5", "size": "720p", "aspect_ratio": "16:9"}
    result = scoped.client.post(route, json=body, headers=_auth_headers(MODEL_ONLY_KEY))
    assert result.status_code == 403, result.text
    assert "key not allowed to access model" in result.text and SEEDANCE in result.text
    assert "not allowed to call this route" not in result.text  # rejected by the model check alone
    assert not scoped.state.enqueued
    # control: the same broad-route key *is* let through for the permitted model
    ok = scoped.client.post(
        route,
        json={"model": "causyn-1.1", "prompt": "kite", "seconds": "5", "size": "768p", "aspect_ratio": "16:9"},
        headers=_auth_headers(MODEL_ONLY_KEY),
    )
    assert _passed_auth(ok), ok.text
