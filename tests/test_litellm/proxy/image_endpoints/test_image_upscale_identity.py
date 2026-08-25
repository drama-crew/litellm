"""Paid image-upscale billing identity must match drama's actual identity model.

drama represents an organization as a LiteLLM **team** (team_id = the Keycloak
org subject; for a personal org it equals the user id). It never creates
LiteLLM ``organization`` rows -- production has zero of them -- so every
drama-minted project key carries team_id + user_id and ``organization_id:
null``.

Requiring a non-empty ``org_id`` therefore rejected 100% of drama's paid image
upscales with "requires complete billing identity", before any provider call.
Attribution is fully determined by (team_id, api_key, user_id); org_id was
redundant with team_id and unsatisfiable in practice.
"""

from types import SimpleNamespace

import pytest

from litellm.llms.libtv.image_upscale import ImageUpscaleSubmitter
from litellm.proxy.image_endpoints.endpoints import (
    _has_complete_paid_image_upscale_identity,
)


def _auth(**kw):
    base = {
        "team_id": "team-1",
        "api_key": "sk-1",
        "user_id": "user-1",
        "org_id": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_identity_is_complete_without_an_organization() -> None:
    """A drama project key (org_id null) must be accepted."""
    assert _has_complete_paid_image_upscale_identity(_auth()) is True


def test_identity_still_accepts_an_explicit_organization() -> None:
    assert _has_complete_paid_image_upscale_identity(_auth(org_id="org-1")) is True


@pytest.mark.parametrize('missing', ['team_id', 'api_key', 'user_id'])
def test_identity_still_fails_closed_without_team_key_or_user(missing: str) -> None:
    """The three fields that actually carry attribution stay mandatory."""
    assert _has_complete_paid_image_upscale_identity(_auth(**{missing: None})) is False
    assert _has_complete_paid_image_upscale_identity(_auth(**{missing: '  '})) is False


@pytest.mark.asyncio
async def test_submitter_accepts_payload_without_organization_id() -> None:
    """The handler-side guard must agree with the endpoint-side one.

    Without this the request passes the endpoint and then dies one layer deeper
    with the same message, which is exactly what production did.
    """
    class _Provider:
        async def create(self, payload):
            return {"task_id": "provider-task-1"}

    submitter = ImageUpscaleSubmitter(("deployment-1", _Provider()))
    receipt = await submitter.submit(
        {
            "request_id": "gen-1",
            "team_id": "team-1",
            "api_key": "sk-1",
            "user_id": "user-1",
            # organization_id deliberately absent -- drama never sets it
            "response_cost": 0.46,
            "source_url": "https://example.com/a.png",
            "style": "Standard V2",
            "scale": 2,
        }
    )
    assert receipt.message != "paid image upscale requires complete billing identity"


# ── source registration parity (sync vs async) ───────────────────────────────


def test_async_image_upscale_registers_the_source_with_libtv() -> None:
    """The async path must register the source, exactly like the sync one.

    libtv does not accept an arbitrary URL: every working path (frames2video,
    mixed2video, video upscale) first turns the source into a libtv-hosted
    reference via ensure_libtv_url / aensure_libtv_url. The sync image-upscale
    path did this; the async path -- the one the proxy actually uses -- passed
    the caller's presigned URL straight through, so production rejected every
    submit with an upstream "invalid target URL" and the receipt stayed
    not_submitted (fail-closed, nothing billed, but the feature was unusable).

    Structural: driving the real path needs a live libtv client, and what must
    not regress is that the async branch calls the registration helper at all.
    """
    import ast
    import inspect

    from litellm.llms.libtv import handler as libtv_handler

    src = inspect.getsource(libtv_handler.LibTVLLM.asubmit_image_upscale)
    tree = ast.parse(inspect.cleandoc(src))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "aensure_libtv_url" in called, (
        "asubmit_image_upscale never registers the source with libtv -- it "
        "forwards the caller's raw URL, which libtv rejects as an invalid "
        "target URL"
    )
