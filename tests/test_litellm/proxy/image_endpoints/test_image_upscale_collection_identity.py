"""Collecting a paid image upscale must not require an organization row.

drama models an organization as a LiteLLM *team* and never mints LiteLLM
organization rows, so every drama-minted project key carries
``organization_id: null``. The submit-side guard was already relaxed to
(team_id, api_key, user_id) for exactly this reason -- see
``_has_complete_paid_image_upscale_identity`` -- but the collection side kept
requiring ``organization_id``.

The consequence is worse than a rejected request: the task is submitted, libtv
bills for it, it completes, and then every poll answers 503 "durable image
upscale identity unavailable" forever. ``_poll_image_upscale`` is the only path
that emits the billing event and persists the terminal result
(``libtv_image_upscale_resolve`` is admin-only, requires submission_state
"unknown", and writes a tombstone rather than collecting), so the money is
spent and the result is unreachable.
"""

import json

import pytest

from litellm.llms.libtv.receipts import StoredReceipt
from litellm.proxy.image_endpoints import endpoints


def _receipt(**overrides):
    base = dict(
        team_id="team-1",
        model="topaz-image-upscaler",
        request_id="request-1",
        fingerprint="f" * 64,
        submission_state="submitted",
        deployment_id="libtv-topaz-image-upscaler-account-1",
        provider_task_id="task-1",
        resume_token="signed",
        response_cost=0.46,
        api_key="sk-1",
        user_id="user-1",
        organization_id=None,
        scale=2,
    )
    base.update(overrides)
    return StoredReceipt(**base)


def test_collection_identity_is_complete_without_an_organization():
    assert endpoints._image_upscale_collection_identity_is_complete(_receipt()) is True


def test_collection_identity_still_accepts_an_explicit_organization():
    assert endpoints._image_upscale_collection_identity_is_complete(_receipt(organization_id="org-1")) is True


@pytest.mark.parametrize("missing", ["api_key", "user_id", "team_id"])
def test_collection_identity_still_fails_closed_without_key_user_or_team(missing):
    """The three fields that actually carry attribution stay mandatory."""
    assert endpoints._image_upscale_collection_identity_is_complete(_receipt(**{missing: None})) is False
    assert endpoints._image_upscale_collection_identity_is_complete(_receipt(**{missing: ""})) is False


@pytest.mark.parametrize(
    "team_id,api_key,user_id,org_id",
    [
        ("team-1", "sk-1", "user-1", None),
        ("team-1", "sk-1", "user-1", "org-1"),
        (None, "sk-1", "user-1", "org-1"),
        ("team-1", None, "user-1", "org-1"),
        ("team-1", "sk-1", None, "org-1"),
        ("  ", "sk-1", "user-1", None),
        ("team-1", "  ", "user-1", None),
        ("team-1", "sk-1", "  ", None),
    ],
)
def test_collection_and_submission_identity_agree(team_id, api_key, user_id, org_id):
    """The two guards must never drift apart again.

    They diverged once already -- submit accepted org_id-less identities while
    collection rejected them -- which is the worst possible split: it lets the
    money out and locks the result in. Compared by behaviour, not by reading
    each other's source.
    """
    from types import SimpleNamespace

    auth = SimpleNamespace(team_id=team_id, api_key=api_key, user_id=user_id, org_id=org_id)
    receipt = _receipt(team_id=team_id, api_key=api_key, user_id=user_id, organization_id=org_id)

    assert endpoints._image_upscale_collection_identity_is_complete(
        receipt
    ) is endpoints._has_complete_paid_image_upscale_identity(auth)


def test_recovery_falls_back_to_env_credentials_for_an_unknown_deployment(monkeypatch):
    """A receipt whose deployment is not in the current pool must still be
    recoverable.

    Before the pool was fixed it was always empty, so recovery took the
    environment-credential branch. Making the pool real must not turn that
    branch into dead code: legacy receipts (including ones written under the
    old literal "unknown") and deployments removed from config would otherwise
    become permanently uncollectable -- the same failure the pool fix exists to
    end.
    """
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setenv("LIBTV_TOKEN", "env-token")
    monkeypatch.setenv("LIBTV_WEBID", "env-webid")

    pool = [{"id": "libtv-topaz-image-upscaler-account-1", "api_key": "os.environ/LIBTV_TOKEN", "webid": "os.environ/LIBTV_WEBID"}]
    monkeypatch.setattr(endpoints, "_image_upscale_deployment_pool", lambda *a, **k: pool)

    client = endpoints._client_for_receipt(_receipt(deployment_id="unknown"))
    assert client.token == "env-token"


def test_recovery_prefers_the_exact_pool_credential(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setenv("LIBTV_TOKEN", "env-token")
    monkeypatch.setenv("LIBTV_WEBID", "env-webid")
    monkeypatch.setenv("LIBTV_TOKEN_2", "account-2-token")
    monkeypatch.setenv("LIBTV_WEBID_2", "account-2-webid")

    pool = [
        {"id": "account-1", "api_key": "os.environ/LIBTV_TOKEN", "webid": "os.environ/LIBTV_WEBID"},
        {"id": "account-2", "api_key": "os.environ/LIBTV_TOKEN_2", "webid": "os.environ/LIBTV_WEBID_2"},
    ]
    monkeypatch.setattr(endpoints, "_image_upscale_deployment_pool", lambda *a, **k: pool)

    client = endpoints._client_for_receipt(_receipt(deployment_id="account-2"))
    assert client.token == "account-2-token"
