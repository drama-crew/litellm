"""A paid image upscale must record the deployment that actually ran it.

Two independent defects combined to make every production receipt record
``deployment_id="unknown"``:

1. ``optional_params["model_info"]`` never arrives on the image path.
   ``model_info`` is a member of ``all_litellm_params``, so
   ``litellm/images/main.py`` excludes it from ``non_default_params`` and it is
   dropped before the provider handler runs. The video path already worked
   around this by falling back to ``libtv_status_model`` (handler.py, video
   create); the image path did not.

2. The proxy's deployment pool required ``custom_llm_provider == "libtv"``,
   which libtv deployments never set -- see
   ``test_endpoints.test_image_upscale_pool_accepts_provider_declared_by_model_prefix``.

With no id and no pool the submitter fell back to the literal string
``"unknown"``, which is not a real deployment. ``_client_for_receipt`` then
refuses to build a client for such a receipt (503), so poll and resume are
permanently unavailable -- a task that was successfully submitted and billed
can never be collected.
"""

from types import SimpleNamespace

import pytest

from litellm.llms.libtv.handler import LibTVLLM


def _optional_params(**overrides):
    base = {
        "request_id": "req-1",
        "source_url": "https://source.example/input.png",
        "style": "Standard V2",
        "scale": 2,
        "receipt_team_id": "team-1",
        "receipt_api_key": "sk-1",
        "receipt_user_id": "user-1",
        "libtv_status_model": "libtv-topaz-image-upscaler-account-1",
    }
    base.update(overrides)
    return base


class _Recorder:
    def __init__(self):
        self.deployment_id = "__unset__"

    async def aresolve_model_spec(self, model):
        return {"vendor": "topazlabs", "task_type": "image", "model_key": "topaz-image-upscaler"}

    async def asubmit_image_upscale(self, model, vendor, source, style, scale, project, request_id, deployment_id, **kwargs):
        self.deployment_id = deployment_id
        return SimpleNamespace(request_id=request_id, submission_state="submitted", deployment_id=deployment_id)


async def _run(monkeypatch, optional_params):
    recorder = _Recorder()
    handler = LibTVLLM()
    monkeypatch.setattr(handler, "_make_client", lambda *a, **k: recorder)
    await handler.asubmit_image_upscale(
        model="topaz-image-upscaler",
        api_key="token",
        api_base=None,
        optional_params=optional_params,
        logging_obj=SimpleNamespace(),
        client=SimpleNamespace(),
    )
    return recorder.deployment_id


@pytest.mark.asyncio
async def test_deployment_id_falls_back_to_status_model_when_model_info_is_stripped(monkeypatch):
    """Production shape: model_info is gone, libtv_status_model survives."""
    assert await _run(monkeypatch, _optional_params()) == "libtv-topaz-image-upscaler-account-1"


@pytest.mark.asyncio
async def test_trusted_config_wins_over_the_caller_supplied_model_info(monkeypatch):
    """``model_info`` is an accepted request-body field on the submit endpoint,
    so a caller can name its own deployment id. Trust the deployment's own
    config first: a caller-chosen id would be written into the receipt and
    signed into the resume token, making the task unrecoverable and the token
    unverifiable against the real deployment's secret."""
    params = _optional_params(model_info={"id": "caller-supplied"})
    assert await _run(monkeypatch, params) == "libtv-topaz-image-upscaler-account-1"


@pytest.mark.asyncio
async def test_model_info_id_is_used_when_config_carries_no_status_model(monkeypatch):
    params = _optional_params(model_info={"id": "explicit-deployment"})
    params.pop("libtv_status_model")
    assert await _run(monkeypatch, params) == "explicit-deployment"


@pytest.mark.asyncio
async def test_deployment_id_is_none_when_neither_source_is_available(monkeypatch):
    """Fail honestly rather than inventing an id: with a pool present the
    submitter derives real ids from it, and without one the receipt is
    already unrecoverable."""
    params = _optional_params()
    params.pop("libtv_status_model")
    assert await _run(monkeypatch, params) is None


def test_model_info_is_stripped_by_litellm_before_handlers_run():
    """Guards the premise: if litellm ever stops treating model_info as a
    litellm param, this fallback becomes dead code and should be revisited."""
    from litellm.types.utils import all_litellm_params

    assert "model_info" in all_litellm_params
    assert "libtv_status_model" not in all_litellm_params


def test_provider_selection_prefers_the_pool():
    from litellm.llms.libtv.client import select_image_upscale_providers

    pool = [("account-1", object(), "tok-1"), ("account-2", object(), "tok-2")]
    selected = select_image_upscale_providers(pool, "account-1", lambda: (object(), "fallback"))
    assert [entry[0] for entry in selected] == ["account-1", "account-2"]


def test_provider_selection_falls_back_to_the_routed_deployment():
    from litellm.llms.libtv.client import select_image_upscale_providers

    provider = object()
    selected = select_image_upscale_providers([], "account-2", lambda: (provider, "tok"))
    assert selected == (("account-2", provider, "tok"),)


@pytest.mark.parametrize("deployment_id", [None, "", "unknown-but-empty-pool"])
def test_provider_selection_never_invents_an_unresolvable_deployment(deployment_id):
    """No pool and no id must yield no candidates at all.

    The previous code substituted the literal string "unknown", which names no
    deployment: ``_client_for_receipt`` then answers 503 for every poll and
    resume, so a task that was really submitted -- and really billed -- could
    never be collected. Refusing before the provider call keeps the failure on
    the safe side of the money.
    """
    from litellm.llms.libtv.client import select_image_upscale_providers

    called = False

    def _fallback():
        nonlocal called
        called = True
        return (object(), "tok")

    selected = select_image_upscale_providers([], deployment_id, _fallback)
    if deployment_id:
        assert selected and selected[0][0] == deployment_id
        assert called
    else:
        assert selected == ()
        assert not called


@pytest.mark.asyncio
async def test_client_refuses_the_provider_call_when_no_deployment_is_identifiable(monkeypatch):
    """Wiring test for the fail-closed path, not just the selection helper.

    Deleting the refusal in LibTVClient.asubmit_image_upscale leaves every
    helper test green: ImageUpscaleSubmitter would instead raise
    ValueError("at least one image upscale deployment is required"), which
    surfaces as submission_state "unknown" -- i.e. "we cannot tell whether you
    were billed", the one state that forbids failover and locks the pool. That
    is strictly worse than refusing, so it needs its own guard.
    """
    from litellm.llms.libtv import client as libtv_client

    calls = []

    class _Store:
        async def readiness(self):
            return True

    async def _forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("no provider work may happen without a deployment identity")

    monkeypatch.setattr(libtv_client, "get_receipt_store", lambda: _Store())
    monkeypatch.setattr(libtv_client.LibTVClient, "aensure_libtv_url", _forbidden)
    monkeypatch.setattr(libtv_client.LibTVClient, "acreate", _forbidden)

    client = libtv_client.LibTVClient(token="t", webid="w", async_client=SimpleNamespace())
    receipt = await client.asubmit_image_upscale(
        "topaz-image-upscaler",
        "topazlabs",
        "https://source.example/input.png",
        "Standard V2",
        2,
        "project",
        "req-1",
        None,
        team_id="team-1",
        durable_receipts=True,
        response_cost=0.46,
        receipt_api_key="sk-1",
        receipt_user_id="user-1",
        deployment_pool=[],
    )

    assert receipt.submission_state == "not_submitted"
    assert receipt.deployment_id != "unknown"
    assert calls == []


def _pool_entry(entry_id, key_var, webid_var):
    return {"id": entry_id, "api_key": f"os.environ/{key_var}", "webid": f"os.environ/{webid_var}"}


def test_each_pool_entry_is_paired_with_its_own_credential(monkeypatch):
    """A deployment's provider must always hold that deployment's credential.

    The routed client used to be reused for whichever entry matched the
    *deployment id*, so if that id came from a key that named a different
    account (``libtv_status_model`` and ``model_info.id`` are separate
    namespaces with nothing enforcing equality), account A's token would be
    recorded and resume-signed under account B's id: recovery then resolves B's
    token, fails signature verification, and polls the wrong account. Matching
    on the credential instead needs no naming convention to hold.
    """
    from litellm.llms.libtv.client import build_image_upscale_providers

    monkeypatch.setenv("LIBTV_TOKEN", "token-1")
    monkeypatch.setenv("LIBTV_WEBID", "webid-1")
    monkeypatch.setenv("LIBTV_TOKEN_2", "token-2")
    monkeypatch.setenv("LIBTV_WEBID_2", "webid-2")

    pool = [_pool_entry("account-1", "LIBTV_TOKEN", "LIBTV_WEBID"), _pool_entry("account-2", "LIBTV_TOKEN_2", "LIBTV_WEBID_2")]
    own = object()
    built = build_image_upscale_providers(
        pool, "token-2", lambda: own, lambda token, webid: (token, webid)
    )

    assert [(entry_id, token) for entry_id, _, token in built] == [("account-1", "token-1"), ("account-2", "token-2")]
    # The routed client is reused for the entry whose credential it actually
    # holds -- account-2 here -- regardless of any deployment id.
    assert built[1][1] is own
    assert built[0][1] == ("token-1", "webid-1")


def test_pool_entries_without_resolvable_credentials_are_skipped(monkeypatch):
    from litellm.llms.libtv.client import build_image_upscale_providers

    monkeypatch.delenv("LIBTV_TOKEN_MISSING", raising=False)
    monkeypatch.setenv("LIBTV_TOKEN", "token-1")
    monkeypatch.setenv("LIBTV_WEBID", "webid-1")

    pool = [
        _pool_entry("account-1", "LIBTV_TOKEN", "LIBTV_WEBID"),
        _pool_entry("account-missing", "LIBTV_TOKEN_MISSING", "LIBTV_WEBID"),
        {"id": "", "api_key": "os.environ/LIBTV_TOKEN", "webid": "os.environ/LIBTV_WEBID"},
    ]
    built = build_image_upscale_providers(pool, "other", lambda: object(), lambda token, webid: (token, webid))
    assert [entry_id for entry_id, _, _ in built] == ["account-1"]


def test_sync_submit_also_refuses_without_a_deployment_identity(monkeypatch):
    """The sync twin must not keep the defect the async path just lost.

    It is unreachable from the proxy today (the route type is
    aimage_generation), but it performs a real provider create and signs a
    resume token, so leaving it substituting "unknown" keeps a live path that
    can bill for an uncollectable task.
    """
    from litellm.llms.libtv import client as libtv_client

    client = libtv_client.LibTVClient(token="t", webid="w", sync_client=SimpleNamespace())

    def _forbidden(*args, **kwargs):
        raise AssertionError("no provider work may happen without a deployment identity")

    monkeypatch.setattr(libtv_client.LibTVClient, "create", _forbidden)

    receipt = client.submit_image_upscale(
        "topaz-image-upscaler", "topazlabs", "https://source.example/input.png",
        "Standard V2", 2, "project", "req-1", None,
    )
    assert receipt.submission_state == "not_submitted"
    assert receipt.deployment_id != "unknown"
    assert receipt.resume_token is None
