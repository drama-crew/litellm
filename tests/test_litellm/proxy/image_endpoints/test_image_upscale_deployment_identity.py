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
async def test_model_info_id_still_wins_when_present(monkeypatch):
    params = _optional_params(model_info={"id": "explicit-deployment"})
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
