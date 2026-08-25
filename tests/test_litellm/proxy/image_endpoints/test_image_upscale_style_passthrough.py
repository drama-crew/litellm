"""The user's chosen Topaz style must actually reach the provider.

``style`` is an OpenAI image parameter, so ``litellm/images/main.py`` hands it
to ``get_optional_params_image_gen`` as a named argument, which keeps it only
for providers whose image-generation config declares support. libtv has none,
so ``style`` was silently dropped and every paid upscale ran the default
"Standard V2" no matter what the user picked -- the selector was inert.

Observed in production: a 4x "CGI" request and a 4x "Standard V2" request on
the same source produced the *identical* request fingerprint
(``request_fingerprint`` hashes the style it was given), proving the handler
never saw the requested style.

The pool and the submit flag already cross this boundary as passthrough keys;
the style now travels the same way.
"""

from types import SimpleNamespace

import pytest

from litellm.llms.libtv.handler import LibTVLLM


def test_style_is_dropped_but_the_alias_survives_optional_param_assembly():
    """Guards the premise. If litellm ever starts preserving ``style`` for
    libtv, the alias becomes redundant rather than wrong -- but this test
    should be revisited."""
    from litellm.utils import get_optional_params_image_gen

    params = get_optional_params_image_gen(
        model="topaz-image-upscaler",
        custom_llm_provider="libtv",
        style="CGI",
        image_upscale_style="CGI",
        scale=4,
    )
    assert "style" not in params
    assert params["image_upscale_style"] == "CGI"


def test_submit_endpoint_stashes_the_style_under_the_passthrough_alias():
    from litellm.proxy.image_endpoints.endpoints import _image_upscale_passthrough_style

    assert _image_upscale_passthrough_style({"style": "CGI"}) == "CGI"
    assert _image_upscale_passthrough_style({}) == "Standard V2"
    assert _image_upscale_passthrough_style({"style": ""}) == "Standard V2"


class _Recorder:
    def __init__(self):
        self.style = "__unset__"

    async def aresolve_model_spec(self, model):
        return {"vendor": "topazlabs", "task_type": "image", "model_key": "topaz-image-upscaler"}

    async def asubmit_image_upscale(self, model, vendor, source, style, scale, project, request_id, deployment_id, **kwargs):
        self.style = style
        return SimpleNamespace(request_id=request_id, submission_state="submitted", deployment_id=deployment_id)


async def _submitted_style(monkeypatch, optional_params):
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
    return recorder.style


def _params(**overrides):
    base = {
        "request_id": "req-1",
        "source_url": "https://source.example/input.png",
        "scale": 4,
        "receipt_team_id": "team-1",
        "receipt_api_key": "sk-1",
        "receipt_user_id": "user-1",
        "libtv_status_model": "libtv-topaz-image-upscaler-account-1",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["CGI", "High Fidelity V2", "Text Refine", "Low Resolution V2"])
async def test_handler_forwards_the_requested_style(monkeypatch, style):
    assert await _submitted_style(monkeypatch, _params(image_upscale_style=style)) == style


@pytest.mark.asyncio
async def test_handler_still_honours_a_surviving_native_style(monkeypatch):
    assert await _submitted_style(monkeypatch, _params(style="CGI")) == "CGI"


@pytest.mark.asyncio
async def test_handler_defaults_when_no_style_reaches_it(monkeypatch):
    assert await _submitted_style(monkeypatch, _params()) == "Standard V2"


def test_distinct_styles_produce_distinct_fingerprints():
    """Regression for the production symptom that exposed the drop."""
    from litellm.llms.libtv.receipts import request_fingerprint

    base = {"source_sha256": "a" * 64, "scale": 4}
    assert request_fingerprint({**base, "style": "CGI"}, "topaz-image-upscaler") != request_fingerprint(
        {**base, "style": "Standard V2"}, "topaz-image-upscaler"
    )
