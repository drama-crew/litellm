"""Contract §9 end to end: real Router + real LibTV handler, faked only at the network boundary.

Two LibTV accounts serve model group ``seedance-2.5``. Account behaviour is
scripted per token; everything above the HTTP layer (client, handler,
litellm exception wrapping, router retry/fallback loop, metered entry) is real.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request

import litellm
import litellm.llms.causyn  # (pre-existing import cycle)
from litellm import Router
from litellm.llms.libtv import handler as libtv_handler
from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.handler import LibTVLLM
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import moderation_execution as execution
from litellm.proxy.video_endpoints import moderation_metering_entry as entry

MODEL_KEY = "seedance-2.5-libtv"
IMAGE_URL = "https://oss.example.com/refs/a.png"
NO_LOGIN = {"code": 401, "msg": "No login"}
NO_CAPACITY = {"code": 1200000136, "msg": "算力不足"}
TIMEOUT = "timeout"


class _Resp:
    def __init__(self, body):
        self.status_code = 200
        self._body = body
        self.text = json.dumps(body)
        self.headers = {}

    def json(self):
        return self._body


def _tool_spec():
    meta = {
        "modelKey": MODEL_KEY,
        "modelVendor": "vendor",
        "properties": {"autoCompliance": {"enable": True}},
        "config": {},
    }
    return {"code": 0, "data": {"tools": [{"type": "video", "metadata": json.dumps(meta)}]}}


class _Script:
    """Per-account scripted behaviour. ``getUserInfo`` / ``create`` are body dicts or TIMEOUT."""

    def __init__(self, get_user_info=None, create=None):
        self.get_user_info = get_user_info
        self.create = create
        self.calls = []


SCRIPTS: dict[str, _Script] = {}


def _fake_http():
    class _Fake:
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, url, headers=None, **kwargs):
            return _Resp(_tool_spec())

        async def post(self, url, json=None, headers=None, **kwargs):
            script = SCRIPTS[headers["token"]]
            script.calls.append(url)
            if url.endswith("/getUserInfo"):
                if script.get_user_info is not None:
                    return _Resp(script.get_user_info)
                return _Resp({"code": 0, "data": {"uuid": "u"}})
            if url.endswith("/project/create"):
                return _Resp({"code": 0, "data": {"projectMeta": {"uuid": "p", "teamId": 1}}})
            if url.endswith("/nodes/batch"):
                return _Resp({"code": 0, "data": {}})
            if url.endswith("/image/verify"):
                return _Resp({"code": 0, "data": {"list": [{"url": IMAGE_URL, "result": False}]}})
            if url.endswith("/generation/create"):
                if script.create == TIMEOUT:
                    raise httpx.ReadTimeout("provider did not answer", request=httpx.Request("POST", url))
                return _Resp(script.create or {"code": 0, "data": {"taskId": "task-1"}})
            raise AssertionError("unexpected libtv call " + url)

        post_once = post

    return _Fake


class _Persistence:
    """Account tok-2 already has the reference image uploaded; tok-1 has not."""

    async def cached_upload(self, account_key, source_key):
        return IMAGE_URL if account_key == self.cached_account else None

    async def cached_asset(self, account_key, url, asset_type, ttl_seconds=None):
        return {"assetId": "asset-1"}

    async def store_asset(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        async def unused(*args, **kwargs):
            return None

        return unused


@pytest.fixture
def libtv(monkeypatch):
    from litellm.llms.libtv.persistence import account_key

    SCRIPTS.clear()
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "s" * 32)
    monkeypatch.setenv("MEDIA_TRANSFER_MODE", "delegated")
    monkeypatch.setenv("LIBTV_WEBID", "webid")
    monkeypatch.delenv("LIBTV_UPLOAD_CACHE_DISABLED", raising=False)
    monkeypatch.setattr(libtv_handler, "AsyncHTTPHandler", _fake_http())
    persistence = _Persistence()
    persistence.cached_account = account_key("tok-2")
    monkeypatch.setattr(LibTVClient, "_get_persistence", lambda self: persistence)
    monkeypatch.setattr(LibTVClient, "_probe_size", lambda self, url: 10)
    handler = LibTVLLM(poll_interval=0.0, fresh_asset_retry_attempts=0)
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "libtv", "custom_handler": handler}])
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    from litellm.utils import custom_llm_setup

    custom_llm_setup()  # what the proxy does when it loads custom_provider_map
    handler.test_persistence = persistence
    return handler


def _router(tokens, *, num_retries=1):
    deployments = [
        {
            "model_name": "seedance-2.5",
            "litellm_params": {"model": "libtv/" + MODEL_KEY, "api_key": token, "webid": "webid", "order": i + 1},
            "model_info": {"id": f"dep-{token}"},
        }
        for i, token in enumerate(tokens)
    ]
    return Router(
        model_list=deployments,
        num_retries=num_retries,
        retry_after=0,
        fallbacks=[{"seedance-3": ["seedance-2.5"]}],
        enable_pre_call_checks=True,
        disable_cooldowns=True,
    )


async def _submit(monkeypatch, router, *, scoped=True, route="avideo_generation"):
    scope = MagicMock()
    scope.store.void_unsent = AsyncMock()
    scope.binding.intent_id = "intent"
    scope.binding.request_digest = "d" * 64
    scope.phase = "submit"
    monkeypatch.setattr(entry, "scope_for", AsyncMock(return_value=scope if scoped else None))
    from litellm.proxy.spend_tracking import budget_reservation

    monkeypatch.setattr(budget_reservation, "release_budget_reservation", AsyncMock())
    request = Request({"type": "http", "method": "POST", "path": "/v1/videos", "headers": [], "app": FastAPI()})
    auth = UserAPIKeyAuth(api_key="k", user_id="u", team_id="t")
    call = router.avideo_generation(model="seedance-2.5", prompt="a cat", seconds="5", image_references=[IMAGE_URL])
    with pytest.raises(Exception) as caught:
        await entry.execute(request, auth, route, call)
    return caught.value, scope


def _assert_ambiguous(error, scope):
    assert not (isinstance(error, HTTPException) and error.status_code == 503)
    assert execution.failure_outcome(error) == "ambiguous"
    scope.store.void_unsent.assert_not_awaited()


@pytest.mark.asyncio
async def test_production_scenario_no_login_then_no_capacity_is_503_not_sent(monkeypatch, libtv):
    SCRIPTS["tok-1"] = _Script(get_user_info=NO_LOGIN)
    SCRIPTS["tok-2"] = _Script(create=NO_CAPACITY)
    error, scope = await _submit(monkeypatch, _router(["tok-1", "tok-2"]))
    assert SCRIPTS["tok-1"].calls and SCRIPTS["tok-2"].calls, "both accounts must have been attempted"
    assert isinstance(error, HTTPException) and error.status_code == 503
    assert error.headers == {"x-drama-submission": "not_sent"}
    assert execution.failure_outcome(error) == "not_sent"
    scope.store.void_unsent.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [["tok-1", "tok-2"], ["tok-2", "tok-1"]])
async def test_no_login_and_create_timeout_is_ambiguous_in_either_order(monkeypatch, libtv, tokens):
    SCRIPTS["tok-1"] = _Script(get_user_info=NO_LOGIN)
    SCRIPTS["tok-2"] = _Script(create=TIMEOUT)
    error, scope = await _submit(monkeypatch, _router(tokens))
    assert SCRIPTS["tok-1"].calls and SCRIPTS["tok-2"].calls, "both accounts must have been attempted"
    _assert_ambiguous(error, scope)


@pytest.mark.asyncio
async def test_same_deployment_retry_no_login_then_timeout_is_ambiguous(monkeypatch, libtv):
    """First try: no login (getUserInfo 401). Retry on the same deployment: create times out."""
    from litellm.llms.libtv.persistence import account_key

    script = _Script(get_user_info=NO_LOGIN)
    SCRIPTS["tok-1"] = script
    router = _router(["tok-1"], num_retries=1)
    original_post = libtv_handler.AsyncHTTPHandler.post

    async def flip_after_first_login_failure(self, url, json=None, headers=None, **kwargs):
        response = await original_post(self, url, json=json, headers=headers, **kwargs)
        if url.endswith("/getUserInfo") and script.get_user_info is NO_LOGIN:
            script.get_user_info = None
            script.create = TIMEOUT
            # the retry finds the reference image already uploaded, so it reaches create
            libtv.test_persistence.cached_account = account_key("tok-1")
        return response

    libtv_handler.AsyncHTTPHandler.post = flip_after_first_login_failure
    try:
        error, scope = await _submit(monkeypatch, router)
    finally:
        libtv_handler.AsyncHTTPHandler.post = original_post
    assert any(url.endswith("/generation/create") for url in script.calls), "retry must reach create"
    _assert_ambiguous(error, scope)
