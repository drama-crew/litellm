"""Real FastAPI Proxy config smoke test for the libtv multi-account pool.

Boots the real ``litellm.proxy.proxy_server`` app from a YAML config file via
``initialize(config=...)`` (the same ``ProxyConfig.load_config`` path a
deployed proxy uses), then drives the real ``/v1/videos*`` endpoints through
``TestClient``. Unlike ``test_libtv_pool_http_e2e.py`` (Router only, no
FastAPI/auth/YAML), this proves:

1. ``router_settings.video_id_model_aliases`` from YAML actually lands on the
   live ``Router`` instance.
2. The custom ``libtv_params`` field ``libtv_require_explicit_credentials``
   survives YAML -> ``LiteLLM_Params`` -> ``optional_params`` without being
   filtered.
3. ``/v1/videos/{id}`` and ``/v1/videos/{id}/content`` resolve a legacy,
   public-model-id-encoded video id to the correct pool deployment (and thus
   the correct libtv account credentials) via the real endpoint code path.
4. ``/v1/models`` (the public listing ``/v1/model/info`` itself does not
   filter for master-key/admin callers) hides both the per-account pool
   deployments and the hidden external fallback deployments behind the
   public model names.

The libtv HTTP boundary is faked by swapping the ``AsyncHTTPHandler``/
``HTTPHandler`` names imported into ``litellm.llms.libtv.handler`` for a
fake transport keyed on request path and the ``token`` header, so the
account-1 vs account-2 credentials actually used are observable without any
real network call.
"""

import asyncio
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.types.videos.utils import encode_video_id_with_provider

PUBLIC_MODELS = (
    "nano-banana-pro",
    "seedance-2.0",
    "seedance-2.0-fast",
    "kling-v3-omni",
    "happy-horse-1.1",
)
MASTER_KEY = "sk-libtv-pool-smoke-test"
BLOCKED_PUBLIC_MODEL = "seedance-2.0-blocked-test"

_MODEL_KEY_BY_PUBLIC_NAME = {
    "nano-banana-pro": "nebula-ultra",
    "seedance-2.0": "star-video2",
    "seedance-2.0-fast": "star-video2-fast",
    "kling-v3-omni": "kling-v3-omni",
    "happy-horse-1.1": "happy-horse-1.1",
}

_CUSTOM_HANDLERS_MODULE = textwrap.dedent(
    """
    from litellm.llms.libtv import libtv_proxy_handler
    """
)


def _pool_config_yaml() -> str:
    model_entries = []
    for public_name in PUBLIC_MODELS:
        model_key = _MODEL_KEY_BY_PUBLIC_NAME[public_name]
        for slot, weight in (("account-1", 1), ("account-2", 0)):
            model_entries.append(
                f"""
  - model_name: {public_name}
    litellm_params:
      model: libtv/{model_key}
      libtv_status_model: {public_name}-{slot}
      libtv_require_explicit_credentials: true
      api_key: os.environ/LIBTV_TOKEN_{"1" if slot == "account-1" else "2"}
      webid: os.environ/LIBTV_WEBID_{"1" if slot == "account-1" else "2"}
      num_retries: 0
      weight: {weight}
    model_info:
      id: {public_name}-{slot}
"""
            )
    for public_name in ("seedance-2.0", "seedance-2.0-fast"):
        model_entries.append(
            f"""
  - model_name: _fallback/{public_name}
    litellm_params:
      model: libtv/{_MODEL_KEY_BY_PUBLIC_NAME[public_name]}
      libtv_require_explicit_credentials: true
      api_key: os.environ/LIBTV_TOKEN_1
      webid: os.environ/LIBTV_WEBID_1
      num_retries: 0
      weight: 1
    model_info:
      id: wavespeed-{public_name}
      hidden: true
"""
        )

    model_entries.append(
        f"""
  - model_name: {BLOCKED_PUBLIC_MODEL}
    litellm_params:
      model: libtv/{_MODEL_KEY_BY_PUBLIC_NAME["seedance-2.0"]}
      libtv_require_explicit_credentials: true
      api_key: os.environ/LIBTV_TOKEN_1
      webid: os.environ/LIBTV_WEBID_1
      num_retries: 0
    model_info:
      id: {BLOCKED_PUBLIC_MODEL}-account-1
      blocked: true
"""
    )

    aliases = "\n".join(
        f'      {public_name}: "{public_name}-account-1"' for public_name in PUBLIC_MODELS
    )

    return f"""
model_list:{"".join(model_entries)}

router_settings:
  routing_strategy: simple-shuffle
  enable_weighted_failover: true
  video_id_model_aliases:
    libtv:
{aliases}

general_settings:
  master_key: {MASTER_KEY}

litellm_settings:
  custom_provider_map:
    - provider: libtv
      custom_handler: libtv_pool_smoke_custom_handlers.libtv_proxy_handler
"""


class _FakeLibTVTransport:
    """Fakes the libtv REST surface the handler talks to, keyed on path + token."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Optional[str]]] = []

    def _token(self, headers: Optional[Dict[str, str]]) -> Optional[str]:
        return (headers or {}).get("token")

    async def post(
        self,
        *,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
    ) -> httpx.Response:
        token = self._token(headers)
        self.calls.append((url, token))
        if url.endswith("/api/canvas/project/create"):
            body = {"data": {"projectMeta": {"uuid": f"proj-{token}", "teamId": 1}}}
        elif url.endswith("/api/canvas/nodes/batch"):
            body = {"data": {}}
        elif url.endswith("/api/task/generation/create"):
            body = {"data": {"taskId": f"task-{token}"}}
        elif url.endswith("/api/task/generation/progress"):
            task_id = (json or {}).get("taskIds", [None])[0]
            body = {
                "data": {
                    "progresses": [
                        {
                            "taskId": task_id,
                            "status": 2,
                            "taskResult": {"videos": [{"url": f"https://cdn.example.com/{token}.mp4"}]},
                        }
                    ]
                }
            }
        else:
            raise AssertionError(f"unexpected POST {url}")
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    async def get(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
    ) -> httpx.Response:
        token = self._token(headers)
        self.calls.append((url, token))
        if url.endswith("/api/tool_spec/list"):
            tools = [
                {
                    "type": "video_generation",
                    "metadata": {
                        "modelKey": model_key,
                        "modelVendor": "seedance2.0",
                        "properties": {},
                        "config": {"settings": []},
                    },
                }
                for model_key in _MODEL_KEY_BY_PUBLIC_NAME.values()
            ]
            body = {"data": {"tools": tools}}
            return httpx.Response(200, json=body, request=httpx.Request("GET", url))
        # Video content CDN download.
        return httpx.Response(200, content=b"FAKEVIDEOBYTES", request=httpx.Request("GET", url))


@pytest.fixture()
def pool_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBTV_TOKEN_1", "token-account-1")
    monkeypatch.setenv("LIBTV_WEBID_1", "webid-account-1")
    monkeypatch.setenv("LIBTV_TOKEN_2", "token-account-2")
    monkeypatch.setenv("LIBTV_WEBID_2", "webid-account-2")

    config_path = tmp_path / "libtv_pool_smoke_config.yaml"
    config_path.write_text(_pool_config_yaml())
    (tmp_path / "libtv_pool_smoke_custom_handlers.py").write_text(_CUSTOM_HANDLERS_MODULE)

    transport = _FakeLibTVTransport()

    import litellm.llms.libtv.handler as libtv_handler_module

    monkeypatch.setattr(libtv_handler_module, "AsyncHTTPHandler", lambda **_: transport)
    monkeypatch.setattr(libtv_handler_module, "HTTPHandler", lambda **_: transport)

    with (
        patch("litellm.proxy.common_utils.banner.show_banner"),
        patch("litellm.proxy.proxy_server.generate_feedback_box"),
    ):
        from litellm.proxy.proxy_server import app, cleanup_router_config_variables, initialize

        try:
            asyncio.run(initialize(config=str(config_path)))
            import litellm.proxy.proxy_server as proxy_server_module

            yield proxy_server_module, TestClient(app), transport
        finally:
            cleanup_router_config_variables()
            litellm.custom_provider_map = []


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {MASTER_KEY}"}


class TestVideoIdModelAliasesLoadFromYaml:
    def test_router_settings_video_id_model_aliases_reach_the_router(self, pool_proxy):
        proxy_server_module, _client, _transport = pool_proxy
        router = proxy_server_module.llm_router
        assert router is not None
        assert router.video_id_model_aliases == {
            "libtv": {public_name: f"{public_name}-account-1" for public_name in PUBLIC_MODELS}
        }


class TestCustomLitellmParamsFieldSurvivesYamlLoad:
    def test_libtv_require_explicit_credentials_present_after_yaml_load(self, pool_proxy):
        proxy_server_module, _client, _transport = pool_proxy
        router = proxy_server_module.llm_router
        deployment = router.get_deployment(model_id="seedance-2.0-account-1")
        assert deployment is not None
        litellm_params = deployment.litellm_params.model_dump()
        assert litellm_params["libtv_require_explicit_credentials"] is True

    def test_field_reaches_handler_optional_params_on_a_real_call(self, pool_proxy):
        _proxy_server_module, client, transport = pool_proxy
        response = client.post(
            "/v1/videos",
            json={"model": "seedance-2.0", "prompt": "a cat playing piano"},
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        # The fake transport only serves canned bodies for the exact libtv REST
        # surface the handler calls; reaching generation/create at all proves
        # ``_make_client`` executed its explicit-credentials branch (the custom
        # field survived filtering) instead of raising/using ambient env creds.
        assert any(url.endswith("/api/task/generation/create") for url, _token in transport.calls)


class TestVideoEndpointsRouteThroughAliasToCorrectPoolDeployment:
    def test_generate_uses_the_weighted_primary_account_1_deployment(self, pool_proxy):
        _proxy_server_module, client, transport = pool_proxy
        response = client.post(
            "/v1/videos",
            json={"model": "seedance-2.0", "prompt": "a cat playing piano"},
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        create_calls = [token for url, token in transport.calls if url.endswith("/api/task/generation/create")]
        assert create_calls == ["token-account-1"]

    def test_legacy_public_model_id_encoded_video_id_resolves_via_alias_to_account_1(self, pool_proxy):
        _proxy_server_module, client, transport = pool_proxy
        # Legacy encoding: model_id is the *public* model name, not a deployment
        # id. Only ``router.video_id_model_aliases`` can resolve this to a real
        # deployment; without it the proxy would 4xx or silently use ambient env
        # credentials instead of the pool's account-1 slot.
        legacy_video_id = encode_video_id_with_provider("legacy-task-1", "libtv", "seedance-2.0")

        status_response = client.get(f"/v1/videos/{legacy_video_id}", headers=_auth_headers())
        assert status_response.status_code == 200, status_response.text
        assert status_response.json()["status"] == "completed"

        content_response = client.get(f"/v1/videos/{legacy_video_id}/content", headers=_auth_headers())
        assert content_response.status_code == 200, content_response.text
        assert content_response.content == b"FAKEVIDEOBYTES"

        progress_calls = [token for url, token in transport.calls if url.endswith("/api/task/generation/progress")]
        assert progress_calls
        assert set(progress_calls) == {"token-account-1"}

    def test_legacy_video_id_never_reaches_account_2_credentials(self, pool_proxy):
        _proxy_server_module, client, transport = pool_proxy
        legacy_video_id = encode_video_id_with_provider("legacy-task-2", "libtv", "seedance-2.0-fast")

        response = client.get(f"/v1/videos/{legacy_video_id}", headers=_auth_headers())
        assert response.status_code == 200, response.text

        progress_calls = [token for url, token in transport.calls if url.endswith("/api/task/generation/progress")]
        assert "token-account-2" not in progress_calls


class TestModelListingHidesAccountSlotsAndHiddenFallbacks:
    def test_v1_models_exposes_only_the_five_public_model_names(self, pool_proxy):
        _proxy_server_module, client, _transport = pool_proxy
        response = client.get("/v1/models", headers=_auth_headers())
        assert response.status_code == 200, response.text
        model_ids = {entry["id"] for entry in response.json()["data"]}
        assert model_ids == set(PUBLIC_MODELS)
        for model_id in model_ids:
            assert "-account-" not in model_id
            assert not model_id.startswith("_fallback")

    def test_v1_models_never_leaks_hidden_wavespeed_fallback_group_names(self, pool_proxy):
        _proxy_server_module, client, _transport = pool_proxy
        response = client.get("/v1/models", headers=_auth_headers())
        assert response.status_code == 200, response.text
        model_ids = {entry["id"] for entry in response.json()["data"]}
        assert "_fallback/seedance-2.0" not in model_ids
        assert "_fallback/seedance-2.0-fast" not in model_ids

    def test_v1_models_never_leaks_a_fully_blocked_admin_paused_model(self, pool_proxy):
        # Regression test: a merge conflict resolution once discarded
        # `blocked_names` (paused-by-admin models) from the hidden-names set
        # used to filter `/v1/models`, so an admin-paused model would
        # reappear in the public listing even though every deployment behind
        # it has `model_info.blocked=True`.
        _proxy_server_module, client, _transport = pool_proxy
        response = client.get("/v1/models", headers=_auth_headers())
        assert response.status_code == 200, response.text
        model_ids = {entry["id"] for entry in response.json()["data"]}
        assert BLOCKED_PUBLIC_MODEL not in model_ids
