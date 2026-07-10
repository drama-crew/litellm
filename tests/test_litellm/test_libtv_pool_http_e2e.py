"""Real-HTTP E2E for the isolated LiteLLM Router + fake provider pool."""

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import litellm
from litellm import Router
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)


PUBLIC_MODELS = (
    "nano-banana-pro",
    "seedance-2.0",
    "seedance-2.0-fast",
    "kling-v3-omni",
    "happy-horse-1.1",
)


class PoolHarness:
    def __init__(self):
        self.failures = {}
        model_list = []
        for model in PUBLIC_MODELS:
            for slot, weight in (("account-1", 1), ("account-2", 0)):
                dep_id = f"{model}-{slot}"
                model_list.append(
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": "openai/gpt-4o",
                            "api_key": slot,
                            "weight": weight,
                            "num_retries": 0,
                        },
                        "model_info": {"id": dep_id},
                    }
                )
        for model in ("seedance-2.0", "seedance-2.0-fast"):
            model_list.append(
                {
                    "model_name": f"_fallback/{model}",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "wavespeed",
                        "weight": 1,
                    },
                    "model_info": {"id": f"wavespeed-{model}"},
                }
            )
        self.router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            enable_weighted_failover=True,
            num_retries=0,
            allowed_fails=0,
            cooldown_time=300,
            fallbacks=[
                {"seedance-2.0": ["_fallback/seedance-2.0"]},
                {"seedance-2.0-fast": ["_fallback/seedance-2.0-fast"]},
            ],
            video_id_model_aliases={
                "libtv": {
                    "seedance-2.0": "seedance-2.0-account-1",
                    "seedance-2.0-fast": "seedance-2.0-fast-account-1",
                    "_fallback2/seedance-2.0": "seedance-2.0-account-2",
                    "_fallback2/seedance-2.0-fast": "seedance-2.0-fast-account-2",
                    "kling-v3-omni": "kling-v3-omni-account-1",
                    "happy-horse-1.1": "happy-horse-1.1-account-1",
                }
            },
        )

    async def generate(self, model: str):
        async def fake_provider(**kwargs):
            slot = kwargs["api_key"]
            deployment_id = kwargs["model_info"]["id"]
            failure = self.failures.get((model, slot))
            if failure:
                if failure == 401:
                    error = litellm.AuthenticationError(message="expired", model=model, llm_provider="fake")
                else:
                    error = litellm.RateLimitError(message="capacity", model=model, llm_provider="fake")
                self.router.deployment_callback_on_failure(
                    kwargs={
                        "exception": error,
                        "litellm_params": {
                            "metadata": {"model_group": model},
                            "model_info": {"id": deployment_id},
                        },
                    },
                    completion_response=None,
                    start_time=None,
                    end_time=None,
                )
                raise error
            result = {"deployment_id": deployment_id, "slot": slot}
            if model != "nano-banana-pro" and slot != "wavespeed":
                result["video_id"] = encode_video_id_with_provider("task-1", "libtv", deployment_id)
            return result

        return await self.router._ageneric_api_call_with_fallbacks(model=model, original_function=fake_provider)

    def lookup_video(self, video_id: str) -> dict:
        decoded = decode_video_id_with_provider(video_id)
        model_id = self.router.resolve_video_model_id_alias(decoded.get("custom_llm_provider"), decoded.get("model_id"))
        deployment = self.router.get_model_info(model_id)
        if deployment is None:
            raise KeyError(model_id)
        return {
            "deployment_id": model_id,
            "slot": deployment["litellm_params"]["api_key"],
        }


@contextmanager
def http_proxy(harness: PoolHarness):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
            if self.path == "/control":
                harness.failures[(body["model"], body["slot"])] = body["status"]
                self._json(200, {"ok": True})
                return
            try:
                self._json(200, asyncio.run(harness.generate(body["model"])))
            except Exception as error:
                self._json(getattr(error, "status_code", 500), {"error": str(error)})

        def do_GET(self):
            marker = "/v1/videos/"
            video_id = self.path.split(marker, 1)[1].split("/content", 1)[0]
            try:
                self._json(200, harness.lookup_video(video_id))
            except KeyError as error:
                self._json(404, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def request_json(base: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_all_five_models_and_video_affinity():
    with http_proxy(PoolHarness()) as base:
        for model in PUBLIC_MODELS:
            status, result = request_json(base, "/v1/generate", {"model": model})
            assert status == 200
            assert result["deployment_id"] == f"{model}-account-1"
            if model != "nano-banana-pro":
                for suffix in ("", "/content"):
                    status, lookup = request_json(base, f"/v1/videos/{result['video_id']}{suffix}")
                    assert status == 200
                    assert lookup["slot"] == "account-1"


@pytest.mark.parametrize("failure", [401, 429])
def test_http_failure_cools_account_and_uses_sibling(failure):
    with http_proxy(PoolHarness()) as base:
        request_json(
            base,
            "/control",
            {"model": "seedance-2.0", "slot": "account-1", "status": failure},
        )
        status, first = request_json(base, "/v1/generate", {"model": "seedance-2.0"})
        assert status == 200
        assert first["slot"] == "account-2"
        status, second = request_json(base, "/v1/generate", {"model": "seedance-2.0"})
        assert status == 200
        assert second["slot"] == "account-2"


def test_http_seedance_reaches_wavespeed_only_after_both_accounts_fail():
    with http_proxy(PoolHarness()) as base:
        for slot in ("account-1", "account-2"):
            request_json(
                base,
                "/control",
                {"model": "seedance-2.0-fast", "slot": slot, "status": 429},
            )
        status, result = request_json(base, "/v1/generate", {"model": "seedance-2.0-fast"})
        assert status == 200
        assert result["slot"] == "wavespeed"


def test_http_non_seedance_has_no_external_fallback():
    with http_proxy(PoolHarness()) as base:
        for slot in ("account-1", "account-2"):
            request_json(
                base,
                "/control",
                {"model": "kling-v3-omni", "slot": slot, "status": 429},
            )
        status, result = request_json(base, "/v1/generate", {"model": "kling-v3-omni"})
        assert status == 429
        assert "capacity" in result["error"]


@pytest.mark.parametrize(
    ("legacy_model_id", "expected_slot"),
    [
        ("seedance-2.0", "account-1"),
        ("_fallback2/seedance-2.0", "account-2"),
        ("seedance-2.0-fast", "account-1"),
        ("_fallback2/seedance-2.0-fast", "account-2"),
        ("kling-v3-omni", "account-1"),
        ("happy-horse-1.1", "account-1"),
    ],
)
def test_http_legacy_video_ids_keep_original_account(legacy_model_id, expected_slot):
    legacy_id = encode_video_id_with_provider("old-task", "libtv", legacy_model_id)
    with http_proxy(PoolHarness()) as base:
        for suffix in ("", "/content"):
            status, result = request_json(base, f"/v1/videos/{legacy_id}{suffix}")
            assert status == 200
            assert result["slot"] == expected_slot
