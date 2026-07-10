import httpx
import pytest
from unittest.mock import patch

import litellm
from litellm import Router


def _deployment(dep_id: str, token: str = "token") -> dict:
    return {
        "model_name": "seedance-2.0",
        "litellm_params": {
            "model": "openai/gpt-4o",
            "api_key": token,
            "webid": f"webid-{dep_id}",
            "num_retries": 0,
            "weight": 1,
        },
        "model_info": {"id": dep_id, "mode": "video_generation"},
    }


@pytest.mark.asyncio
async def test_generic_async_failure_carries_deployment_metadata(monkeypatch):
    deployment = _deployment("account-1")
    router = Router(model_list=[deployment], num_retries=2)

    async def select(*args, **kwargs):
        return deployment

    async def precheck(*args, **kwargs):
        return None

    async def failing(**kwargs):
        raise litellm.RateLimitError(
            message="capacity",
            model="star-video2",
            llm_provider="libtv",
            response=httpx.Response(
                429,
                request=httpx.Request("POST", "https://api.liblib.tv"),
            ),
        )

    monkeypatch.setattr(router, "async_get_available_deployment", select)
    monkeypatch.setattr(router, "async_routing_strategy_pre_call_checks", precheck)

    with pytest.raises(litellm.RateLimitError) as exc:
        await router._ageneric_api_call_with_fallbacks_helper(model="seedance-2.0", original_generic_function=failing)
    assert exc.value.failed_deployment_id == "account-1"
    assert exc.value.num_retries == 0


def test_failover_eligibility_blocks_deterministic_400():
    bad_request = litellm.BadRequestError(
        message="invalid material",
        model="star-video2",
        llm_provider="libtv",
    )
    assert Router._is_failover_eligible_exception(bad_request) is False
    assert (
        Router._is_failover_eligible_exception(
            litellm.ContentPolicyViolationError(message="blocked", model="star-video2", llm_provider="libtv")
        )
        is False
    )
    assert (
        Router._is_failover_eligible_exception(
            litellm.RateLimitError(message="capacity", model="star-video2", llm_provider="libtv")
        )
        is True
    )


def test_video_aliases_resolve_only_via_explicit_video_method():
    account_1 = _deployment("libtv-seedance-2-standard-account-1", "one")
    account_2 = _deployment("libtv-seedance-2-standard-account-2", "two")
    router = Router(
        model_list=[account_1, account_2],
        video_id_model_aliases={
            "libtv": {
                "seedance-2.0": "libtv-seedance-2-standard-account-1",
                "_fallback2/seedance-2.0": "libtv-seedance-2-standard-account-2",
            }
        },
    )
    assert router.resolve_video_model_id_alias("libtv", "seedance-2.0") == "libtv-seedance-2-standard-account-1"
    assert (
        router.resolve_video_model_id_alias("libtv", "_fallback2/seedance-2.0") == "libtv-seedance-2-standard-account-2"
    )
    assert router.resolve_video_model_id_alias("wavespeed", "seedance-2.0") == "seedance-2.0"
    # Normal create resolution remains the public model group.
    assert router.resolve_model_name_from_model_id("seedance-2.0") == "seedance-2.0"


def test_retry_after_header_controls_deployment_cooldown():
    deployment = _deployment("account-1")
    router = Router(model_list=[deployment], cooldown_time=300)
    error = litellm.RateLimitError(
        message="capacity",
        model="star-video2",
        llm_provider="libtv",
        response=httpx.Response(
            429,
            headers={"Retry-After": "17"},
            request=httpx.Request("POST", "https://api.liblib.tv"),
        ),
    )
    kwargs = {
        "exception": error,
        "litellm_params": {
            "metadata": {"model_group": "seedance-2.0"},
            "model_info": {"id": "account-1"},
        },
    }
    with patch("litellm.router._set_cooldown_deployments") as set_cooldown:
        router.deployment_callback_on_failure(
            kwargs=kwargs,
            completion_response=None,
            start_time=None,
            end_time=None,
        )
    assert set_cooldown.call_args.kwargs["time_to_cooldown"] == 17


def _pool_router(*, fallbacks=None) -> Router:
    def deployment(group: str, dep_id: str, key: str, weight: int) -> dict:
        return {
            "model_name": group,
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": key,
                "weight": weight,
                "num_retries": 0,
            },
            "model_info": {"id": dep_id},
        }

    return Router(
        model_list=[
            deployment("pool", "account-1", "A", 1),
            deployment("pool", "account-2", "B", 0),
            deployment("external", "wavespeed", "W", 1),
        ],
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
        num_retries=0,
        fallbacks=fallbacks or [],
    )


@pytest.mark.asyncio
async def test_generic_async_tries_sibling_before_external_fallback():
    router = _pool_router(fallbacks=[{"pool": ["external"]}])
    calls = []

    async def fake_video(**kwargs):
        key = kwargs["api_key"]
        calls.append(key)
        if key in {"A", "B"}:
            raise litellm.RateLimitError(message="capacity", model="video", llm_provider="fake")
        return "wavespeed-ok"

    result = await router._ageneric_api_call_with_fallbacks(model="pool", original_function=fake_video)
    assert result == "wavespeed-ok"
    assert calls == ["A", "B", "W"]


@pytest.mark.asyncio
async def test_generic_async_bad_request_is_not_replayed():
    router = _pool_router(fallbacks=[{"pool": ["external"]}])
    calls = []

    async def fake_video(**kwargs):
        calls.append(kwargs["api_key"])
        raise litellm.BadRequestError(message="invalid material", model="video", llm_provider="fake")

    with pytest.raises(litellm.BadRequestError):
        await router._ageneric_api_call_with_fallbacks(model="pool", original_function=fake_video)
    assert calls == ["A"]
