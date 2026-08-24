import base64
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.libtv.video_id_codec import (
    OPAQUE_VIDEO_ID_PREFIX,
    VideoIdKeyUnavailable,
    decode_libtv_video_id,
    encode_libtv_video_id,
)
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider


@pytest.mark.parametrize(
    "route",
    [
        "/internal/v1/validated-media-transfer",
        "/internal/v1/validated-media-transfer/readiness",
        "/internal/v1/image-upscale/submit",
        "/internal/v1/image-upscale/receipt/receipt-123",
        "/internal/v1/image-upscale/poll",
        "/internal/v1/image-upscale/finalize",
        "/internal/v1/image-upscale/resolve",
        "/v1/libtv/validated-media-transfer",
        "/v1/libtv/validated-media-transfer/readiness",
        "/v1/libtv/image-upscale/submit",
        "/v1/libtv/image-upscale/receipt/legacy-receipt-123",
        "/v1/libtv/image-upscale/poll",
        "/v1/libtv/image-upscale/finalize",
        "/v1/libtv/image-upscale/resolve",
    ],
)
def test_image_upscale_and_transfer_routes_are_llm_data_plane(route):
    from litellm.proxy.auth.route_checks import RouteChecks

    assert RouteChecks.is_llm_api_route(route) is True


@pytest.mark.parametrize(
    "route",
    [
        "/internal/v1/image-upscale/receipt",
        "/internal/v1/image-upscale/receipt/receipt-123/extra",
        "/internal/v1/image-upscale/submit/extra",
        "/internal/v1/management/image-upscale/submit",
        "/internal/v1/validated-media-transfer/admin",
        "/v1/libtv/image-upscale/receipt",
        "/v1/libtv/image-upscale/receipt/legacy-receipt-123/extra",
        "/v1/libtv/image-upscale/submit/extra",
        "/v1/libtv/management/image-upscale/submit",
        "/v1/libtv/validated-media-transfer/admin",
    ],
)
def test_unregistered_image_upscale_and_transfer_routes_are_not_llm_data_plane(route):
    from litellm.proxy.auth.route_checks import RouteChecks

    assert RouteChecks.is_llm_api_route(route) is False


def test_libtv_video_id_is_authenticated_opaque_and_roundtrips(monkeypatch):
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "stable-test-secret")
    value = encode_libtv_video_id("task-123", "libtv-deployment-account-1")

    assert value.startswith(OPAQUE_VIDEO_ID_PREFIX)
    assert "libtv" not in value.lower()
    assert "deployment" not in value.lower()
    assert "liblib" not in value.lower()
    decoded_blob = base64.urlsafe_b64decode(value[len(OPAQUE_VIDEO_ID_PREFIX) :] + "===")
    assert b"libtv" not in decoded_blob.lower()
    assert b"libtv-deployment-account-1" not in decoded_blob

    decoded = decode_video_id_with_provider(value)
    assert decoded["custom_llm_provider"] == "libtv"
    assert decoded["model_id"] == "libtv-deployment-account-1"
    assert decoded["video_id"] == "task-123"


def test_libtv_video_id_tamper_and_wrong_key_fail_closed(monkeypatch):
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "stable-test-secret")
    value = encode_libtv_video_id("task-123", "deployment-1")
    tampered = value[:-1] + ("A" if value[-1] != "A" else "B")
    assert decode_libtv_video_id(tampered) is None
    monkeypatch.setenv("LITELLM_VIDEO_ID_SECRET", "different-secret")
    assert decode_libtv_video_id(value) is None
    assert decode_video_id_with_provider(value)["custom_llm_provider"] is None


@pytest.mark.parametrize(
    "secret_env",
    ["LITELLM_MASTER_KEY", "LITELLM_SALT_KEY", "LITELLM_VIDEO_ID_SECRET"],
)
def test_libtv_video_id_uses_stable_key_across_worker_cold_start(monkeypatch, secret_env):
    for key in ("LITELLM_VIDEO_ID_SECRET", "LITELLM_SALT_KEY", "LITELLM_MASTER_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(secret_env, "shared-stable-secret")
    value = encode_libtv_video_id("task-123", "deployment-1")

    import litellm.types.videos.utils as video_utils

    video_utils._VIDEO_ID_CODECS.clear()
    sys.modules.pop("litellm.llms.libtv.video_id_codec", None)
    decoded = decode_video_id_with_provider(value)
    assert decoded["custom_llm_provider"] == "libtv"
    assert decoded["video_id"] == "task-123"


def test_missing_video_id_key_prevents_any_provider_create(monkeypatch):
    from unittest.mock import MagicMock

    from litellm.llms.libtv.handler import LibTVLLM

    monkeypatch.delenv("LITELLM_VIDEO_ID_SECRET", raising=False)
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    client = MagicMock()
    with pytest.raises(VideoIdKeyUnavailable):
        LibTVLLM().video_generation("model", "prompt", None, None, {}, None, client=client)
    client.resolve_model_spec.assert_not_called()
    client.create.assert_not_called()


def test_libtv_video_id_requires_stable_key(monkeypatch):
    monkeypatch.delenv("LITELLM_VIDEO_ID_SECRET", raising=False)
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    with pytest.raises(VideoIdKeyUnavailable):
        encode_libtv_video_id("task-123", "deployment-1")


def test_legacy_libtv_video_id_still_decodes(monkeypatch):
    legacy = encode_video_id_with_provider("old-task", "libtv", "old-deployment")
    decoded = decode_video_id_with_provider(legacy)
    assert decoded["custom_llm_provider"] == "libtv"
    assert decoded["model_id"] == "old-deployment"
    assert decoded["video_id"] == "old-task"


def test_legacy_video_status_rewraps_id_before_sync_poll(monkeypatch):
    from litellm.llms.libtv.handler import LibTVLLM

    monkeypatch.setenv("LITELLM_MASTER_KEY", "shared-video-key")
    legacy = encode_video_id_with_provider("old-task", "libtv", "deployment-1")
    client = MagicMock()
    client.poll_once.return_value = {"status": 1}
    handler = LibTVLLM()
    monkeypatch.setattr(handler, "_make_client", lambda *args, **kwargs: client)

    response = handler.video_status(legacy, None, None, {}, None)

    assert response.id.startswith(OPAQUE_VIDEO_ID_PREFIX)
    assert "libtv" not in response.id.lower()
    assert "deployment-1" not in response.id
    client.poll_once.assert_called_once_with("old-task", "video")

    client.poll_once.reset_mock()
    continued = handler.video_status(response.id, None, None, {}, None)
    assert continued.id == response.id
    client.poll_once.assert_called_once_with("old-task", "video")


@pytest.mark.asyncio
async def test_legacy_video_status_rewraps_id_before_async_poll(monkeypatch):
    from litellm.llms.libtv.handler import LibTVLLM

    monkeypatch.setenv("LITELLM_MASTER_KEY", "shared-video-key")
    legacy = encode_video_id_with_provider("old-task", "libtv", "deployment-1")
    client = MagicMock()
    client.apoll_once = AsyncMock(return_value={"status": 1})
    handler = LibTVLLM()
    monkeypatch.setattr(handler, "_make_client", lambda *args, **kwargs: client)

    response = await handler.avideo_status(legacy, None, None, {}, None)

    assert response.id.startswith(OPAQUE_VIDEO_ID_PREFIX)
    assert "libtv" not in response.id.lower()
    assert "deployment-1" not in response.id
    client.apoll_once.assert_awaited_once_with("old-task", "video")


def test_legacy_video_status_missing_key_fails_before_poll(monkeypatch):
    from litellm.llms.libtv.handler import LibTVLLM

    for key in ("LITELLM_VIDEO_ID_SECRET", "LITELLM_SALT_KEY", "LITELLM_MASTER_KEY"):
        monkeypatch.delenv(key, raising=False)
    legacy = encode_video_id_with_provider("old-task", "libtv", "deployment-1")
    client = MagicMock()
    handler = LibTVLLM()
    monkeypatch.setattr(handler, "_make_client", lambda *args, **kwargs: client)

    with pytest.raises(VideoIdKeyUnavailable, match="video IDs"):
        handler.video_status(legacy, None, None, {}, None)
    client.poll_once.assert_not_called()


def test_public_provider_model_serializer_hides_private_fields():
    from litellm.proxy.common_utils.public_model_serialization import serialize_public_provider_models

    model = {
        "model_name": "seedance-2.0",
        "litellm_params": {"model": "libtv/seedance-2.0", "api_base": "https://internal"},
        "model_info": {
            "id": "libtv-account-1",
            "litellm_provider": "libtv",
            "deployment": "private-deployment",
            "mode": "video_generation",
        },
    }
    result = serialize_public_provider_models([model, model], is_full_proxy_admin=False)
    assert len(result) == 1
    assert result[0]["model_name"] == "seedance-2.0"
    assert "litellm_params" not in result[0]
    assert result[0]["model_info"] == {"mode": "video_generation"}

    # View-only admins are still on the user boundary; only full proxy admins
    # may inspect provider-private deployment metadata.
    assert serialize_public_provider_models([model], is_full_proxy_admin=False) == result
    assert serialize_public_provider_models([model], is_full_proxy_admin=True) == [model]

    ordinary = {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}}
    assert serialize_public_provider_models([ordinary], is_full_proxy_admin=False) == [ordinary]


def test_model_group_provider_sanitizer_preserves_other_providers():
    from litellm.proxy.common_utils.public_surface_sanitization import sanitize_provider_names

    assert sanitize_provider_names(["openai", "libtv", "anthropic", "liblib.internal"]) == [
        "openai",
        "provider",
        "anthropic",
        "provider",
    ]


@pytest.mark.parametrize(
    "value",
    ["libtv", "LIBTV", "lib-tv", "lib_tv", "lib.tv", "lib/lib", "LibLib", "lib-lib", "lib_lib", "liblib.tv"],
)
def test_shared_private_provider_marker_matches_case_and_separator_variants(value):
    from litellm.proxy.common_utils.public_surface_sanitization import contains_private_provider_marker

    assert contains_private_provider_marker(value) is True


@pytest.mark.parametrize("value", ["ordinary text", "library lookup failed", "public upstream failed", "liberty"])
def test_shared_private_provider_marker_does_not_match_unrelated_text(value):
    from litellm.proxy.common_utils.public_surface_sanitization import contains_private_provider_marker

    assert contains_private_provider_marker(value) is False


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("LibLib", "provider"),
        ("LIB-LIB upstream failure", "provider upstream failure"),
        ("lib_lib account failure", "provider account failure"),
        ("liblib.tv failure", "provider failure"),
    ],
)
def test_video_failed_reason_hides_all_private_marker_variants(reason, expected):
    from litellm.llms.libtv.handler import _public_provider_text

    assert _public_provider_text(reason, "video generation failed") == expected


@pytest.mark.parametrize("reason", ["ordinary text", "library unavailable", "public generation failed"])
def test_video_failed_reason_preserves_unrelated_text(reason):
    from litellm.llms.libtv.handler import _public_provider_text

    assert _public_provider_text(reason, "video generation failed") == reason


def test_public_response_headers_hide_private_identity_but_keep_billing_usage():
    from litellm.proxy.common_utils.public_surface_sanitization import sanitize_public_response_headers
    from litellm.proxy._types import LitellmUserRoles

    non_admin = MagicMock(
        user_role=LitellmUserRoles.CUSTOMER,
        spend=1.0,
        tpm_limit=100,
        rpm_limit=10,
        max_budget=20,
    )
    headers = sanitize_public_response_headers(
        {
            "x-litellm-model-id": "libtv-private-deployment",
            "x-litellm-model-api-base": "https://liblib.internal/v1",
            "X-LibTV-Deployment": "safe-looking-value",
            "provider_hint": "libtv",
            "x-litellm-response-cost": "0.25",
            "x-litellm-key-spend": "1.25",
            "x-ratelimit-libtv-debug": "10",
            "x-ratelimit-remaining": "libtv-private-account",
            "x-ratelimit-reset": "60",
        },
        non_admin,
    )
    assert "x-litellm-model-id" not in headers
    assert "x-litellm-model-api-base" not in headers
    assert "X-LibTV-Deployment" not in headers
    assert "provider_hint" not in headers
    assert headers["x-litellm-response-cost"] == "0.25"
    assert headers["x-litellm-key-spend"] == "1.25"
    assert "x-ratelimit-libtv-debug" not in headers
    assert "x-ratelimit-remaining" not in headers
    assert headers["x-ratelimit-reset"] == "60"

    admin = MagicMock(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        spend=1.0,
        tpm_limit=100,
        rpm_limit=10,
        max_budget=20,
    )
    admin_headers = sanitize_public_response_headers(
        {
            "x-litellm-model-id": "libtv-private-deployment",
            "x-litellm-model-api-base": "https://liblib.internal/v1",
        },
        admin,
    )
    assert admin_headers["x-litellm-model-id"] == "libtv-private-deployment"
    assert admin_headers["x-litellm-model-api-base"] == "https://liblib.internal/v1"


def test_final_response_header_sanitizer_removes_callback_injection_from_mutable_response():
    from litellm.proxy.common_utils.public_surface_sanitization import finalize_public_response_headers
    from litellm.proxy._types import LitellmUserRoles
    from starlette.responses import Response

    response = Response(headers={"X-LibTV-Callback": "private", "x-safe": "ok"})
    finalize_public_response_headers(
        response.headers,
        MagicMock(user_role=LitellmUserRoles.CUSTOMER),
    )

    assert "x-libtv-callback" not in response.headers
    assert response.headers["x-safe"] == "ok"


def test_view_only_admin_uses_public_header_boundary():
    from litellm.proxy.common_utils.public_surface_sanitization import sanitize_public_response_headers
    from litellm.proxy._types import LitellmUserRoles

    view_only = MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    result = sanitize_public_response_headers(
        {"X-LibTV-Deployment": "private", "x-ratelimit-remaining": "9"},
        view_only,
    )
    assert "X-LibTV-Deployment" not in result
    assert result["x-ratelimit-remaining"] == "9"


def test_health_public_boundary_treats_view_only_as_non_full_admin():
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.common_utils.public_surface_sanitization import (
        is_full_proxy_admin,
        sanitize_health_endpoint,
    )

    view_only = MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert is_full_proxy_admin(view_only) is False
    assert sanitize_health_endpoint(
        {
            "healthy_endpoints": [
                {"model": "libtv/seedance", "api_base": "https://private", "status": "healthy"}
            ]
        }
    ) == {"healthy_endpoints": [{"status": "healthy"}]}


def test_project_llm_api_routes_allow_only_registered_data_plane_paths(monkeypatch):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.route_checks import RouteChecks

    monkeypatch.setattr(RouteChecks, "is_auth_enforced_pass_through_route", lambda **_: False)
    key = UserAPIKeyAuth(allowed_routes=["llm_api_routes"])
    allowed = [
        "/internal/v1/validated-media-transfer",
        "/internal/v1/validated-media-transfer/readiness",
        "/internal/v1/image-upscale/submit",
        "/internal/v1/image-upscale/receipt/receipt-123",
        "/internal/v1/image-upscale/poll",
        "/internal/v1/image-upscale/finalize",
        "/internal/v1/image-upscale/resolve",
        "/v1/libtv/validated-media-transfer",
        "/v1/libtv/validated-media-transfer/readiness",
        "/v1/libtv/image-upscale/submit",
        "/v1/libtv/image-upscale/receipt/legacy-receipt-123",
        "/v1/libtv/image-upscale/poll",
        "/v1/libtv/image-upscale/finalize",
        "/v1/libtv/image-upscale/resolve",
    ]
    for route in allowed:
        assert RouteChecks.is_virtual_key_allowed_to_call_route(route, key) is True

    assert RouteChecks.is_llm_api_route("/internal/v1/management/routes") is False


def test_public_health_sanitizer_hides_private_identity_fields():
    from litellm.proxy.common_utils.public_surface_sanitization import sanitize_health_endpoint

    result = sanitize_health_endpoint(
        {
            "model": "libtv/seedance",
            "model_id": "liblib-deployment-1",
            "provider": "libtv",
            "api_base": "https://liblib.internal",
            "status": "unhealthy",
            "error": "libtv upstream failed",
            "details": {"model": "openai/gpt-4o", "api_base": "https://api.openai.com"},
        }
    )
    assert "model" not in result
    assert "model_id" not in result
    assert "provider" not in result
    assert "api_base" not in result
    assert result["status"] == "unhealthy"
    assert result["error"] == "provider health check failed"
    assert result["details"] == {"model": "openai/gpt-4o"}


def test_public_health_sanitizer_hides_marker_bearing_keys_at_all_nesting_levels():
    from litellm.proxy.common_utils.public_surface_sanitization import sanitize_health_endpoint

    result = sanitize_health_endpoint(
        {
            "libtv_debug": "private",
            "safe": {
                "liblib": "private",
                "nested": {
                    "LIB-LIB": "private",
                    "status": "healthy",
                    "items": [[{"liblib_debug": "private", "status": "healthy"}]],
                },
                "status": "healthy",
            },
        }
    )

    assert result == {
        "safe": {
            "nested": {
                "status": "healthy",
                "items": [[{"status": "healthy"}]],
            },
            "status": "healthy",
        }
    }


def test_health_latest_hides_private_outer_key_for_view_only_admin():
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.common_utils.public_surface_sanitization import sanitize_public_health_checks

    result = sanitize_public_health_checks(
        {
            "libtv-deployment-1": {"model": "libtv/seedance"},
            "seedance-2.0": {"provider": "liblib", "status": "healthy"},
        },
        MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
    )

    assert "libtv-deployment-1" not in result
    assert result["seedance-2.0"] == {"status": "healthy"}


def test_health_history_uses_shared_full_admin_boundary_for_view_only():
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.common_utils.public_surface_sanitization import (
        is_full_proxy_admin,
        sanitize_health_endpoint,
    )

    view_only = MagicMock(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert is_full_proxy_admin(view_only) is False
    assert sanitize_health_endpoint(
        {"model_name": "liblib-private-model", "error_message": "libtv upstream failed"}
    ) == {"error_message": "provider health check failed"}
