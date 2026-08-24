"""Shared sanitizers for data that can cross the public proxy boundary."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

PRIVATE_PROVIDER_MARKERS = ("libtv", "liblib")


def is_full_proxy_admin(user_or_role: Any) -> bool:
    """Return whether a role is allowed to inspect provider-private details."""
    role = getattr(user_or_role, "user_role", user_or_role)
    role = getattr(role, "value", role)
    return str(role).lower() == "proxy_admin"


def contains_private_provider_marker(value: Any) -> bool:
    """Check nested values without stringifying arbitrary objects."""
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in PRIVATE_PROVIDER_MARKERS)
    if isinstance(value, Mapping):
        return any(contains_private_provider_marker(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_private_provider_marker(item) for item in value)
    return False


def neutralize_provider_name(value: Any) -> Any:
    """Replace only provider-private names; leave unrelated providers intact."""
    return "provider" if contains_private_provider_marker(value) else value


def sanitize_provider_names(values: Any) -> Any:
    """Neutralize provider-private entries one by one, preserving other entries."""
    if not isinstance(values, list):
        return values
    return [neutralize_provider_name(value) for value in values]


_BILLING_AND_USAGE_HEADER_PREFIXES = (
    "x-litellm-response-cost",
    "x-litellm-key-",
    "x-litellm-ratelimit-",
    "x-ratelimit-",
)


def _is_billing_or_usage_header(name: str) -> bool:
    return name.lower().startswith(_BILLING_AND_USAGE_HEADER_PREFIXES)


def sanitize_public_response_headers(headers: Mapping[str, Any], user_or_role: Any) -> dict[str, str]:
    """Drop provider-private identity headers for non-admin callers.

    Billing, quota and usage headers are intentionally retained even if an
    upstream callback happened to put a provider marker in their value.
    """
    if is_full_proxy_admin(user_or_role):
        return {str(key): str(value) for key, value in headers.items() if value is not None}
    result: dict[str, str] = {}
    for key, value in headers.items():
        if value is None:
            continue
        header_name = str(key)
        # A provider marker in the *name* is always private. This check must
        # precede the billing/usage allowlist so ``X-RateLimit-LibTV-Debug``
        # cannot masquerade as an allowed quota header. Clean billing/usage
        # names may still carry provider-marked values from upstream callbacks.
        if contains_private_provider_marker(header_name):
            continue
        if _is_billing_or_usage_header(header_name):
            result[header_name] = str(value)
            continue
        if contains_private_provider_marker(value):
            continue
        result[header_name] = str(value)
    return result


def finalize_public_response_headers(headers: Mapping[str, Any], user_or_role: Any) -> dict[str, str]:
    """Sanitize the complete header map immediately before it is written.

    Callback hooks run after the normal response-header builder and can add or
    override entries. Mutating mutable containers here makes this safe for both
    response dictionaries and Starlette response header objects.
    """
    sanitized = sanitize_public_response_headers(headers, user_or_role)
    if isinstance(headers, MutableMapping):
        headers.clear()
        headers.update(sanitized)
    elif all(hasattr(headers, attr) for attr in ("keys", "__delitem__", "update")):
        # Starlette's MutableHeaders deliberately is not a MutableMapping but
        # supports the same key mutation operations.
        for key in list(headers.keys()):
            del headers[key]  # type: ignore[index]
        headers.update(sanitized)  # type: ignore[attr-defined]
    return sanitized


def sanitize_health_endpoint(endpoint: Any) -> Any:
    """Remove provider-private health fields while preserving health status."""
    if not isinstance(endpoint, Mapping):
        return endpoint
    sanitized: dict[str, Any] = {}
    marker_sensitive_fields = {
        "model",
        "model_id",
        "model_name",
        "provider",
        "custom_llm_provider",
        "litellm_provider",
        "deployment",
        "deployment_id",
        "api_base",
        "api_version",
    }
    for key, value in endpoint.items():
        if key in {"api_base", "api_version"}:
            continue
        if key in marker_sensitive_fields and contains_private_provider_marker(value):
            continue
        if isinstance(value, str) and contains_private_provider_marker(value):
            if key in {"error", "error_message", "message", "detail"}:
                sanitized[key] = "provider health check failed"
            continue
        if isinstance(value, Mapping):
            sanitized[key] = sanitize_health_endpoint(value)
            continue
        if isinstance(value, list):
            sanitized[key] = [
                sanitize_health_endpoint(item) if isinstance(item, Mapping) else item for item in value
            ]
            continue
        sanitized[key] = value
    return sanitized


def sanitize_public_health_checks(checks_data: Mapping[Any, Any], user_or_role: Any) -> dict:
    """Sanitize the keyed ``/health/latest`` payload for non-full admins."""
    if is_full_proxy_admin(user_or_role):
        return dict(checks_data)
    return {
        key: sanitize_health_endpoint(value)
        for key, value in checks_data.items()
        if not contains_private_provider_marker(key)
    }
