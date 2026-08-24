"""Provider-private model response serialization without proxy imports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_PUBLIC_PROVIDER_MODEL_INFO_KEYS = frozenset(
    {
        "mode",
        "max_input_tokens",
        "max_output_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_second",
        "output_cost_per_second",
        "supports_function_calling",
        "supports_parallel_function_calling",
        "supports_vision",
        "supports_reasoning",
        "supported_openai_params",
        "team_public_model_name",
        "team_id",
        "access_via_team_ids",
        "direct_access",
    }
)


def is_provider_private_model(model: dict[str, Any]) -> bool:
    params = model.get("litellm_params") or {}
    info = model.get("model_info") or {}
    provider = params.get("custom_llm_provider") or info.get("litellm_provider")
    routed_model = str(params.get("model") or "")
    return str(provider or "").lower() == "libtv" or routed_model.lower().startswith("libtv/")


def serialize_public_provider_model(
    model: dict[str, Any], translate_model_name: Callable[[dict[str, Any]], dict[str, Any]] | None = None
) -> dict[str, Any]:
    if not isinstance(model, dict) or not is_provider_private_model(model):
        return model
    translated = translate_model_name(model) if translate_model_name is not None else model
    info = translated.get("model_info") or {}
    public_info = {key: value for key, value in info.items() if key in _PUBLIC_PROVIDER_MODEL_INFO_KEYS}
    return {"model_name": translated.get("model_name"), "model_info": public_info}


def serialize_public_provider_models(
    models: list[dict[str, Any]],
    is_full_proxy_admin: bool,
    translate_model_name: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if is_full_proxy_admin:
        return models
    serialized: list[dict[str, Any]] = []
    seen_public_names: set[str] = set()
    for model in models:
        item = serialize_public_provider_model(model, translate_model_name)
        public_name = item.get("model_name") if isinstance(item, dict) else None
        if is_provider_private_model(model) and isinstance(public_name, str):
            if public_name in seen_public_names:
                continue
            seen_public_names.add(public_name)
        serialized.append(item)
    return serialized
