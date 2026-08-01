from .handler import XiaoyunqueLLM

# Module-level instance referenced by the proxy config's
# litellm_settings.custom_provider_map[].custom_handler import path.
xiaoyunque_proxy_handler = XiaoyunqueLLM()

__all__ = ["XiaoyunqueLLM", "xiaoyunque_proxy_handler"]
