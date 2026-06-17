from .handler import LibTVLLM

# Module-level instance referenced by the proxy config's
# litellm_settings.custom_provider_map[].custom_handler import path.
libtv_proxy_handler = LibTVLLM()

__all__ = ["LibTVLLM", "libtv_proxy_handler"]
