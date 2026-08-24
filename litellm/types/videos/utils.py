"""
Utility functions for video ID encoding/decoding with provider information.

Follows the pattern used in responses/utils.py for consistency.
Format: vid_{base64_encoded_string}
"""

import base64
import importlib
from typing import Callable, Optional

from litellm._logging import verbose_logger
from litellm.types.utils import SpecialEnums
from litellm.types.videos.main import DecodedVideoId

VIDEO_ID_PREFIX = "video_"
CAUSYN_VIDEO_ID_PREFIX = "causyn_"
CHARACTER_ID_PREFIX = "character_"
CHARACTER_ID_TEMPLATE = "litellm:custom_llm_provider:{};model_id:{};character_id:{}"

# Provider codecs register themselves at provider-module import time.  Keeping
# this registry in the type layer lets the generic video router decode opaque
# IDs without importing proxy internals (or any concrete provider module).
_VIDEO_ID_CODECS: dict[str, tuple[str, Callable[[str], Optional[DecodedVideoId]]]] = {}
_VIDEO_ID_CODEC_MODULES = {"video_v2_": "litellm.llms.libtv.video_id_codec"}


def register_video_id_codec(
    provider: str, prefix: str, decoder: Callable[[str], Optional[DecodedVideoId]]
) -> None:
    """Register a provider-neutral decoder for a versioned public ID."""
    if provider and prefix and callable(decoder):
        _VIDEO_ID_CODECS[prefix] = (provider, decoder)


def _load_video_id_codec(prefix: str) -> None:
    """Lazy-load opaque codecs so cold-start SDK imports can decode IDs."""
    module_name = _VIDEO_ID_CODEC_MODULES.get(prefix)
    if module_name is not None and prefix not in _VIDEO_ID_CODECS:
        try:
            importlib.import_module(module_name)
        except Exception:
            return


class DecodedCharacterId(dict):
    """Structure representing a decoded character ID."""

    custom_llm_provider: Optional[str]
    model_id: Optional[str]
    character_id: str


def _add_base64_padding(value: str) -> str:
    """
    Add missing base64 padding when IDs are copied without trailing '=' chars.
    """
    missing_padding = len(value) % 4
    if missing_padding:
        value += "=" * (4 - missing_padding)
    return value


def encode_video_id_with_provider(video_id: str, provider: str, model_id: Optional[str] = None) -> str:
    """Encode provider and model_id into video_id using base64."""
    if not provider or not video_id:
        return video_id

    # Try to decode the ID first to check if it's already encoded
    # This handles the case where Azure/OpenAI return IDs that start with "video_"
    # but are not yet encoded with provider information
    decoded = decode_video_id_with_provider(video_id)
    if decoded.get("custom_llm_provider") is not None:
        # ID is already encoded, return as-is
        return video_id

    # ID is not encoded (even if it starts with video_), so encode it
    assembled_id = str(SpecialEnums.LITELLM_MANAGED_VIDEO_COMPLETE_STR.value).format(provider, model_id or "", video_id)

    base64_encoded_id: str = base64.b64encode(assembled_id.encode("utf-8")).decode("utf-8")

    return f"{VIDEO_ID_PREFIX}{base64_encoded_id}"


def decode_video_id_with_provider(encoded_video_id: str) -> DecodedVideoId:
    """Decode provider and model_id from encoded video_id."""
    if not encoded_video_id:
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    if encoded_video_id.startswith(CAUSYN_VIDEO_ID_PREFIX):
        return DecodedVideoId(
            custom_llm_provider="causyn",
            model_id="causyn-1-0",
            video_id=encoded_video_id,
        )

    if not encoded_video_id.startswith(VIDEO_ID_PREFIX):
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    for prefix, (_provider, decoder) in _VIDEO_ID_CODECS.items():
        if encoded_video_id.startswith(prefix):
            decoded = decoder(encoded_video_id)
            if decoded is not None:
                return decoded
            # Invalid opaque IDs must never fall through to the legacy parser;
            # doing so could turn attacker-controlled bytes into routing data.
            return DecodedVideoId(custom_llm_provider=None, model_id=None, video_id=encoded_video_id)

    for prefix in _VIDEO_ID_CODEC_MODULES:
        if encoded_video_id.startswith(prefix):
            _load_video_id_codec(prefix)
            registered = _VIDEO_ID_CODECS.get(prefix)
            if registered is None:
                return DecodedVideoId(custom_llm_provider=None, model_id=None, video_id=encoded_video_id)
            decoded = registered[1](encoded_video_id)
            return decoded or DecodedVideoId(
                custom_llm_provider=None, model_id=None, video_id=encoded_video_id
            )

    try:
        cleaned_id = encoded_video_id[len(VIDEO_ID_PREFIX) :]
        cleaned_id = _add_base64_padding(cleaned_id)
        decoded_id = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

        if ";" not in decoded_id:
            return DecodedVideoId(
                custom_llm_provider=None,
                model_id=None,
                video_id=encoded_video_id,
            )

        parts = decoded_id.split(";")

        custom_llm_provider = None
        model_id = None
        decoded_video_id = encoded_video_id

        if len(parts) >= 3:
            custom_llm_provider_part = parts[0]
            model_id_part = parts[1]
            video_id_part = parts[2]

            custom_llm_provider = custom_llm_provider_part.replace("litellm:custom_llm_provider:", "")
            model_id = model_id_part.replace("model_id:", "")
            decoded_video_id = video_id_part.replace("video_id:", "")

        return DecodedVideoId(
            custom_llm_provider=custom_llm_provider,
            model_id=model_id,
            video_id=decoded_video_id,
        )
    except Exception as e:
        verbose_logger.debug(f"Error decoding video_id '{encoded_video_id}': {e}")
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )


def extract_original_video_id(encoded_video_id: str) -> str:
    """Extract original video ID without encoding."""
    decoded = decode_video_id_with_provider(encoded_video_id)
    return decoded.get("video_id", encoded_video_id)


def encode_character_id_with_provider(character_id: str, provider: str, model_id: Optional[str] = None) -> str:
    """Encode provider and model_id into character_id using base64."""
    if not provider or not character_id:
        return character_id

    decoded = decode_character_id_with_provider(character_id)
    if decoded.get("custom_llm_provider") is not None:
        return character_id

    assembled_id = CHARACTER_ID_TEMPLATE.format(provider, model_id or "", character_id)
    base64_encoded_id: str = base64.b64encode(assembled_id.encode("utf-8")).decode("utf-8")
    return f"{CHARACTER_ID_PREFIX}{base64_encoded_id}"


def decode_character_id_with_provider(encoded_character_id: str) -> DecodedCharacterId:
    """Decode provider and model_id from encoded character_id."""
    if not encoded_character_id:
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )

    if not encoded_character_id.startswith(CHARACTER_ID_PREFIX):
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )

    try:
        cleaned_id = encoded_character_id.replace(CHARACTER_ID_PREFIX, "")
        cleaned_id = _add_base64_padding(cleaned_id)
        decoded_id = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

        if ";" not in decoded_id:
            return DecodedCharacterId(
                custom_llm_provider=None,
                model_id=None,
                character_id=encoded_character_id,
            )

        parts = decoded_id.split(";")

        custom_llm_provider = None
        model_id = None
        decoded_character_id = encoded_character_id

        if len(parts) >= 3:
            custom_llm_provider_part = parts[0]
            model_id_part = parts[1]
            character_id_part = parts[2]

            custom_llm_provider = custom_llm_provider_part.replace("litellm:custom_llm_provider:", "")
            model_id = model_id_part.replace("model_id:", "")
            decoded_character_id = character_id_part.replace("character_id:", "")

        return DecodedCharacterId(
            custom_llm_provider=custom_llm_provider,
            model_id=model_id,
            character_id=decoded_character_id,
        )
    except Exception as e:
        verbose_logger.debug(f"Error decoding character_id '{encoded_character_id}': {e}")
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )


def extract_original_character_id(encoded_character_id: str) -> str:
    """Extract original character ID without encoding."""
    decoded = decode_character_id_with_provider(encoded_character_id)
    return decoded.get("character_id", encoded_character_id)
