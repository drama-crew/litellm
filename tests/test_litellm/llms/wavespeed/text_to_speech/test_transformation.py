import litellm
from litellm.llms.wavespeed.text_to_speech.transformation import (
    WaveSpeedTextToSpeechConfig,
    is_wavespeed_speech_model,
)


def test_is_wavespeed_speech_model_matches_bare_and_prefixed_ids():
    assert is_wavespeed_speech_model("minimax-speech-2.8-turbo") is True
    assert is_wavespeed_speech_model("minimax-speech-2.8-hd") is True
    assert is_wavespeed_speech_model("openai/minimax-speech-2.8-turbo") is True


def test_is_wavespeed_speech_model_rejects_unrelated_models():
    assert is_wavespeed_speech_model("tts-1-hd") is False
    assert is_wavespeed_speech_model("tts-1") is False
    assert is_wavespeed_speech_model("gpt-4o-mini-tts") is False


def test_litellm_dispatches_wavespeed_config_for_minimax_speech_models():
    config = litellm.ProviderConfigManager.get_provider_text_to_speech_config(
        model="minimax-speech-2.8-turbo",
        provider=litellm.LlmProviders.OPENAI,
    )
    assert isinstance(config, WaveSpeedTextToSpeechConfig)

    config = litellm.ProviderConfigManager.get_provider_text_to_speech_config(
        model="minimax-speech-2.8-hd",
        provider=litellm.LlmProviders.OPENAI,
    )
    assert isinstance(config, WaveSpeedTextToSpeechConfig)


def test_litellm_does_not_dispatch_wavespeed_config_for_unrelated_openai_tts_models():
    config = litellm.ProviderConfigManager.get_provider_text_to_speech_config(
        model="tts-1-hd",
        provider=litellm.LlmProviders.OPENAI,
    )
    assert config is None


def test_map_openai_params_moves_whitelisted_kwargs_into_extra_body():
    config = WaveSpeedTextToSpeechConfig()
    voice, mapped = config.map_openai_params(
        model="minimax-speech-2.8-turbo",
        optional_params={"response_format": "wav", "speed": 1.2},
        voice="alloy",
        drop_params=False,
        kwargs={
            "voice_id": "female-yujie",
            "emotion": "happy",
            "pitch": 2,
            "volume": 1.5,
            "language_boost": "Chinese",
            "sample_rate": 32000,
            "bitrate": 128000,
            "channel": "mono",
            "pronunciation_dict": ["西恩/xi1 en1"],
            "english_normalization": True,
            "unrelated_kwarg": "should-not-leak",
        },
    )
    assert voice == "alloy"
    assert mapped["response_format"] == "wav"
    assert mapped["speed"] == 1.2
    assert mapped["extra_body"] == {
        "voice_id": "female-yujie",
        "emotion": "happy",
        "pitch": 2,
        "volume": 1.5,
        "language_boost": "Chinese",
        "sample_rate": 32000,
        "bitrate": 128000,
        "channel": "mono",
        "pronunciation_dict": ["西恩/xi1 en1"],
        "english_normalization": True,
    }


def test_map_openai_params_omits_none_valued_whitelisted_kwargs():
    config = WaveSpeedTextToSpeechConfig()
    _voice, mapped = config.map_openai_params(
        model="minimax-speech-2.8-turbo",
        optional_params={},
        voice="alloy",
        drop_params=False,
        kwargs={"voice_id": "female-yujie", "emotion": None},
    )
    assert mapped["extra_body"] == {"voice_id": "female-yujie"}


def test_map_openai_params_merges_with_caller_supplied_extra_body():
    config = WaveSpeedTextToSpeechConfig()
    _voice, mapped = config.map_openai_params(
        model="minimax-speech-2.8-turbo",
        optional_params={},
        voice="alloy",
        drop_params=False,
        kwargs={"voice_id": "female-yujie", "extra_body": {"custom_field": "kept"}},
    )
    assert mapped["extra_body"] == {"custom_field": "kept", "voice_id": "female-yujie"}


def test_map_openai_params_no_whitelisted_kwargs_leaves_optional_params_untouched():
    config = WaveSpeedTextToSpeechConfig()
    voice, mapped = config.map_openai_params(
        model="minimax-speech-2.8-turbo",
        optional_params={"response_format": "wav"},
        voice="alloy",
        drop_params=False,
        kwargs={},
    )
    assert voice == "alloy"
    assert mapped == {"response_format": "wav"}
    assert "extra_body" not in mapped
