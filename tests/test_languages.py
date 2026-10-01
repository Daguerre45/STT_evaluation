from tts_eval.languages import PHONEMIZER_LANGUAGE, WHISPER_LANGUAGE


def test_supported_languages_have_model_mappings() -> None:
    assert WHISPER_LANGUAGE["pt-PT"] == "pt"
    assert WHISPER_LANGUAGE["es-ES"] == "es"
    assert PHONEMIZER_LANGUAGE["pt-PT"] == "pt-pt"
    assert PHONEMIZER_LANGUAGE["es-ES"] == "es"
