from typing import Literal

EvaluationLanguage = Literal["pt-PT", "es-ES"]

WHISPER_LANGUAGE: dict[EvaluationLanguage, str] = {
    "pt-PT": "pt",
    "es-ES": "es",
}

PHONEMIZER_LANGUAGE: dict[EvaluationLanguage, str] = {
    "pt-PT": "pt-pt",
    "es-ES": "es",
}
