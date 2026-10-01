from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from .config import Settings
from .languages import WHISPER_LANGUAGE, EvaluationLanguage
from .schemas import WordTiming


@lru_cache(maxsize=1)
def _model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(path: Path, language: EvaluationLanguage, settings: Settings) -> tuple[str, list[WordTiming]]:
    model = _model(settings.whisper_model, settings.device, settings.compute_type)
    segments, _ = model.transcribe(
        str(path), language=WHISPER_LANGUAGE[language], task="transcribe", word_timestamps=True, vad_filter=True,
        beam_size=5, condition_on_previous_text=False,
    )
    transcript_parts: list[str] = []
    words: list[WordTiming] = []
    for segment in segments:
        transcript_parts.append(segment.text.strip())
        for word in segment.words or []:
            if word.start is not None and word.end is not None:
                words.append(
                    WordTiming(
                        word=word.word.strip(), start=round(word.start, 3), end=round(word.end, 3),
                        confidence=round(float(word.probability), 4) if word.probability is not None else None,
                    )
                )
    return " ".join(part for part in transcript_parts if part), words


def release_whisper_model() -> None:
    """Release VRAM before a second local model, such as Qwen, is loaded."""
    _model.cache_clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
