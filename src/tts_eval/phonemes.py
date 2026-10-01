import os
import shutil

from .languages import PHONEMIZER_LANGUAGE, EvaluationLanguage
from .text import strip_ssml

# Auto-detect eSpeak NG on standard Windows paths if not present in PATH
if not shutil.which("espeak-ng"):
    possible_paths = [
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ["PHONEMIZER_ESPEAK_PATH"] = path
            os.environ["PATH"] = os.path.dirname(path) + os.path.pathsep + os.environ.get("PATH", "")
            break


def expected_phonemes(text: str, language: EvaluationLanguage) -> str | None:
    """Best-effort phonemization. Absence of eSpeak must not block evaluation."""
    try:
        from phonemizer import phonemize

        return phonemize(
            strip_ssml(text), language=PHONEMIZER_LANGUAGE[language], backend="espeak", strip=True,
            preserve_punctuation=True, njobs=1,
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        return None

