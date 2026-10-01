import re
import shutil
import subprocess
from pathlib import Path

from .config import Settings
from .languages import EvaluationLanguage
from .schemas import WordTiming
from .text import normalize_text

_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s+xmin = (?P<start>[\d.]+)\s+xmax = (?P<end>[\d.]+)\s+text = \"(?P<word>[^\"]*)\"",
    re.MULTILINE,
)
_WORDS_TIER_RE = re.compile(
    r'name = "words"(?P<contents>.*?)(?=\n\s*item \[\d+\]:|\Z)', re.DOTALL
)


def mfa_resources(settings: Settings, language: EvaluationLanguage) -> tuple[str | None, str | None]:
    if language == "pt-PT":
        return settings.mfa_ptpt_acoustic_model, settings.mfa_ptpt_dictionary
    return settings.mfa_eses_acoustic_model, settings.mfa_eses_dictionary


def align_with_mfa(
    audio_path: Path, reference_text: str, language: EvaluationLanguage, settings: Settings
) -> list[WordTiming] | None:
    """Run MFA only when explicit resources for the selected language are configured.

    MFA produces TextGrid files. The parser intentionally only consumes labelled intervals;
    callers keep Whisper times as a fallback if the installed model has a different layout.
    """
    acoustic_model, dictionary = mfa_resources(settings, language)
    if not (settings.mfa_enabled and acoustic_model and dictionary):
        return None
    if not (Path(settings.mfa_command).is_file() or shutil.which(settings.mfa_command)):
        raise RuntimeError(f"No se encontro el ejecutable MFA configurado: {settings.mfa_command}")
    job_dir = settings.work_dir / f"mfa-{audio_path.stem}"
    corpus_dir = job_dir / "corpus"
    output_dir = job_dir / "aligned"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_audio = corpus_dir / f"utterance{audio_path.suffix}"
    corpus_text = corpus_dir / "utterance.lab"
    corpus_audio.write_bytes(audio_path.read_bytes())
    corpus_text.write_text(normalize_text(reference_text), encoding="utf-8")
    command = [
        settings.mfa_command, "align", str(corpus_dir), dictionary, acoustic_model,
        str(output_dir), "--clean", "--quiet",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr else "error desconocido"
        raise RuntimeError(f"MFA no pudo alinear el audio: {detail}")
    grids = list(output_dir.rglob("*.TextGrid"))
    if not grids:
        return None
    contents = grids[0].read_text(encoding="utf-8", errors="replace")
    words_tier = _WORDS_TIER_RE.search(contents)
    if words_tier is None:
        return None
    timings = [
        WordTiming(word=match.group("word"), start=round(float(match.group("start")), 3), end=round(float(match.group("end")), 3))
        for match in _INTERVAL_RE.finditer(words_tier.group("contents"))
        if match.group("word").strip() and not match.group("word").startswith("<")
    ]
    return timings or None
