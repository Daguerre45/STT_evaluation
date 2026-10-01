import tempfile
import uuid
from pathlib import Path

import httpx

from .alignment import align_with_mfa, mfa_resources
from .anomalies import detect_all_anomalies
from .audio import analyze_audio, prepare_for_analysis
from .config import Settings
from .languages import EvaluationLanguage
from .local_llm import redact_with_ollama
from .metrics import derive_metrics, overall_score, score_all
from .phonemes import expected_phonemes
from .reporting import deterministic_report
from .schemas import AudioMetrics, EvaluationResponse, ProsodyMetrics
from .squim import compute_squim_metrics
from .transcription import release_whisper_model, transcribe


class EvaluationPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate(
        self, audio_path: Path, reference_text: str, language: EvaluationLanguage = "pt-PT"
    ) -> EvaluationResponse:
        if not reference_text.strip():
            raise ValueError("reference_text no puede estar vacio")
        with tempfile.TemporaryDirectory(prefix="tts-eval-") as directory:
            prepared_path = Path(directory) / "prepared.wav"
            prepare_for_analysis(audio_path, prepared_path, self.settings.ffmpeg_bin)
            analysis = analyze_audio(prepared_path)
            transcript, timings = transcribe(prepared_path, language, self.settings)
            warnings: list[str] = []
            try:
                mfa_timings = align_with_mfa(prepared_path, reference_text, language, self.settings)
                if mfa_timings:
                    timings = mfa_timings
            except RuntimeError as exc:
                warnings.append(str(exc))
            acoustic_model, dictionary = mfa_resources(self.settings, language)
            if not self.settings.mfa_enabled:
                warnings.append("Alineamiento MFA no habilitado; se usan marcas por palabra de Whisper.")
            elif not (acoustic_model and dictionary):
                warnings.append(f"MFA esta habilitado, pero faltan recursos para {language}; se usan marcas de Whisper.")
            derived = derive_metrics(
                reference_text,
                transcript,
                timings,
                analysis,
                min_pause_seconds=self.settings.min_pause_seconds,
                brief_pause_seconds=self.settings.brief_pause_seconds,
                long_pause_seconds=self.settings.long_pause_seconds,
                misplaced_pause_seconds=self.settings.misplaced_pause_seconds,
            )
            squim_metrics = compute_squim_metrics(prepared_path, self.settings)
            initial_scores = score_all(analysis, derived, language)
            initial_overall = overall_score(initial_scores)
            anomalies, verdict = detect_all_anomalies(
                audio=analysis,
                timings=timings,
                reference_text=reference_text,
                transcript=transcript,
                pauses=derived.fluency.pauses,
                settings=self.settings,
                wer=derived.fidelity.word_error_rate,
                overall_score=initial_overall,
            )
            scores = score_all(analysis, derived, language, anomalies=anomalies)
            phonemes = expected_phonemes(reference_text, language)
            if phonemes is None:
                warnings.append(f"No se generaron fonemas: instala eSpeak NG para habilitar el analisis {language}.")
            result = EvaluationResponse(
                language=language, transcript=transcript, expected_phonemes=phonemes, word_timings=timings,
                audio=AudioMetrics(
                    duration_seconds=analysis.duration_seconds, rms_dbfs=analysis.rms_dbfs,
                    clipping_ratio=analysis.clipping_ratio, leading_silence_seconds=analysis.leading_silence_seconds,
                    trailing_silence_seconds=analysis.trailing_silence_seconds,
                ),
                prosody=ProsodyMetrics(
                    voiced_seconds=analysis.voiced_seconds, f0_median_hz=analysis.f0_median_hz,
                    f0_range_semitones=analysis.f0_range_semitones, f0_std_semitones=analysis.f0_std_semitones,
                    mean_energy_db=analysis.mean_energy_db,
                ),
                fluency=derived.fluency, fidelity=derived.fidelity, scores=scores,
                overall_score=overall_score(scores), verdict=verdict, anomalies=anomalies, squim=squim_metrics,
                report="", warnings=warnings,
            )
            release_whisper_model()
            result.report = deterministic_report(result)
            try:
                local_report = redact_with_ollama(result, self.settings)
                if local_report:
                    result.report = local_report
            except (OSError, ValueError, httpx.HTTPError) as exc:
                warnings.append(f"No se pudo redactar con el LLM local: {exc}")
            return result

    def stage_upload(self, source: Path) -> Path:
        destination_dir = self.settings.work_dir / str(uuid.uuid4())
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        destination.write_bytes(source.read_bytes())
        return destination
