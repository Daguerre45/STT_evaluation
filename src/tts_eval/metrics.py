from dataclasses import dataclass
from itertools import pairwise

from .audio import AudioAnalysis
from .languages import EvaluationLanguage
from .schemas import (
    AudioAnomaly,
    FluencyMetrics,
    InterwordPause,
    MetricScore,
    TextFidelityMetrics,
    WordTiming,
)
from .text import sentence_end_word_indexes, tokenize, word_error_counts


@dataclass(frozen=True)
class DerivedMetrics:
    fluency: FluencyMetrics
    fidelity: TextFidelityMetrics


def derive_metrics(
    reference_text: str,
    transcript: str,
    timings: list[WordTiming],
    audio: AudioAnalysis,
    min_pause_seconds: float = 0.15,
    brief_pause_seconds: float = 0.25,
    long_pause_seconds: float = 0.70,
    misplaced_pause_seconds: float = 0.45,
) -> DerivedMetrics:
    expected, recognized = tokenize(reference_text), tokenize(transcript)
    edits = word_error_counts(expected, recognized)
    wer = edits.total / max(len(expected), 1)
    gaps = [(index, word, next_word, max(0.0, next_word.start - word.end)) for index, (word, next_word) in enumerate(pairwise(timings))]
    pause_gaps = [gap for gap in gaps if gap[3] >= min_pause_seconds]
    long_pause_count = sum(gap[3] >= long_pause_seconds for gap in pause_gaps)
    allowed_indexes = sentence_end_word_indexes(reference_text)
    pauses = [
        InterwordPause(
            after_word=word.word,
            before_word=next_word.word,
            start_seconds=round(word.end, 3),
            end_seconds=round(next_word.start, 3),
            duration_seconds=round(duration, 3),
            category=(
                "long"
                if duration >= long_pause_seconds
                else "brief"
                if duration >= brief_pause_seconds
                else "micro"
            ),
            expected_at_boundary=index in allowed_indexes or word.word.rstrip().endswith((".", "!", "?", ";", ":")),
        )
        for index, word, next_word, duration in pause_gaps
    ]
    misplaced = sum(
        not pause.expected_at_boundary
        for pause in pauses
        if pause.duration_seconds >= misplaced_pause_seconds
    )
    active_seconds = max(audio.duration_seconds - sum(pause.duration_seconds for pause in pauses), 0.001)
    wpm = len(recognized) / max(audio.duration_seconds, 0.001) * 60 if recognized else None
    articulation = len(recognized) / active_seconds * 60 if recognized else None
    return DerivedMetrics(
        fidelity=TextFidelityMetrics(
            expected_words=len(expected), recognized_words=len(recognized), word_error_rate=round(wer, 4),
            substitutions=edits.substitutions, deletions=edits.deletions, insertions=edits.insertions,
        ),
        fluency=FluencyMetrics(
            words_per_minute=round(wpm, 2) if wpm else None,
            articulation_words_per_minute=round(articulation, 2) if articulation else None,
            pause_count=len(pauses), long_pause_count=long_pause_count,
            total_pause_seconds=round(sum(pause.duration_seconds for pause in pauses), 3),
            misplaced_pause_count=misplaced, pauses=pauses,
        ),
    )


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def score_all(
    audio: AudioAnalysis,
    metrics: DerivedMetrics,
    language: EvaluationLanguage,
    anomalies: list[AudioAnomaly] | None = None,
) -> dict[str, MetricScore]:
    fidelity_score = _clamp(100 - metrics.fidelity.word_error_rate * 100)
    fidelity_notes = [f"WER lexical: {metrics.fidelity.word_error_rate:.1%}."]
    if metrics.fidelity.deletions:
        fidelity_notes.append(f"{metrics.fidelity.deletions} palabras esperadas no reconocidas.")

    rhythm_score = 100 - metrics.fluency.long_pause_count * 8 - metrics.fluency.misplaced_pause_count * 7
    rhythm_notes = [
        f"{metrics.fluency.pause_count} pausas detectadas; {metrics.fluency.long_pause_count} largas.",
    ]
    if metrics.fluency.words_per_minute:
        rhythm_notes.append(f"Velocidad: {metrics.fluency.words_per_minute:.0f} palabras/minuto.")

    prosody_score = 78.0
    prosody_notes: list[str] = []
    if audio.f0_range_semitones is None:
        prosody_score = 30
        prosody_notes.append("No se pudo extraer F0 de forma fiable.")
    else:
        prosody_notes.append(f"Rango tonal P10-P90: {audio.f0_range_semitones:.1f} semitonos.")
        if audio.f0_range_semitones < 2.2:
            prosody_score -= 28
            prosody_notes.append("Variacion tonal muy baja para una voz expresiva.")
        elif audio.f0_range_semitones < 3.5:
            prosody_score -= 12
            prosody_notes.append("Variacion tonal moderadamente baja.")
        elif audio.f0_range_semitones > 16:
            prosody_score -= 15
            prosody_notes.append("Variacion tonal inusualmente alta; revisar saltos de F0.")
    prosody_score -= metrics.fluency.misplaced_pause_count * 4

    quality_score = 100.0
    quality_notes = [f"Nivel RMS: {audio.rms_dbfs:.1f} dBFS."]
    if audio.clipping_ratio > 0.0001:
        quality_score -= min(55, audio.clipping_ratio * 10000)
        quality_notes.append(f"Clipping: {audio.clipping_ratio:.3%} de muestras.")
    if audio.rms_dbfs < -32:
        quality_score -= 18
        quality_notes.append("Nivel de audio bajo.")
    if audio.leading_silence_seconds > 1.0 or audio.trailing_silence_seconds > 1.0:
        quality_score -= 8
        quality_notes.append("Silencio inicial o final prolongado.")

    pronunciation_score = _clamp(fidelity_score * 0.8 + min(20, 20 * audio.voiced_seconds / max(audio.duration_seconds, 0.1)))
    pronunciation_notes = [
        "Indicador inicial: combina fidelidad lexical y proporcion de voz.",
        f"No equivale todavia a una puntuacion fonetica GOP validada para {language}.",
    ]

    # Aplicar ajustes por anomalías locales detectadas
    if anomalies:
        pitch_jumps = [a for a in anomalies if a.type == "pitch_jump"]
        if pitch_jumps:
            penalty = min(25.0, sum(10.0 if a.severity == "alta" else 5.0 for a in pitch_jumps))
            prosody_score -= penalty
            prosody_notes.append(f"{len(pitch_jumps)} quiebre(s)/salto(s) de entonación abrupto(s) (-{penalty:.0f} pts).")

        rate_anomalies = [a for a in anomalies if a.type in ("rushed_word", "stretched_word")]
        if rate_anomalies:
            penalty = min(20.0, sum(6.0 if a.severity == "alta" else 3.0 for a in rate_anomalies))
            rhythm_score -= penalty
            rhythm_notes.append(f"{len(rate_anomalies)} anomalía(s) de ritmo local (palabras atropelladas o estiradas) (-{penalty:.0f} pts).")

        low_confs = [a for a in anomalies if a.type == "low_confidence"]
        if low_confs:
            penalty = min(25.0, sum(7.0 if a.severity == "alta" else 4.0 for a in low_confs))
            pronunciation_score = _clamp(pronunciation_score - penalty)
            pronunciation_notes.append(f"{len(low_confs)} palabra(s) con baja inteligibilidad/confianza (-{penalty:.0f} pts).")

    return {
        "fidelity": MetricScore(score=fidelity_score, rationale=fidelity_notes),
        "rhythm": MetricScore(score=_clamp(rhythm_score), rationale=rhythm_notes),
        "prosody": MetricScore(score=_clamp(prosody_score), rationale=prosody_notes),
        "audio_quality": MetricScore(score=_clamp(quality_score), rationale=quality_notes),
        "pronunciation_risk": MetricScore(score=pronunciation_score, rationale=pronunciation_notes),
    }


def overall_score(scores: dict[str, MetricScore]) -> float:
    weights = {"fidelity": 0.30, "rhythm": 0.20, "prosody": 0.25, "audio_quality": 0.15, "pronunciation_risk": 0.10}
    return round(sum(scores[name].score * weight for name, weight in weights.items()), 1)
