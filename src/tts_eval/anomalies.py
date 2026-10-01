import math

from .audio import AudioAnalysis
from .config import Settings
from .schemas import AudioAnomaly, InterwordPause, WordTiming
from .syllables import count_syllables
from .text import align_tokens, tokenize


def _detect_pitch_jumps(
    pitch_frames: tuple[tuple[float, float], ...],
    timings: list[WordTiming],
    threshold_semitones: float,
) -> list[AudioAnomaly]:
    if len(pitch_frames) < 2:
        return []

    jumps: list[tuple[float, float, float, float, float]] = []  # t1, t2, f1, f2, semi
    for i in range(len(pitch_frames) - 1):
        t1, f1 = pitch_frames[i]
        t2, f2 = pitch_frames[i + 1]
        dt = t2 - t1
        if 0 < dt <= 0.035 and f1 > 0 and f2 > 0:
            semi = 12 * abs(math.log2(f2 / f1))
            if semi >= threshold_semitones:
                jumps.append((t1, t2, f1, f2, semi))

    if not jumps:
        return []

    # Group close jumps (< 0.15s apart) into a single anomaly
    grouped: list[list[tuple[float, float, float, float, float]]] = []
    current_group: list[tuple[float, float, float, float, float]] = [jumps[0]]
    for jump in jumps[1:]:
        if jump[0] - current_group[-1][1] <= 0.15:
            current_group.append(jump)
        else:
            grouped.append(current_group)
            current_group = [jump]
    if current_group:
        grouped.append(current_group)

    anomalies: list[AudioAnomaly] = []
    for group in grouped:
        max_jump = max(group, key=lambda x: x[4])
        t_start = group[0][0]
        t_end = group[-1][1]
        semi = max_jump[4]
        f1, f2 = max_jump[2], max_jump[3]

        # Find corresponding word if any
        associated_word = None
        for w in timings:
            if w.start <= t_start <= w.end or w.start <= t_end <= w.end or (t_start <= w.start and t_end >= w.end):
                associated_word = w.word
                break

        severity = "alta" if semi >= threshold_semitones + 2.5 else "media"
        anomalies.append(
            AudioAnomaly(
                timestamp_start=round(t_start, 3),
                timestamp_end=round(t_end, 3),
                word=associated_word,
                type="pitch_jump",
                severity=severity,
                evidence=f"Salto abrupto de {semi:.1f} semitonos ({f1:.0f} Hz -> {f2:.0f} Hz)",
                detail=(
                    "Quiebre de entonación o salto brusco de tono"
                    + (f" en la palabra '{associated_word}'" if associated_word else "")
                    + "; puede sonar a 'gallo', caída artificial o fallo de entonación del TTS."
                ),
            )
        )
    return anomalies


def _detect_word_speed_anomalies(
    timings: list[WordTiming],
    rushed_threshold: float,
    stretched_threshold: float,
) -> list[AudioAnomaly]:
    anomalies: list[AudioAnomaly] = []
    for i, item in enumerate(timings):
        clean_word = item.word.strip().strip(".,!?;:\"'()[]{}")
        syl = count_syllables(clean_word)
        dur = item.end - item.start
        if syl < 2 or dur <= 0.02:
            continue
        rate = syl / dur

        if rate >= rushed_threshold:
            severity = "alta" if rate >= rushed_threshold + 3.0 else "media"
            anomalies.append(
                AudioAnomaly(
                    timestamp_start=round(item.start, 3),
                    timestamp_end=round(item.end, 3),
                    word=item.word,
                    type="rushed_word",
                    severity=severity,
                    evidence=f"{syl} sílabas en {dur:.2f}s ({rate:.1f} sílabas/s)",
                    detail=f"La palabra '{item.word}' se pronuncia excesivamente rápido ({rate:.1f} sílabas/s); suena apresurada o atropellada.",
                )
            )
        elif rate <= stretched_threshold and dur >= 0.75:
            # Check if it's the last word (lengthening at end of sentence is often natural)
            is_sentence_final = (i == len(timings) - 1) or item.word.endswith((".", "!", "?"))
            if not is_sentence_final:
                severity = "alta" if rate <= 1.2 else "media"
                anomalies.append(
                    AudioAnomaly(
                        timestamp_start=round(item.start, 3),
                        timestamp_end=round(item.end, 3),
                        word=item.word,
                        type="stretched_word",
                        severity=severity,
                        evidence=f"{syl} sílabas en {dur:.2f}s ({rate:.1f} sílabas/s)",
                        detail=f"La palabra '{item.word}' tiene una duración anormalmente prolongada ({dur:.2f}s); suena arrastrada o artificialmente estirada.",
                    )
                )
    return anomalies


def _detect_low_confidence_words(
    timings: list[WordTiming],
    threshold: float,
) -> list[AudioAnomaly]:
    anomalies: list[AudioAnomaly] = []
    for item in timings:
        if item.confidence is not None and item.confidence < threshold:
            severity = "alta" if item.confidence < 0.40 else "media"
            anomalies.append(
                AudioAnomaly(
                    timestamp_start=round(item.start, 3),
                    timestamp_end=round(item.end, 3),
                    word=item.word,
                    type="low_confidence",
                    severity=severity,
                    evidence=f"Confianza de reconocimiento Whisper: {item.confidence:.1%}",
                    detail=f"Posible articulación defectuosa o sonido anómalo en la palabra '{item.word}'. Whisper tuvo dificultades para reconocerla con certeza.",
                )
            )
    return anomalies


def _detect_lexical_mismatches(
    reference_text: str,
    transcript: str,
    timings: list[WordTiming],
) -> list[AudioAnomaly]:
    ref_tokens = tokenize(reference_text)
    hyp_tokens = tokenize(transcript)
    ops = align_tokens(ref_tokens, hyp_tokens)

    anomalies: list[AudioAnomaly] = []
    for op in ops:
        if op.op == "substitution":
            t_start = 0.0
            t_end = 0.0
            if op.hyp_index is not None and op.hyp_index < len(timings):
                t_start = timings[op.hyp_index].start
                t_end = timings[op.hyp_index].end
            anomalies.append(
                AudioAnomaly(
                    timestamp_start=round(t_start, 3),
                    timestamp_end=round(t_end, 3),
                    word=op.hyp_word,
                    type="word_substitution",
                    severity="alta",
                    evidence=f"Esperado: '{op.ref_word}' | Reconocido: '{op.hyp_word}'",
                    detail=f"Discrepancia textual: el audio reproduce '{op.hyp_word}' en lugar de la palabra esperada '{op.ref_word}'.",
                )
            )
        elif op.op == "deletion":
            # Estimate timestamp from neighbor words
            t_pos = 0.0
            if op.ref_index is not None and timings:
                approx_idx = min(op.ref_index, len(timings) - 1)
                t_pos = timings[approx_idx].start
            anomalies.append(
                AudioAnomaly(
                    timestamp_start=round(t_pos, 3),
                    timestamp_end=round(t_pos + 0.2, 3),
                    word=op.ref_word,
                    type="word_omission",
                    severity="alta",
                    evidence=f"Palabra esperada no articulada: '{op.ref_word}'",
                    detail=f"La palabra '{op.ref_word}' fue omitida por el TTS o no llegó a sintetizarse.",
                )
            )
        elif op.op == "insertion":
            t_start = 0.0
            t_end = 0.0
            if op.hyp_index is not None and op.hyp_index < len(timings):
                t_start = timings[op.hyp_index].start
                t_end = timings[op.hyp_index].end
            anomalies.append(
                AudioAnomaly(
                    timestamp_start=round(t_start, 3),
                    timestamp_end=round(t_end, 3),
                    word=op.hyp_word,
                    type="word_hallucination",
                    severity="alta",
                    evidence=f"Palabra no prevista en referencia: '{op.hyp_word}'",
                    detail=f"El audio contiene la palabra '{op.hyp_word}', la cual no estaba presente en el texto de referencia.",
                )
            )
    return anomalies


def _detect_pause_anomalies(
    pauses: list[InterwordPause],
    misplaced_threshold: float,
) -> list[AudioAnomaly]:
    anomalies: list[AudioAnomaly] = []
    for pause in pauses:
        if not pause.expected_at_boundary and pause.duration_seconds >= misplaced_threshold:
            severity = "alta" if pause.duration_seconds >= 0.70 else "media"
            anomalies.append(
                AudioAnomaly(
                    timestamp_start=round(pause.start_seconds, 3),
                    timestamp_end=round(pause.end_seconds, 3),
                    word=f"{pause.after_word} -> {pause.before_word}",
                    type="unexpected_pause",
                    severity=severity,
                    evidence=f"Pausa de {pause.duration_seconds:.2f}s entre '{pause.after_word}' y '{pause.before_word}'",
                    detail="Pausa injustificada dentro de una cláusula sintáctica que no finaliza en signo de puntuación.",
                )
            )
    return anomalies


def _detect_audio_quality_anomalies(audio: AudioAnalysis) -> list[AudioAnomaly]:
    anomalies: list[AudioAnomaly] = []
    if audio.clipping_ratio > 0.0001:
        severity = "alta" if audio.clipping_ratio >= 0.01 else "media"
        anomalies.append(
            AudioAnomaly(
                timestamp_start=0.0,
                timestamp_end=round(audio.duration_seconds, 3),
                word=None,
                type="audio_clipping",
                severity=severity,
                evidence=f"Clipping digital en {audio.clipping_ratio:.3%} de las muestras",
                detail="Saturación digital detectada en la forma de onda. Puede producir asperezas, chasquidos o pérdida de dinámica.",
            )
        )
    if audio.leading_silence_seconds > 1.2 or audio.trailing_silence_seconds > 1.2:
        anomalies.append(
            AudioAnomaly(
                timestamp_start=0.0,
                timestamp_end=round(audio.duration_seconds, 3),
                word=None,
                type="excessive_silence",
                severity="baja",
                evidence=f"Silencio inicial: {audio.leading_silence_seconds:.2f}s, final: {audio.trailing_silence_seconds:.2f}s",
                detail="Margen de silencio inicial o final excesivo alrededor de la voz sintetizada.",
            )
        )
    return anomalies


def calculate_verdict(
    anomalies: list[AudioAnomaly],
    wer: float,
    overall_score: float,
) -> str:
    high_count = sum(1 for a in anomalies if a.severity == "alta")
    medium_count = sum(1 for a in anomalies if a.severity == "media")

    if high_count >= 1 or medium_count >= 3 or wer > 0.20:
        return "probable_problema_tts"
    if medium_count >= 1 or overall_score < 80.0 or wer > 0.05:
        return "requiere_revision"
    return "sin_incidencias_claras"


def detect_all_anomalies(
    audio: AudioAnalysis,
    timings: list[WordTiming],
    reference_text: str,
    transcript: str,
    pauses: list[InterwordPause],
    settings: Settings,
    wer: float,
    overall_score: float,
) -> tuple[list[AudioAnomaly], str]:
    """Runs full human-like diagnostic checks to detect local and acoustic anomalies."""
    anomalies: list[AudioAnomaly] = []

    # 1. Pitch jumps
    anomalies.extend(_detect_pitch_jumps(audio.pitch_frames, timings, settings.pitch_jump_threshold_semitones))

    # 2. Rushed and stretched words
    anomalies.extend(_detect_word_speed_anomalies(timings, settings.rushed_syllable_rate, settings.stretched_syllable_rate))

    # 3. Low confidence words
    anomalies.extend(_detect_low_confidence_words(timings, settings.low_confidence_threshold))

    # 4. Lexical mismatches (omissions, substitutions, hallucinations)
    anomalies.extend(_detect_lexical_mismatches(reference_text, transcript, timings))

    # 5. Unexpected pauses
    anomalies.extend(_detect_pause_anomalies(pauses, settings.misplaced_pause_seconds))

    # 6. Audio quality clipping / silences
    anomalies.extend(_detect_audio_quality_anomalies(audio))

    # Sort anomalies chronologically by start timestamp
    anomalies.sort(key=lambda a: (a.timestamp_start, 0 if a.severity == "alta" else 1))

    verdict = calculate_verdict(anomalies, wer, overall_score)
    return anomalies, verdict
