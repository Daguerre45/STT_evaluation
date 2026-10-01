from tts_eval.anomalies import detect_all_anomalies
from tts_eval.audio import AudioAnalysis
from tts_eval.config import Settings
from tts_eval.reporting import deterministic_report
from tts_eval.schemas import (
    AudioMetrics,
    EvaluationResponse,
    FluencyMetrics,
    InterwordPause,
    ProsodyMetrics,
    SquimMetrics,
    TextFidelityMetrics,
    WordTiming,
)
from tts_eval.syllables import count_syllables
from tts_eval.text import align_tokens


def test_count_syllables() -> None:
    assert count_syllables("hola") == 2
    assert count_syllables("responsabilidad") >= 6
    assert count_syllables("día") == 2
    assert count_syllables("construcción") == 3
    assert count_syllables("bom") == 1
    assert count_syllables("português") >= 3


def test_align_tokens_substitution_and_omission() -> None:
    ref = ["hola", "amigo", "estas"]
    hyp = ["hola", "socio", "estas"]  # "amigo" -> "socio"
    ops = align_tokens(ref, hyp)

    op_types = [op.op for op in ops]
    assert "match" in op_types
    assert "substitution" in op_types
    sub = next(op for op in ops if op.op == "substitution")
    assert sub.ref_word == "amigo"
    assert sub.hyp_word == "socio"

    # Test deletion
    ref2 = ["hola", "como", "estas"]
    hyp2 = ["hola", "estas"]
    ops2 = align_tokens(ref2, hyp2)
    assert any(op.op == "deletion" and op.ref_word == "como" for op in ops2)


def test_detect_pitch_jump_anomaly() -> None:
    settings = Settings()
    # 100 Hz to 180 Hz in 20 ms -> 12 * log2(1.8) = ~10.2 semitones
    pitch_frames = (
        (0.50, 100.0),
        (0.52, 180.0),
        (0.54, 180.0),
    )
    audio = AudioAnalysis(
        duration_seconds=2.0,
        rms_dbfs=-18.0,
        clipping_ratio=0.0,
        leading_silence_seconds=0.1,
        trailing_silence_seconds=0.1,
        mean_energy_db=-18.0,
        voiced_seconds=1.5,
        f0_median_hz=120.0,
        f0_range_semitones=11.0,
        f0_std_semitones=2.0,
        pitch_frames=pitch_frames,
    )
    timings = [
        WordTiming(word="Hola", start=0.1, end=0.4, confidence=0.99),
        WordTiming(word="amigo", start=0.45, end=0.8, confidence=0.98),
    ]
    anomalies, verdict = detect_all_anomalies(
        audio=audio,
        timings=timings,
        reference_text="Hola amigo",
        transcript="Hola amigo",
        pauses=[],
        settings=settings,
        wer=0.0,
        overall_score=90.0,
    )
    assert any(a.type == "pitch_jump" for a in anomalies)
    pitch_anomaly = next(a for a in anomalies if a.type == "pitch_jump")
    assert pitch_anomaly.severity == "alta"
    assert pitch_anomaly.word == "amigo"
    assert verdict == "probable_problema_tts"


def test_detect_rushed_and_stretched_words() -> None:
    settings = Settings()
    audio = AudioAnalysis(
        duration_seconds=3.0,
        rms_dbfs=-18.0,
        clipping_ratio=0.0,
        leading_silence_seconds=0.1,
        trailing_silence_seconds=0.1,
        mean_energy_db=-18.0,
        voiced_seconds=2.5,
        f0_median_hz=130.0,
        f0_range_semitones=5.0,
        f0_std_semitones=1.0,
        pitch_frames=(),
    )
    timings = [
        # "responsabilidad" (6 syllables in 0.3s -> 20 syl/s -> rushed)
        WordTiming(word="responsabilidad", start=0.2, end=0.5, confidence=0.99),
        # "casa" (2 syllables in 1.4s -> 1.43 syl/s -> stretched)
        WordTiming(word="casa", start=0.6, end=2.0, confidence=0.99),
        WordTiming(word="fin.", start=2.1, end=2.4, confidence=0.99),
    ]
    anomalies, verdict = detect_all_anomalies(
        audio=audio,
        timings=timings,
        reference_text="responsabilidad casa fin.",
        transcript="responsabilidad casa fin.",
        pauses=[],
        settings=settings,
        wer=0.0,
        overall_score=85.0,
    )
    types = [a.type for a in anomalies]
    assert "rushed_word" in types
    assert "stretched_word" in types
    assert verdict in ("probable_problema_tts", "requiere_revision")


def test_detect_low_confidence_word() -> None:
    settings = Settings()
    audio = AudioAnalysis(
        duration_seconds=2.0,
        rms_dbfs=-18.0,
        clipping_ratio=0.0,
        leading_silence_seconds=0.1,
        trailing_silence_seconds=0.1,
        mean_energy_db=-18.0,
        voiced_seconds=1.5,
        f0_median_hz=130.0,
        f0_range_semitones=5.0,
        f0_std_semitones=1.0,
        pitch_frames=(),
    )
    timings = [
        WordTiming(word="palabra", start=0.2, end=0.6, confidence=0.32),
    ]
    anomalies, verdict = detect_all_anomalies(
        audio=audio,
        timings=timings,
        reference_text="palabra",
        transcript="palabra",
        pauses=[],
        settings=settings,
        wer=0.0,
        overall_score=85.0,
    )
    assert any(a.type == "low_confidence" for a in anomalies)
    assert verdict == "probable_problema_tts"


def test_deterministic_report_formatting() -> None:
    settings = Settings()
    audio = AudioAnalysis(
        duration_seconds=2.0,
        rms_dbfs=-18.0,
        clipping_ratio=0.0,
        leading_silence_seconds=0.1,
        trailing_silence_seconds=0.1,
        mean_energy_db=-18.0,
        voiced_seconds=1.5,
        f0_median_hz=130.0,
        f0_range_semitones=5.0,
        f0_std_semitones=1.0,
        pitch_frames=(),
    )
    timings = [
        WordTiming(word="responsabilidad", start=0.2, end=0.5, confidence=0.99),
    ]
    pauses = [
        InterwordPause(
            after_word="Hola",
            before_word="amigo",
            start_seconds=0.5,
            end_seconds=1.1,
            duration_seconds=0.6,
            category="brief",
            expected_at_boundary=False,
        )
    ]
    anomalies, verdict = detect_all_anomalies(
        audio=audio,
        timings=timings,
        reference_text="responsabilidad",
        transcript="responsabilidad",
        pauses=pauses,
        settings=settings,
        wer=0.0,
        overall_score=80.0,
    )
    response = EvaluationResponse(
        language="es-ES",
        transcript="responsabilidad",
        expected_phonemes="r e s p o n s a b i l i d a d",
        word_timings=timings,
        audio=AudioMetrics(
            duration_seconds=2.0, rms_dbfs=-18.0, clipping_ratio=0.0,
            leading_silence_seconds=0.1, trailing_silence_seconds=0.1,
        ),
        prosody=ProsodyMetrics(
            voiced_seconds=1.5, f0_median_hz=130.0, f0_range_semitones=5.0,
            f0_std_semitones=1.0, mean_energy_db=-18.0,
        ),
        fluency=FluencyMetrics(
            words_per_minute=120.0, articulation_words_per_minute=140.0,
            pause_count=1, long_pause_count=0, total_pause_seconds=0.6,
            misplaced_pause_count=1, pauses=pauses,
        ),
        fidelity=TextFidelityMetrics(
            expected_words=1, recognized_words=1, word_error_rate=0.0,
            substitutions=0, deletions=0, insertions=0,
        ),
        scores={},
        overall_score=80.0,
        verdict=verdict,
        anomalies=anomalies,
        squim=SquimMetrics(predicted_mos=4.3, predicted_stoi=0.98, predicted_pesq=3.7),
        report="",
        warnings=[],
    )
    report = deterministic_report(response)
    assert "Veredicto:" in report
    assert "Anomalías detectadas:" in report
    assert "responsabilidad" in report
    assert "Calidad perceptual neuronal estimada: MOS 4.3" in report


def test_squim_metrics_disabled_and_enabled() -> None:
    from pathlib import Path

    from tts_eval.squim import compute_squim_metrics

    audio_path = Path("audios/dump_es-ES-SMTg01_00000.wav")
    assert compute_squim_metrics(audio_path, Settings(squim_enabled=False)) is None

    if audio_path.exists():
        metrics = compute_squim_metrics(audio_path, Settings(squim_enabled=True))
        assert metrics is not None
        assert 1.0 <= metrics.predicted_mos <= 5.0
        assert 0.0 <= metrics.predicted_stoi <= 1.0

