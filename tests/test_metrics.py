from tts_eval.audio import AudioAnalysis
from tts_eval.metrics import derive_metrics, overall_score, score_all
from tts_eval.schemas import WordTiming


def test_scores_are_bounded() -> None:
    audio = AudioAnalysis(3.0, -18.0, 0.0, 0.0, 0.0, -18.0, 2.4, 155.0, 5.0, 1.2)
    words = [WordTiming(word="Bom", start=0, end=0.4), WordTiming(word="dia", start=0.45, end=0.8)]
    metrics = derive_metrics("Bom dia.", "Bom dia.", words, audio)
    scores = score_all(audio, metrics, "pt-PT")
    assert metrics.fidelity.word_error_rate == 0
    assert metrics.fluency.pause_count == 0
    assert all(0 <= value.score <= 100 for value in scores.values())
    assert 0 <= overall_score(scores) <= 100


def test_reports_each_interword_pause() -> None:
    audio = AudioAnalysis(2.0, -18.0, 0.0, 0.0, 0.0, -18.0, 1.5, 155.0, 5.0, 1.2)
    words = [WordTiming(word="Hola,", start=0.0, end=0.4), WordTiming(word="como", start=0.8, end=1.0)]
    metrics = derive_metrics("Hola, como estas?", "Hola, como estas?", words, audio)

    assert metrics.fluency.pause_count == 1
    assert metrics.fluency.total_pause_seconds == 0.4
    assert metrics.fluency.pauses[0].after_word == "Hola,"
    assert metrics.fluency.pauses[0].before_word == "como"
    assert metrics.fluency.pauses[0].category == "brief"


def test_reports_a_200ms_sentence_pause_as_micro() -> None:
    audio = AudioAnalysis(2.0, -18.0, 0.0, 0.0, 0.0, -18.0, 1.5, 155.0, 5.0, 1.2)
    words = [WordTiming(word="rayes.", start=0.38, end=0.84), WordTiming(word="Ya", start=1.04, end=1.48)]
    metrics = derive_metrics("No te rayes. Ya veras", "No te rayes. Ya veras", words, audio)

    assert metrics.fluency.pause_count == 1
    assert metrics.fluency.total_pause_seconds == 0.2
    assert metrics.fluency.pauses[0].category == "micro"
    assert metrics.fluency.pauses[0].expected_at_boundary
