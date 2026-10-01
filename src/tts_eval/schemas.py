from pydantic import BaseModel, Field

from .languages import EvaluationLanguage


class WordTiming(BaseModel):
    word: str
    start: float
    end: float
    confidence: float | None = None


class MetricScore(BaseModel):
    score: float = Field(ge=0, le=100)
    rationale: list[str]


class AudioMetrics(BaseModel):
    duration_seconds: float
    rms_dbfs: float
    clipping_ratio: float
    leading_silence_seconds: float
    trailing_silence_seconds: float


class ProsodyMetrics(BaseModel):
    voiced_seconds: float
    f0_median_hz: float | None
    f0_range_semitones: float | None
    f0_std_semitones: float | None
    mean_energy_db: float


class InterwordPause(BaseModel):
    after_word: str
    before_word: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    category: str
    expected_at_boundary: bool


class FluencyMetrics(BaseModel):
    words_per_minute: float | None
    articulation_words_per_minute: float | None
    pause_count: int
    long_pause_count: int
    total_pause_seconds: float
    misplaced_pause_count: int
    pauses: list[InterwordPause]


class TextFidelityMetrics(BaseModel):
    expected_words: int
    recognized_words: int
    word_error_rate: float
    substitutions: int
    deletions: int
    insertions: int


class AudioAnomaly(BaseModel):
    timestamp_start: float
    timestamp_end: float
    word: str | None = None
    type: str  # "pitch_jump", "rushed_word", "stretched_word", "low_confidence", "word_substitution", "word_omission", "word_hallucination", "unexpected_pause", "audio_clipping", etc.
    severity: str  # "alta", "media", "baja", "informativa"
    evidence: str
    detail: str


class SquimMetrics(BaseModel):
    predicted_mos: float | None = None
    predicted_stoi: float | None = None
    predicted_pesq: float | None = None
    predicted_si_sdr: float | None = None


class EvaluationResponse(BaseModel):
    language: EvaluationLanguage
    transcript: str
    expected_phonemes: str | None
    word_timings: list[WordTiming]
    audio: AudioMetrics
    prosody: ProsodyMetrics
    fluency: FluencyMetrics
    fidelity: TextFidelityMetrics
    scores: dict[str, MetricScore]
    overall_score: float = Field(ge=0, le=100)
    verdict: str = "sin_incidencias_claras"
    anomalies: list[AudioAnomaly] = []
    squim: SquimMetrics | None = None
    report: str
    warnings: list[str]
