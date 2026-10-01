from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TTS_EVAL_", extra="ignore")

    device: str = "cuda"
    compute_type: str = "float16"
    whisper_model: str = "turbo"
    ffmpeg_bin: str = "ffmpeg"
    work_dir: Path = Path("artifacts")
    min_pause_seconds: float = 0.15
    brief_pause_seconds: float = 0.25
    long_pause_seconds: float = 0.70
    misplaced_pause_seconds: float = 0.45

    mfa_enabled: bool = False
    mfa_command: str = "mfa"
    mfa_ptpt_acoustic_model: str | None = None
    mfa_ptpt_dictionary: str | None = None
    mfa_eses_acoustic_model: str | None = None
    mfa_eses_dictionary: str | None = None

    ollama_enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"

    # Detección de anomalías perceptuales
    low_confidence_threshold: float = 0.60
    pitch_jump_threshold_semitones: float = 6.0
    rushed_syllable_rate: float = 12.0
    stretched_syllable_rate: float = 1.8

    # Evaluación objetiva TorchAudio SQUIM (MOS, PESQ, STOI)
    squim_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
