import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf


@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class AudioAnalysis:
    duration_seconds: float
    rms_dbfs: float
    clipping_ratio: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    mean_energy_db: float
    voiced_seconds: float
    f0_median_hz: float | None
    f0_range_semitones: float | None
    f0_std_semitones: float | None
    pitch_frames: tuple[tuple[float, float], ...] = ()


def prepare_for_analysis(source: Path, destination: Path, ffmpeg_bin: str) -> None:
    """Decode every common input format to the mono 16 kHz WAV used downstream."""
    command = [
        ffmpeg_bin, "-nostdin", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr else "error desconocido"
        raise RuntimeError(f"FFmpeg no pudo preparar el audio: {detail}")


def load_mono_audio(path: Path) -> LoadedAudio:
    samples, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    return LoadedAudio(np.asarray(samples, dtype=np.float32), sample_rate)


def _silence_edges(samples: np.ndarray, sample_rate: int, threshold_db: float = -45.0) -> tuple[float, float]:
    threshold = 10 ** (threshold_db / 20)
    active = np.flatnonzero(np.abs(samples) > threshold)
    if not len(active):
        duration = len(samples) / sample_rate
        return duration, duration
    return active[0] / sample_rate, (len(samples) - 1 - active[-1]) / sample_rate


def _pitch_stats(
    path: Path, duration: float
) -> tuple[float, float | None, float | None, float | None, tuple[tuple[float, float], ...]]:
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
    times = pitch.xs()
    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]
    voiced_seconds = len(voiced) * 0.01
    pitch_frames = tuple(
        (round(float(t), 3), round(float(f), 2))
        for t, f in zip(times, values)
        if f > 0
    )
    if len(voiced) < 3:
        return voiced_seconds, None, None, None, pitch_frames
    semitones = 12 * np.log2(voiced / np.median(voiced))
    return (
        min(voiced_seconds, duration),
        float(np.median(voiced)),
        float(np.percentile(semitones, 90) - np.percentile(semitones, 10)),
        float(np.std(semitones)),
        pitch_frames,
    )


def analyze_audio(path: Path) -> AudioAnalysis:
    audio = load_mono_audio(path)
    duration = len(audio.samples) / audio.sample_rate
    rms = float(np.sqrt(np.mean(np.square(audio.samples)))) if len(audio.samples) else 0.0
    rms_dbfs = 20 * np.log10(max(rms, 1e-10))
    leading, trailing = _silence_edges(audio.samples, audio.sample_rate)
    voiced, median, pitch_range, pitch_std, pitch_frames = _pitch_stats(path, duration)
    return AudioAnalysis(
        duration_seconds=round(duration, 3),
        rms_dbfs=round(float(rms_dbfs), 2),
        clipping_ratio=round(float(np.mean(np.abs(audio.samples) >= 0.999)), 6),
        leading_silence_seconds=round(leading, 3),
        trailing_silence_seconds=round(trailing, 3),
        mean_energy_db=round(float(rms_dbfs), 2),
        voiced_seconds=round(voiced, 3),
        f0_median_hz=round(median, 2) if median else None,
        f0_range_semitones=round(pitch_range, 2) if pitch_range is not None else None,
        f0_std_semitones=round(pitch_std, 2) if pitch_std is not None else None,
        pitch_frames=pitch_frames,
    )
