from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

from .config import Settings
from .schemas import SquimMetrics


@lru_cache(maxsize=1)
def _get_squim_objective_model():
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    return SQUIM_OBJECTIVE.get_model()


@lru_cache(maxsize=1)
def _get_squim_subjective_model():
    from torchaudio.pipelines import SQUIM_SUBJECTIVE
    return SQUIM_SUBJECTIVE.get_model()


def compute_squim_metrics(audio_path: Path, settings: Settings) -> SquimMetrics | None:
    """Computes TorchAudio SQUIM speech quality metrics (MOS, STOI, PESQ, SI-SDR)."""
    if not settings.squim_enabled:
        return None
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        objective_model = _get_squim_objective_model()
        subjective_model = _get_squim_subjective_model()

        with torch.no_grad():
            stoi, pesq, si_sdr = objective_model(waveform)
            mos = subjective_model(waveform, waveform)

        return SquimMetrics(
            predicted_mos=round(float(mos[0]), 2),
            predicted_stoi=round(float(stoi[0]), 3),
            predicted_pesq=round(float(pesq[0]), 2),
            predicted_si_sdr=round(float(si_sdr[0]), 2),
        )
    except (RuntimeError, ValueError, OSError, TypeError):
        return None
