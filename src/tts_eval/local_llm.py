import json

import httpx

from .config import Settings
from .schemas import EvaluationResponse


def redact_with_ollama(result: EvaluationResponse, settings: Settings) -> str | None:
    """Optional local copy editor. Scores remain produced by deterministic rules."""
    if not settings.ollama_enabled:
        return None
    lang_label = "portugués europeo" if result.language == "pt-PT" else "español de España"
    evidence = {
        "language": result.language,
        "verdict": result.verdict,
        "overall_score": result.overall_score,
        "scores": {name: item.model_dump() for name, item in result.scores.items()},
        "fidelity": result.fidelity.model_dump(),
        "fluency": result.fluency.model_dump(),
        "prosody": result.prosody.model_dump(),
        "audio": result.audio.model_dump(),
        "anomalies": [item.model_dump() for item in result.anomalies],
        "squim": result.squim.model_dump() if result.squim else None,
    }
    prompt = (
        f"Actúa como auditor técnico de calidad de voz TTS. Redacta en {lang_label} un informe conciso, explicable y constructivo. "
        "Analiza el veredicto, las métricas y menciona específicamente las anomalías detectadas con sus marcas temporales (saltos de tono, palabras atropelladas/estiradas o fallos de articulación). "
        "No inventes hechos no respaldados por los datos ni modifiques puntuaciones. No menciones modelos ni APIs. "
        "Incluye como máximo tres recomendaciones concretas de mejora técnica para el sintetizador. Datos:\n"
        + json.dumps(evidence, ensure_ascii=False)
    )
    response = httpx.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=90,
    )
    response.raise_for_status()
    report = response.json().get("response", "").strip()
    return report or None
