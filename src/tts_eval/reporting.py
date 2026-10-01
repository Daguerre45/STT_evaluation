from .schemas import EvaluationResponse


def deterministic_report(result: EvaluationResponse) -> str:
    parts: list[str] = []

    # 1. Veredicto
    verdict_labels = {
        "sin_incidencias_claras": "Veredicto: Sin incidencias claras.",
        "requiere_revision": "Veredicto: Requiere revisión (se detectaron irregularidades menores o moderadas).",
        "probable_problema_tts": "Veredicto: Probable problema TTS (se detectaron anomalías significativas que degradan la naturalidad o inteligibilidad).",
    }
    parts.append(verdict_labels.get(result.verdict, f"Veredicto: {result.verdict}."))

    # 2. Resumen perceptual SQUIM si está disponible
    if result.squim and result.squim.predicted_mos is not None:
        squim_note = f"Calidad perceptual neuronal estimada: MOS {result.squim.predicted_mos:.1f}/5.0"
        if result.squim.predicted_pesq is not None:
            squim_note += f", PESQ {result.squim.predicted_pesq:.2f}"
        if result.squim.predicted_stoi is not None:
            squim_note += f", STOI {result.squim.predicted_stoi:.2f}"
        parts.append(squim_note + ".")

    # 3. Lista de anomalías detectadas en la línea temporal
    if result.anomalies:
        parts.append("Anomalías detectadas:")
        for anomaly in result.anomalies[:6]:  # Primeras 6 incidencias clave
            time_str = f"[{anomaly.timestamp_start:.2f}s - {anomaly.timestamp_end:.2f}s]"
            word_str = f" en '{anomaly.word}'" if anomaly.word else ""
            parts.append(f"- {time_str} ({anomaly.severity.upper()}) {anomaly.type}{word_str}: {anomaly.evidence}. {anomaly.detail}")
        if len(result.anomalies) > 6:
            parts.append(f"- ... y {len(result.anomalies) - 6} anomalías adicionales detalladas en el campo 'anomalies'.")
    else:
        parts.append("No se detectaron saltos tonales abruptos, atropellos de ritmo ni deformaciones acústicas destacables.")

    # 4. Observaciones globales adicionales si aplican
    if result.fidelity.word_error_rate > 0.05 and not any(a.type.startswith("word_") for a in result.anomalies):
        parts.append("La locución difiere del texto esperado; revisa posibles discrepancias textuales.")
    if result.audio.clipping_ratio > 0.0001 and not any(a.type == "audio_clipping" for a in result.anomalies):
        parts.append("El audio presenta clipping digital; conviene reducir la ganancia de salida del sintetizador.")

    return "\n".join(parts)
