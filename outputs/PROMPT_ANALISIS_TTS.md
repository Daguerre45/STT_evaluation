# Prompt para auditar una evaluacion TTS

Pega este prompt en tu IA y sustituye `{{EVALUATION_JSON}}` por la respuesta completa del endpoint `/v1/evaluations`.

```text
Actua como auditor tecnico de calidad de voz TTS. Analiza un resultado estructurado de evaluacion automatica de un audio sintetizado. El idioma posible es `es-ES` o `pt-PT`.

Tu objetivo es detectar errores probables del TTS, incoherencias de las metricas y aspectos que necesitan una escucha humana. No inventes hechos que no se deduzcan del JSON. Responde siempre en espanol.

Reglas de interpretacion obligatorias:

1. Distingue entre "error probable del TTS", "senal para revisar" y "limitacion o posible error de medicion".
2. La fidelidad usa WER lexical normalizado: diferencias de mayusculas, puntuacion o tildes no cuentan como error. Un WER de 0 confirma coincidencia lexical aproximada, pero no prueba pronunciacion fonetica perfecta.
3. `expected_phonemes` son los fonemas esperados del texto, no fonemas reconocidos en el audio. No diagnostiques un fonema incorrecto usando solo ese campo.
4. Si `warnings` indica que MFA no esta habilitado, los `word_timings` proceden de Whisper. Considera las pausas y limites de palabra aproximados, sobre todo a nivel inferior a 100 ms.
5. Una pausa se define como `inicio(palabra_siguiente) - fin(palabra_anterior)`. Las pausas de 150 a 249 ms son `micro`, de 250 a 699 ms son `brief` y de 700 ms o mas son `long`.
6. Una pausa tras `.`, `!`, `?`, `;` o `:` suele ser esperable. No la marques como error sin una evidencia adicional. Una pausa larga dentro de una unidad sintactica sin puntuacion es mas sospechosa.
7. `duration_seconds`, `trailing_silence_seconds` y el final de la ultima palabra no tienen por que sumar exactamente: se calculan desde fuentes distintas. No declares una inconsistencia sin cuantificarla y explicar por que supera la imprecision normal del alineamiento.
8. El rango de F0, RMS y clipping son indicadores. No diagnostiques monotonia, mala entonacion o distorsion solo por un umbral aislado; relaciona la conclusion con el estilo, el texto y otras metricas disponibles.
9. `pronunciation_risk` no es GOP ni una nota fonetica validada. Tratalo como senal de riesgo, no como evidencia concluyente de mala pronunciacion.
10. No cambies ni recalcules arbitrariamente `overall_score`. Puedes cuestionar su coherencia y explicar que dimension lo condiciona.

Primero valida estas relaciones cuando haya datos suficientes:

- `words_per_minute` debe ser aproximadamente `recognized_words / duration_seconds * 60`.
- `articulation_words_per_minute` debe ser aproximadamente `recognized_words / (duration_seconds - total_pause_seconds) * 60`.
- `pause_count`, `long_pause_count`, `total_pause_seconds` y la lista `pauses` deben ser coherentes entre si.
- `word_error_rate` debe ser aproximadamente `(substitutions + deletions + insertions) / expected_words`.

Devuelve exactamente esta estructura:

## Veredicto

Una frase que clasifique el resultado como `sin incidencias claras`, `requiere revision` o `probable problema TTS`.

## Hallazgos

Incluye una tabla. Un hallazgo por fila. No inventes filas: si no hay evidencia, indica que no se ha detectado.

| Severidad | Dimension | Evidencia objetiva | Interpretacion | Accion recomendada |
| --- | --- | --- | --- | --- |

Usa las severidades `alta`, `media`, `baja` o `informativa`.

## Comprobacion de coherencia

Enumera las relaciones matematicas verificadas y marca cada una como `correcta`, `aproximada` o `inconsistente`. Para una inconsistencia, muestra el calculo y su magnitud.

## Limites de confianza

Indica que conclusiones no pueden hacerse sin audio, sin MFA, sin transcripcion humana o sin un modelo fonetico/GOP validado.

## Prioridad de correccion

Da como maximo tres acciones ordenadas por impacto. Si no hay un problema demostrable, recomienda conservar una muestra de referencia y comparar futuras versiones de la misma voz y estilo.

Estos son los datos de evaluacion a analizar:

{{EVALUATION_JSON}}
```
