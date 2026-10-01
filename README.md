# Evaluador local de TTS en portugues europeo y espanol de Espana

Servicio local para evaluar un archivo de audio sintetizado junto con el texto o SSML que debia locutarse. Soporta `pt-PT` y `es-ES`, y produce un JSON explicable con fidelidad textual, ritmo, prosodia, calidad de audio y recomendaciones.

No envia audio ni metricas a servicios externos. La transcripcion usa `faster-whisper` en la GPU; el informe puede ser determinista o redactarse con un Qwen local mediante Ollama. El proceso descarga inicialmente los pesos elegidos desde su repositorio de modelos, pero tras ello puede operar desconectado.

## Que mide el servicio

El servicio recibe siempre dos fuentes de verdad: el audio sintetizado y el texto o SSML que el TTS debia reproducir. No asigna una nota opaca: devuelve las medidas intermedias en JSON para que cada resultado pueda verificarse.

Las puntuaciones no sustituyen a una prueba de escucha humana. Son una primera heuristica para detectar regresiones entre versiones de una voz, textos o configuraciones del sintetizador. Antes de usarlas como criterio de aceptacion se deben calibrar con muestras evaluadas por personas.

## Medicion de pausas entre palabras

Esta es la medida central para analizar ritmo. Tras transcribir el audio, el pipeline obtiene la marca temporal de cada palabra. Con MFA activado usa sus limites de palabra; de lo contrario usa las marcas por palabra de Whisper.

Para cada par consecutivo de palabras se calcula:

```text
hueco_i = max(0, inicio(palabra_i+1) - fin(palabra_i))
```

El `max(0, ...)` evita producir pausas negativas cuando dos marcas se solapan por una imprecision del alineador. Se mide en segundos y se conserva con tres decimales.

Se aplican estos umbrales iniciales:

| Hueco | Clasificacion | Uso actual |
| --- | --- | --- |
| Menor de 150 ms | Microseparacion no informada | No se devuelve como pausa ni se descuenta del tiempo de articulacion. |
| Entre 150 y 249 ms | `micro` | Se devuelve como pausa breve entre palabras y se descuenta del tiempo de articulacion. |
| Entre 250 y 699 ms | `brief` | Se devuelve como pausa entre palabras. |
| 700 ms o mas | `long` | Se devuelve como pausa larga y reduce la puntuacion de ritmo. |

Cada elemento de `fluency.pauses` contiene `after_word`, `before_word`, `start_seconds`, `end_seconds`, `duration_seconds`, `category` y `expected_at_boundary`. Por ejemplo:

```json
{
  "after_word": "Hola,",
  "before_word": "como",
  "start_seconds": 1.02,
  "end_seconds": 1.4,
  "duration_seconds": 0.38,
  "category": "brief",
  "expected_at_boundary": false
}
```

`expected_at_boundary` se calcula a partir de la puntuacion del texto de referencia. Una pausa despues de `.`, `!`, `?`, `;` o `:` se considera esperable. Una pausa de 450 ms o mas fuera de esos limites aumenta `misplaced_pause_count`. No significa necesariamente que sea un error: una coma, un enfasis expresivo o un SSML con pausa intencionada pueden justificarla. Para evaluar SSML con precision, la siguiente mejora recomendada es interpretar de forma explicita las etiquetas `<break>` y convertirlas en limites permitidos.

`pause_count` es el numero de pausas de 150 ms o mas. `long_pause_count` cuenta las de 700 ms o mas. `total_pause_seconds` suma exclusivamente las pausas informadas, no todos los huecos diminutos entre palabras. Esto permite comparar voces de distinta velocidad sin confundir coarticulacion normal con silencio. Los cuatro umbrales se pueden ajustar en `.env` mediante `TTS_EVAL_MIN_PAUSE_SECONDS`, `TTS_EVAL_BRIEF_PAUSE_SECONDS`, `TTS_EVAL_LONG_PAUSE_SECONDS` y `TTS_EVAL_MISPLACED_PAUSE_SECONDS`.

## Ritmo y velocidad

`words_per_minute` mide la velocidad global:

```text
WPM = numero de palabras transcritas / duracion total del audio * 60
```

Incluye silencios iniciales, finales y pausas. Es la medida adecuada para comprobar la duracion de una locucion completa.

`articulation_words_per_minute` mide la velocidad mientras la voz esta articulando:

```text
WPM de articulacion = numero de palabras / (duracion total - total_pause_seconds) * 60
```

No descuenta microseparaciones, porque forman parte natural de la transicion entre palabras. Una WPM global baja con una WPM de articulacion normal suele indicar demasiadas pausas; si ambas son altas, el TTS probablemente habla demasiado deprisa. Los rangos aceptables dependen de idioma, estilo, publico y tipo de texto, asi que el servicio los expone sin imponer una velocidad universal.

La puntuacion `rhythm` comienza en 100. Resta 8 puntos por cada pausa larga y 7 puntos por cada pausa de 450 ms o mas que no siga puntuacion de cierre. Es una regla inicial para detectar regresiones, no una definicion de naturalidad.

## Fidelidad del texto

El sistema normaliza texto esperado y transcripcion antes de compararlos: pasa a minusculas, elimina puntuacion, quita etiquetas SSML y no considera un error que `como` y `cómo` difieran. Esto evita falsos errores como comparar `Hola, como estas?` con `Hola, ¿cómo estás?` cuando Whisper conserva las tildes correctamente.

Despues calcula distancia de Levenshtein por palabra:

```text
WER = (sustituciones + omisiones + inserciones) / palabras esperadas
```

`word_error_rate`, `substitutions`, `deletions` e `insertions` se devuelven en `fidelity`. La puntuacion `fidelity` es `100 * (1 - WER)`, limitada entre 0 y 100. Esta medida valida que la salida sea inteligible y fiel al texto, pero no detecta por si sola un fonema mal articulado si el ASR adivina la palabra correcta.

## Prosodia e intonacion

El analisis usa Praat/parselmouth sobre audio mono a 16 kHz. Se extrae F0 cada 10 ms, con un rango de busqueda inicial de 60 a 500 Hz. Los fotogramas sin voz se descartan.

- `voiced_seconds`: segundos con F0 valido. Si es muy bajo respecto a `duration_seconds`, no se debe confiar en la metrica tonal.
- `f0_median_hz`: mediana de F0 en hertz. Describe la altura tonal habitual de ese audio, no una calidad ni una identidad de hablante.
- `f0_range_semitones`: diferencia entre los percentiles 90 y 10 de F0, expresada en semitonos respecto a la mediana. El uso de percentiles evita que un salto aislado domine la medida.
- `f0_std_semitones`: desviacion estandar de F0 en semitonos. Captura variacion tonal global, pero es sensible a cambios de locutor o a errores de extraccion.
- `mean_energy_db`: RMS global en dBFS; ayuda a contextualizar intensidad y calidad tecnica.

La puntuacion `prosody` parte de 78. Un rango inferior a 3.5 semitonos resta 12 puntos y uno inferior a 2.2 resta 28; por encima de 16 resta 15 como posible salto tonal anomalo. Tambien resta 4 puntos por pausa larga mal situada. Estos limites son deliberadamente conservadores: una voz calmada, un anuncio, una pregunta o una locucion dramatica necesitan perfiles distintos. La calibracion recomendable compara cada estilo TTS contra una distribucion de referencia del mismo estilo.

## Calidad tecnica del audio

Antes del analisis, FFmpeg convierte la entrada a WAV mono de 16 kHz. Las metricas se calculan sobre ese audio normalizado.

- `duration_seconds`: numero de muestras dividido por frecuencia de muestreo.
- `rms_dbfs`: nivel medio RMS respecto al maximo digital. Valores mas negativos implican menor nivel.
- `clipping_ratio`: fraccion de muestras con amplitud absoluta igual o superior a 0.999. Cualquier valor sostenido indica recorte digital.
- `leading_silence_seconds` y `trailing_silence_seconds`: silencio antes y despues de voz, detectado con un umbral de -45 dBFS.

La puntuacion `audio_quality` empieza en 100. Reduce hasta 55 puntos si hay clipping, 18 puntos si el RMS es menor que -32 dBFS y 8 puntos si hay mas de un segundo de silencio inicial o final. No mide ruido, reverberacion ni distorsion neural de forma completa; esos son candidatos para una futura metrica SQUIM o MOS calibrada.

Las marcas de `word_timings` pertenecen a Whisper o MFA, mientras que `duration_seconds` y el silencio final proceden de la forma de onda. No deben sumarse como si fueran fronteras identicas: puede haber una diferencia de decenas de milisegundos entre `fin(ultima_palabra) + trailing_silence_seconds` y `duration_seconds`. Por ejemplo, Whisper puede terminar la palabra antes de que la energia de una consonante final caiga por debajo de -45 dBFS. Con MFA activado las fronteras de palabra suelen ser mas precisas, pero siguen siendo estimaciones.

## Pronunciacion

`expected_phonemes` contiene la representacion fonetica generada por eSpeak NG para el idioma solicitado: `pt-PT` o `es-ES`. Es util para inspeccion y para preparar alineamiento fonetico, pero todavia no alimenta una nota fonetica definitiva.

La puntuacion `pronunciation_risk` combina 80% de fidelidad lexical y hasta 20 puntos por proporcion de audio con voz. Sirve como semaforo de riesgo: una transcripcion distinta del texto o una proporción de voz muy baja merecen revisar el audio. No es una medida GOP ni diagnostica un fonema concreto.

Para puntuacion de pronunciacion real se necesita un modelo de posteriors foneticos o GOP entrenado y validado por idioma, junto con un conjunto de evaluaciones humanas. MFA mejora los limites temporales de palabras y fonemas, pero alinear un fonema esperado no demuestra por si mismo que se haya pronunciado correctamente.

## Puntuacion global

La nota `overall_score` es una combinacion ponderada:

| Dimension | Peso |
| --- | ---: |
| Fidelidad | 30% |
| Ritmo | 20% |
| Prosodia | 25% |
| Calidad de audio | 15% |
| Riesgo de pronunciacion | 10% |

La ponderacion favorece que el TTS diga el texto correcto y mantenga una prosodia estable. Ajustala tras recoger ejemplos de tus voces y las valoraciones humanas correspondientes.

## Requisitos en Windows

1. Instala Python 3.10 a 3.12, Git y [FFmpeg](https://ffmpeg.org/download.html), y deja `ffmpeg` disponible en `PATH`.
2. Instala el driver NVIDIA reciente y comprueba `nvidia-smi`.
3. Crea y activa un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

4. Instala primero PyTorch con CUDA siguiendo el selector oficial de [PyTorch](https://pytorch.org/get-started/locally/). Despues:

```powershell
pip install -r requirements-gpu.txt
Copy-Item .env.example .env
```

5. Instala `eSpeak NG` y deja `espeak-ng` en `PATH` para que `phonemizer` genere fonemas portugueses. Es opcional para el MVP, pero necesario para el detalle fonetico.

## Ejecutar

```powershell
uvicorn tts_eval.api:app --app-dir src --reload --port 8000
```

Abre `http://127.0.0.1:8000/docs`. La primera solicitud descarga el modelo Whisper configurado.

### Endpoints disponibles

- `GET /health`: Comprobación rápida de estado del servicio, dispositivo (`cuda`/`cpu`), modelo de Whisper configurado e idiomas admitidos (`pt-PT`, `es-ES`).
- `POST /v1/evaluations`: Evaluación completa de un audio TTS frente a un texto/SSML de referencia.
  - Parámetros multipart/form-data:
    - `audio`: Archivo de audio (WAV, MP3, M4A, FLAC, OGG, etc.; decodificado automáticamente con FFmpeg a WAV mono 16 kHz).
    - `reference_text`: Texto o SSML esperado.
    - `language`: `pt-PT` (por defecto) o `es-ES`.
    - `style`: Estilo del audio (por defecto `neutral`).

Ejemplo de comprobación de salud:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Ejemplo de evaluación con `curl.exe`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/evaluations" `
  -F "audio=@C:\audio\voz.wav" `
  -F "reference_text=Bom dia. Como esta?" `
  -F "language=pt-PT" `
  -F "style=neutral"
```

Para español de España, envía `-F "language=es-ES"`. La especificación interactiva OpenAPI muestra ambos valores en `http://127.0.0.1:8000/docs`.

### Estructura de la respuesta JSON

La respuesta devuelta por `/v1/evaluations` contiene:

- `verdict`: Veredicto global de calidad auditada al estilo humano (`sin_incidencias_claras`, `requiere_revision` o `probable_problema_tts`).
- `anomalies`: Lista cronológica de anomalías locales detectadas (con `timestamp_start`, `timestamp_end`, `word`, `type`, `severity`, `evidence` y `detail`).
  - Tipos detectados:
    - `pitch_jump`: Saltos tonales o quiebres de entonación abruptos ("gallos" o distorsión del vocoder) evaluados sobre la derivada cuadro a cuadro de F0.
    - `rushed_word`: Palabras pronunciadas con excesiva rapidez (atropelladas), evaluadas mediante la tasa de sílabas por segundo.
    - `stretched_word`: Palabras con duración excesivamente prolongada (arrastradas o ritmo robótico).
    - `low_confidence`: Palabras con baja certidumbre de reconocimiento acústico en Whisper (posible balbuceo, fonema anómalo o sonido extraño).
    - `word_substitution` / `word_omission` / `word_hallucination`: Discrepancias textuales exactas mapeadas en la línea temporal.
    - `unexpected_pause`: Pausas injustificadas dentro de una cláusula sin puntuación.
    - `audio_clipping`: Saturación y recorte digital en la forma de onda.
- `squim`: Evaluación perceptual neuronal objetiva y subjetiva con TorchAudio SQUIM (si está habilitado con `TTS_EVAL_SQUIM_ENABLED=true`):
  - `predicted_mos`: Mean Opinion Score subjetivo estimado (escala 1.0 a 5.0).
  - `predicted_pesq`: Calidad perceptual del habla (1.0 a 4.5).
  - `predicted_stoi`: Inteligibilidad objetiva (0.0 a 1.0).
  - `predicted_si_sdr`: Ratio de distorsión señal-interferencia en dB.
- `language`: Idioma evaluado (`pt-PT` o `es-ES`).
- `transcript`: Transcripción obtenida por Whisper.
- `expected_phonemes`: Fonemas generados por eSpeak NG para el texto de referencia (o `null` si no está instalado).
- `word_timings`: Lista de palabras con marcas temporales `start`, `end` y opcionalmente `confidence` (de MFA o Whisper).
- `audio`: Métricas técnicas globales (`duration_seconds`, `rms_dbfs`, `clipping_ratio`, `leading_silence_seconds`, `trailing_silence_seconds`).
- `prosody`: Métricas de tono y energía (`voiced_seconds`, `f0_median_hz`, `f0_range_semitones`, `f0_std_semitones`, `mean_energy_db`).
- `fluency`: Métricas de ritmo (`words_per_minute`, `articulation_words_per_minute`, `pause_count`, `long_pause_count`, `total_pause_seconds`, `misplaced_pause_count`, lista detallada de `pauses`).
- `fidelity`: Métricas de texto (`expected_words`, `recognized_words`, `word_error_rate`, `substitutions`, `deletions`, `insertions`).
- `scores`: Puntuaciones (0-100) y justificaciones (`rationale`) para `fidelity`, `rhythm`, `prosody`, `audio_quality` y `pronunciation_risk`.
- `overall_score`: Puntuación global ponderada (0-100), ajustada automáticamente si se detectan anomalías graves.
- `report`: Diagnóstico textual explicativo con el veredicto y el desglose de incidencias detectadas por segundo (heurístico determinista o redactado por Ollama/Qwen si está habilitado).
- `warnings`: Lista de advertencias no bloqueantes.

## Configuración (.env)

El servicio permite parametrizar su comportamiento mediante variables de entorno en [.env](file:///c:/Users/mqa-s/Documents/Codex/2026-09-29/quiero-que-seas-un-experto-de/.env):

| Variable | Por defecto | Descripción |
| --- | --- | --- |
| `TTS_EVAL_DEVICE` | `cuda` | Dispositivo de inferencia para `faster-whisper` (`cuda` o `cpu`). |
| `TTS_EVAL_COMPUTE_TYPE` | `float16` | Precisión de cómputo (`float16`, `int8_float16`, `int8`, etc.). |
| `TTS_EVAL_WHISPER_MODEL` | `turbo` | Modelo de faster-whisper a descargar/usar (`turbo`, `large-v3`, `medium`, etc.). |
| `TTS_EVAL_FFMPEG_BIN` | `ffmpeg` | Binario o ruta absoluta de `ffmpeg`. |
| `TTS_EVAL_MIN_PAUSE_SECONDS` | `0.15` | Umbral mínimo para computar una pausa interpalabra (150 ms). |
| `TTS_EVAL_BRIEF_PAUSE_SECONDS` | `0.25` | Umbral para clasificar una pausa como `brief` (250 ms). |
| `TTS_EVAL_LONG_PAUSE_SECONDS` | `0.70` | Umbral para clasificar una pausa como `long` (700 ms). |
| `TTS_EVAL_MISPLACED_PAUSE_SECONDS` | `0.45` | Umbral para considerar anómala una pausa fuera de límite sintáctico (450 ms). |
| `TTS_EVAL_LOW_CONFIDENCE_THRESHOLD` | `0.60` | Umbral de certidumbre de Whisper para alertar sobre posibles palabras raras o defectuosas. |
| `TTS_EVAL_PITCH_JUMP_THRESHOLD_SEMITONES` | `6.0` | Umbral de salto tonal en semitonos para detectar "gallos" o quiebres de entonación abruptos. |
| `TTS_EVAL_RUSHED_SYLLABLE_RATE` | `12.0` | Tasa de sílabas por segundo a partir de la cual una palabra se considera atropellada. |
| `TTS_EVAL_STRETCHED_SYLLABLE_RATE` | `1.8` | Tasa de sílabas por segundo por debajo de la cual una palabra se considera arrastrada. |
| `TTS_EVAL_SQUIM_ENABLED` | `false` | Activa la evaluación neuronal no intrusiva TorchAudio SQUIM (MOS subjetivo, PESQ, STOI). |
| `TTS_EVAL_MFA_ENABLED` | `false` | Activa o desactiva la alineación fonética/temporal con MFA. |
| `TTS_EVAL_MFA_COMMAND` | `mfa` | Binario o comando de ejecución de Montreal Forced Aligner. |
| `TTS_EVAL_MFA_PTPT_ACOUSTIC_MODEL` | *(vacío)* | Modelo acústico MFA para `pt-PT`. |
| `TTS_EVAL_MFA_PTPT_DICTIONARY` | *(vacío)* | Diccionario de pronunciación MFA para `pt-PT`. |
| `TTS_EVAL_MFA_ESES_ACOUSTIC_MODEL` | `spanish_mfa` | Modelo acústico MFA para `es-ES`. |
| `TTS_EVAL_MFA_ESES_DICTIONARY` | `spanish_mfa` | Diccionario de pronunciación MFA para `es-ES`. |
| `TTS_EVAL_OLLAMA_ENABLED` | `false` | Activa la redacción del informe final con un LLM local. |
| `TTS_EVAL_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | URL base de la API de Ollama. |
| `TTS_EVAL_OLLAMA_MODEL` | `qwen3:4b` | Modelo en Ollama para redactar el informe. |

## Alineamiento MFA opcional

Para español de España, instala Miniforge y ejecuta `powershell -ExecutionPolicy Bypass -File scripts\setup-mfa-eses.ps1`. El script crea un entorno aislado, descarga `spanish_mfa` y actualiza `.env` con el ejecutable y recursos correctos. Para portugués europeo, configura `TTS_EVAL_MFA_PTPT_*` con recursos que hayas validado. El servicio conserva las marcas de Whisper como respaldo cuando MFA no esté disponible.

## Informe con Qwen local opcional

Instala Ollama, descarga un Qwen que quepa en tu VRAM y activa `TTS_EVAL_OLLAMA_ENABLED=true`. El pipeline libera el modelo de Whisper antes de solicitar la redacción para que ambos modelos no compartan VRAM. Qwen solo redacta el informe a partir de las métricas: nunca decide la nota.

## Pruebas

```powershell
$env:PYTHONPATH = "src"
pytest -q
ruff check src tests
```

## Siguiente fase recomendada

Crear conjuntos independientes de 200-500 frases `pt-PT` y `es-ES`, con valoraciones humanas de naturalidad y errores conocidos. Con ellos se sustituye la heuristica de puntuacion por modelos calibrados y se incorpora un modelo fonetico/GOP validado por idioma.
