import os
import sys

# Si ejecutamos en Windows, registramos las carpetas bin de nvidia-cublas y nvidia-cudnn
if sys.platform == "win32":
    # Obtiene la ruta global de site-packages o del entorno actual
    for path in sys.path:
        if "site-packages" in path:
            cublas_bin = os.path.join(path, "nvidia", "cublas", "bin")
            cudnn_bin = os.path.join(path, "nvidia", "cudnn", "bin")
            
            if os.path.isdir(cublas_bin):
                os.add_dll_directory(cublas_bin)
                os.environ["PATH"] = cublas_bin + os.path.pathsep + os.environ.get("PATH", "")
            if os.path.isdir(cudnn_bin):
                os.add_dll_directory(cudnn_bin)
                os.environ["PATH"] = cudnn_bin + os.path.pathsep + os.environ.get("PATH", "")

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import get_settings
from .languages import EvaluationLanguage
from .pipeline import EvaluationPipeline
from .schemas import EvaluationResponse

app = FastAPI(title="TTS Evaluator pt-PT and es-ES", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "device": settings.device,
        "whisper_model": settings.whisper_model,
        "languages": "pt-PT, es-ES",
    }


@app.post("/v1/evaluations", response_model=EvaluationResponse)
async def create_evaluation(
    audio: Annotated[UploadFile, File()],
    reference_text: Annotated[str, Form()],
    language: Annotated[EvaluationLanguage, Form()] = "pt-PT",
    style: Annotated[str, Form()] = "neutral",
) -> EvaluationResponse:
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(await audio.read())
        path = Path(temporary.name)
    try:
        return EvaluationPipeline(get_settings()).evaluate(path, reference_text, language)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)
