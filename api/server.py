"""Servidor FastAPI de inferencia para el módulo de orientación de fibras (HU5).

Uso:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
    MODEL_PATH=models/ppo_v3 uvicorn api.server:app --port 8000

Endpoints:
    GET  /        → info del servidor
    GET  /health  → liveness/readiness
    POST /infer   → estimación de ángulo a partir de imagen
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_DEFAULT_MODEL: str = os.getenv("MODEL_PATH", "models/ppo_v1")
_DEFAULT_METHOD: Literal["rl", "ellipse"] = os.getenv("INFER_METHOD", "rl")  # type: ignore[assignment]

_fiber_model = None  # FiberOrientationModel, cargado en lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _fiber_model
    from pretrained_model import FiberOrientationModel
    _fiber_model = FiberOrientationModel(model_path=_DEFAULT_MODEL, method=_DEFAULT_METHOD)
    logger.info("Modelo cargado: %s", _fiber_model)
    yield
    _fiber_model = None
    logger.info("Servidor detenido.")


app = FastAPI(
    title="Orientador de Fibras Musculares — API de Inferencia",
    description="Estima el ángulo de orientación dominante (0°–180°) de fibras musculares.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    method: str
    model_path: str


class InferResponse(BaseModel):
    angle_deg: float
    method_used: str
    processing_time_ms: float
    visualization_b64: Optional[str] = None


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Orientador de Fibras — API de Inferencia",
        "version": "1.0.0",
        "model_path": _DEFAULT_MODEL,
        "method": _fiber_model.active_method if _fiber_model else "unknown",
        "endpoints": {"GET /health": "Estado de salud", "POST /infer": "Estimar ángulo"},
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    ready = _fiber_model is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        method=_fiber_model.active_method if _fiber_model else "unknown",
        model_path=_DEFAULT_MODEL,
    )


@app.post("/infer", response_model=InferResponse, tags=["Inferencia"])
async def infer(
    file: UploadFile = File(..., description="Imagen PNG o JPG"),
    method: Optional[Literal["rl", "ellipse"]] = Form(default=None),
    viz: bool = Form(default=False, description="Devolver visualización en base64"),
):
    """Estima el ángulo de orientación dominante de fibras en la imagen recibida.

    La imagen se convierte a escala de grises y se redimensiona a 128×128 px.

    **Ejemplo:**
    ```bash
    curl -X POST http://localhost:8000/infer -F "file=@fibra.png" -F "viz=true"
    ```
    """
    if _fiber_model is None:
        raise HTTPException(status_code=503, detail="Servidor no listo.")

    raw = await file.read()
    img = _decode_image(raw, file.filename or "imagen")

    # Si el cliente pide un método distinto al configurado, crear instancia temporal
    effective_method = method or _fiber_model.active_method
    if method and method != _fiber_model.active_method:
        from pretrained_model import FiberOrientationModel
        model_to_use = FiberOrientationModel(model_path=_DEFAULT_MODEL, method=method)
    else:
        model_to_use = _fiber_model

    t0 = time.perf_counter()
    angle, viz_b64 = model_to_use.predict(img, return_visualization=viz)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info("Inferencia: método=%s  ángulo=%.2f°  tiempo=%.1f ms", effective_method, angle, elapsed_ms)

    return InferResponse(
        angle_deg=round(angle, 4),
        method_used=effective_method,
        processing_time_ms=round(elapsed_ms, 2),
        visualization_b64=viz_b64 if viz else None,
    )


def _decode_image(raw_bytes: bytes, filename: str) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f"No se pudo decodificar '{filename}'.")
    if img.shape != (128, 128):
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    return img
