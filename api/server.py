"""Servidor de inferencia FastAPI para el módulo de orientación de fibras (HU5).

Expone un endpoint REST que acepta una imagen PNG/JPG y devuelve el ángulo
estimado de orientación dominante de las fibras musculares.

Uso:
    # Con modelo RL (por defecto busca models/ppo_v1.zip)
    uvicorn api.server:app --host 0.0.0.0 --port 8000

    # Especificando modelo
    MODEL_PATH=models/ppo_v3 uvicorn api.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /           → información del servidor y modelo cargado
    GET  /health     → estado de salud (liveness/readiness)
    POST /infer      → estimación de ángulo a partir de imagen
"""

import base64
import io
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Configuración desde variables de entorno
# ---------------------------------------------------------------------------

# Ruta al modelo PPO (sin extensión .zip); se puede sobreescribir via env var
_DEFAULT_MODEL = os.getenv("MODEL_PATH", "models/ppo_v1")

# Método por defecto: "rl" usa el agente PPO, "ellipse" usa el fallback geométrico
_DEFAULT_METHOD: Literal["rl", "ellipse"] = os.getenv("INFER_METHOD", "rl")  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Estado global del servidor (modelo cargado en memoria una sola vez)
# ---------------------------------------------------------------------------

class _AppState:
    """Contiene el modelo y metadatos cargados al arrancar el servidor."""
    model = None          # PPO model (solo si method = "rl")
    model_path: str = ""
    method: str = _DEFAULT_METHOD
    load_time_s: float = 0.0
    ready: bool = False


_state = _AppState()


# ---------------------------------------------------------------------------
# Ciclo de vida: carga el modelo al iniciar, libera recursos al cerrar
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo PPO una sola vez al arrancar el servidor."""
    _state.method = _DEFAULT_METHOD
    _state.model_path = _DEFAULT_MODEL

    if _state.method == "rl":
        try:
            from stable_baselines3 import PPO
            t0 = time.perf_counter()
            _state.model = PPO.load(_state.model_path)
            _state.load_time_s = time.perf_counter() - t0
            logger.info(
                "Modelo PPO cargado desde '%s' en %.2f s",
                _state.model_path, _state.load_time_s,
            )
        except Exception as exc:
            # Si el modelo no existe, arrancar de todas formas pero marcar como no listo
            logger.error("No se pudo cargar el modelo PPO: %s", exc)
            logger.warning(
                "Servidor iniciado SIN modelo RL. "
                "Solo disponible el método 'ellipse'."
            )
            _state.method = "ellipse"

    _state.ready = True
    yield
    # Limpieza al cerrar
    _state.model = None
    _state.ready = False
    logger.info("Servidor detenido.")


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Orientador de Fibras Musculares — API de Inferencia",
    description=(
        "Estima el ángulo de orientación dominante (0°–180°) de fibras musculares "
        "a partir de una imagen de máscara o histológica."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Permitir llamadas desde cualquier origen (necesario para UIs web locales)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Esquemas de respuesta
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    method: str
    model_path: str


class InferResponse(BaseModel):
    angle_deg: float
    """Ángulo estimado en grados, rango [0°, 180°)."""

    method_used: str
    """Método efectivamente empleado: 'rl' o 'ellipse'."""

    processing_time_ms: float
    """Tiempo de procesamiento en milisegundos."""

    visualization_b64: Optional[str] = None
    """Imagen PNG con el vector de orientación superpuesto, codificada en base64.
    Solo se incluye si el cliente solicita visualización (viz=true)."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    """Devuelve información básica del servidor y del modelo cargado."""
    return {
        "service": "Orientador de Fibras — API de Inferencia",
        "version": "1.0.0",
        "model_path": _state.model_path,
        "method": _state.method,
        "model_loaded": _state.model is not None or _state.method == "ellipse",
        "endpoints": {
            "GET  /health":      "Estado de salud del servidor",
            "POST /infer":       "Estimar ángulo de orientación a partir de imagen",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Liveness/readiness check — útil para monitoreo y contenedores."""
    model_loaded = (_state.model is not None) or (_state.method == "ellipse")
    return HealthResponse(
        status="ok" if (_state.ready and model_loaded) else "degraded",
        model_loaded=model_loaded,
        method=_state.method,
        model_path=_state.model_path,
    )


@app.post("/infer", response_model=InferResponse, tags=["Inferencia"])
async def infer(
    file: UploadFile = File(..., description="Imagen PNG o JPG de la fibra (máscara o histológica)"),
    method: Optional[Literal["rl", "ellipse"]] = Form(
        default=None,
        description="Método de estimación. Si se omite, usa el método por defecto del servidor.",
    ),
    viz: bool = Form(
        default=False,
        description="Si es true, devuelve la imagen con el vector de orientación superpuesto (base64).",
    ),
):
    """Estima el ángulo de orientación dominante de fibras en la imagen recibida.

    - Acepta imágenes en escala de grises o color (se convierte internamente a grises).
    - La imagen se redimensiona automáticamente a 128×128 px.
    - Devuelve el ángulo en grados [0°, 180°) y, opcionalmente, una visualización PNG.

    **Ejemplo con curl:**
    ```bash
    curl -X POST http://localhost:8000/infer \\
         -F "file=@fibra.png" \\
         -F "viz=true"
    ```
    """
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Servidor no listo todavía.")

    # Determinar método efectivo
    effective_method = method or _state.method

    # Leer y decodificar la imagen recibida
    raw_bytes = await file.read()
    img = _decode_image(raw_bytes, file.filename or "imagen")

    t0 = time.perf_counter()

    if effective_method == "ellipse":
        angle = _infer_ellipse(img)
    else:
        # Usar modelo RL; si no está cargado, hacer fallback a elipse
        if _state.model is None:
            logger.warning("Modelo RL no disponible. Usando fallback ellipse.")
            effective_method = "ellipse"
            angle = _infer_ellipse(img)
        else:
            angle = _infer_rl(img, _state.model)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Inferencia: método=%s  ángulo=%.2f°  tiempo=%.1f ms",
        effective_method, angle, elapsed_ms,
    )

    # Visualización opcional
    viz_b64: Optional[str] = None
    if viz:
        viz_b64 = _build_visualization_b64(img, angle)

    return InferResponse(
        angle_deg=round(angle, 4),
        method_used=effective_method,
        processing_time_ms=round(elapsed_ms, 2),
        visualization_b64=viz_b64,
    )


# ---------------------------------------------------------------------------
# Funciones internas de inferencia
# ---------------------------------------------------------------------------

def _decode_image(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Decodifica los bytes de la imagen y la convierte a uint8 128×128 en grises.

    Args:
        raw_bytes: Contenido binario del archivo subido.
        filename: Nombre del archivo (solo para mensajes de error).

    Returns:
        Array uint8 de shape (128, 128).

    Raises:
        HTTPException 400 si la imagen no se puede decodificar.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo decodificar la imagen '{filename}'. "
                   "Asegúrate de que sea un PNG o JPG válido.",
        )

    # Redimensionar al tamaño esperado por el modelo
    if img.shape != (128, 128):
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    return img


def _infer_rl(img: np.ndarray, model) -> float:
    """Ejecuta el agente PPO sobre la imagen y devuelve el ángulo estimado.

    Args:
        img: Imagen uint8 (128, 128) en escala de grises.
        model: Modelo PPO ya cargado en memoria.

    Returns:
        Ángulo estimado en grados [0°, 180°).
    """
    from env.fiber_env import FiberOrientationEnv
    from env.synthetic_generator import generate_fiber_image

    # Construir entorno temporal para la inferencia
    env = FiberOrientationEnv()
    env._img_objetivo = img
    env._theta_estimado = 90.0   # punto de partida neutral
    env._step_count = 0
    env._img_estimada = generate_fiber_image(90.0, size=env.size)

    # Observación inicial: (1, 2, H, W) float32 normalizado
    obs_hwc = env._get_obs()  # (H, W, 2) uint8
    obs = np.transpose(obs_hwc, (2, 0, 1)).astype(np.float32)[np.newaxis] / 255.0

    done = False
    theta_estimated = 90.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        _, _, terminated, truncated, info = env.step(action.flatten())
        obs = np.transpose(env._get_obs(), (2, 0, 1)).astype(np.float32)[np.newaxis] / 255.0
        done = terminated or truncated
        theta_estimated = info.get("theta_estimated", theta_estimated)

    env.close()
    return float(theta_estimated)


def _infer_ellipse(img: np.ndarray) -> float:
    """Fallback geométrico: ajuste de elipse mínima (Plan B).

    Args:
        img: Imagen uint8 (128, 128) en escala de grises.

    Returns:
        Ángulo estimado en grados [0°, 180°).
    """
    from utils.ellipse_fallback import estimate_orientation_ellipse
    return estimate_orientation_ellipse(img)


def _build_visualization_b64(img: np.ndarray, angle_deg: float) -> str:
    """Genera una imagen PNG con el vector de orientación superpuesto, en base64.

    El vector se dibuja como una flecha bidireccional centrada en la imagen,
    orientada según el ángulo estimado.

    Args:
        img: Imagen uint8 (128, 128) en escala de grises.
        angle_deg: Ángulo de orientación estimado en grados.

    Returns:
        String base64 del PNG resultante (sin el prefijo "data:image/png;base64,").
    """
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cx, cy = 64, 64
    length = 50
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Puntos extremos del vector de orientación
    x1 = int(cx - length * cos_a)
    y1 = int(cy - length * sin_a)
    x2 = int(cx + length * cos_a)
    y2 = int(cy + length * sin_a)

    # Dibujar vector bidireccional (las fibras no tienen sentido único)
    cv2.arrowedLine(img_color, (x1, y1), (x2, y2), color=(0, 200, 50), thickness=2, tipLength=0.15)
    cv2.arrowedLine(img_color, (x2, y2), (x1, y1), color=(0, 200, 50), thickness=2, tipLength=0.15)

    # Anotar el ángulo en la imagen
    cv2.putText(
        img_color,
        f"{angle_deg:.1f}°",
        (4, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 50),
        1,
        cv2.LINE_AA,
    )

    # Codificar a PNG en memoria y convertir a base64
    success, buffer = cv2.imencode(".png", img_color)
    if not success:
        return ""

    return base64.b64encode(buffer.tobytes()).decode("utf-8")
