"""Módulo de inferencia standalone del modelo preentrenado de orientación de fibras.

Encapsula la carga del modelo PPO y expone una única función pública `predict`
que cualquier UI puede importar y llamar sin conocer los detalles internos del módulo RL.

Uso desde cualquier UI:
    from pretrained_model import FiberOrientationModel

    model = FiberOrientationModel("models/ppo_v1")   # carga una sola vez
    angle = model.predict("ruta/fibra.png")          # devuelve float en [0°, 180°)
    angle, viz = model.predict("ruta/fibra.png", return_visualization=True)
"""

import base64
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Tamaño de imagen esperado por el modelo
_IMG_SIZE = 128


class FiberOrientationModel:
    """Modelo preentrenado de orientación de fibras musculares.

    Carga el modelo PPO una sola vez en memoria y reutiliza la instancia
    para todas las predicciones. Si el modelo no está disponible, cae
    automáticamente al método geométrico de ajuste de elipse (Plan B).

    Args:
        model_path: Ruta al archivo del modelo PPO (con o sin extensión .zip).
                    Si es None o el archivo no existe, se usa el fallback geométrico.
        method: "auto" (RL si existe, sino elipse) | "rl" | "ellipse".

    Example:
        >>> model = FiberOrientationModel("models/ppo_v1")
        >>> angle = model.predict("imagen.png")
        >>> print(f"Orientación estimada: {angle:.1f}°")
    """

    def __init__(
        self,
        model_path: Optional[str] = "models/ppo_v1",
        method: str = "auto",
    ) -> None:
        self._ppo_model = None
        self._method = method
        self._model_path = model_path
        self._active_method: str = "ellipse"  # se actualiza al cargar

        self._load_model()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        return_visualization: bool = False,
    ) -> Union[float, tuple[float, str]]:
        """Estima el ángulo de orientación dominante de las fibras.

        Args:
            image: Ruta a un archivo PNG/JPG, o array numpy uint8 de cualquier shape.
                   Si es array multicanal se convierte a escala de grises internamente.
            return_visualization: Si es True, devuelve también la imagen con el vector
                                  de orientación superpuesto, codificada en base64 PNG.

        Returns:
            Si return_visualization es False: float con el ángulo en grados [0°, 180°).
            Si return_visualization es True:  tupla (angle_deg: float, viz_b64: str).

        Raises:
            ValueError: Si la imagen no se puede cargar o decodificar.
        """
        img = self._load_image(image)

        if self._active_method == "rl" and self._ppo_model is not None:
            angle = self._predict_rl(img)
        else:
            angle = self._predict_ellipse(img)

        logger.debug("predict → ángulo=%.2f° (método=%s)", angle, self._active_method)

        if return_visualization:
            viz_b64 = self._build_visualization_b64(img, angle)
            return angle, viz_b64

        return angle

    @property
    def active_method(self) -> str:
        """Método de inferencia activo: 'rl' o 'ellipse'."""
        return self._active_method

    @property
    def is_rl_loaded(self) -> bool:
        """True si el modelo PPO está cargado en memoria."""
        return self._ppo_model is not None

    def __repr__(self) -> str:
        return (
            f"FiberOrientationModel("
            f"model_path={self._model_path!r}, "
            f"active_method={self._active_method!r}, "
            f"rl_loaded={self.is_rl_loaded})"
        )

    # ------------------------------------------------------------------
    # Métodos privados — carga y configuración
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Intenta cargar el modelo PPO; si falla, activa el fallback de elipse."""
        if self._method == "ellipse":
            # El usuario pidió explícitamente el método geométrico
            self._active_method = "ellipse"
            logger.info("Método forzado: ellipse (fallback geométrico).")
            return

        if not self._model_path:
            logger.warning("No se especificó ruta de modelo. Usando fallback ellipse.")
            self._active_method = "ellipse"
            return

        # Resolver ruta con o sin extensión .zip
        path = Path(self._model_path)
        zip_path = path.with_suffix(".zip")
        resolved = zip_path if zip_path.exists() else path

        if not resolved.exists() and not path.with_suffix(".zip").exists():
            logger.warning(
                "Modelo no encontrado en '%s'. Usando fallback ellipse.", self._model_path
            )
            self._active_method = "ellipse"
            return

        try:
            from stable_baselines3 import PPO
            self._ppo_model = PPO.load(str(path))
            self._active_method = "rl"
            logger.info("Modelo PPO cargado desde '%s'.", self._model_path)
        except Exception as exc:
            logger.error("Error cargando modelo PPO (%s). Usando fallback ellipse.", exc)
            self._active_method = "ellipse"

    # ------------------------------------------------------------------
    # Métodos privados — preprocesamiento
    # ------------------------------------------------------------------

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Carga y normaliza la imagen a uint8 128×128 escala de grises.

        Args:
            image: Ruta de archivo o array numpy.

        Returns:
            Array uint8 de shape (128, 128).

        Raises:
            ValueError: Si la imagen no puede cargarse.
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image}")
        elif isinstance(image, np.ndarray):
            # Convertir a escala de grises si es multicanal
            if image.ndim == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img = image.copy()
            # Asegurar dtype uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Tipo de imagen no soportado: {type(image)}")

        # Redimensionar si es necesario
        if img.shape != (_IMG_SIZE, _IMG_SIZE):
            img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)

        return img

    # ------------------------------------------------------------------
    # Métodos privados — inferencia
    # ------------------------------------------------------------------

    def _predict_rl(self, img: np.ndarray) -> float:
        """Ejecuta el agente PPO para estimar el ángulo.

        Adapta automáticamente el formato de la observación al número de canales
        con el que fue entrenado el modelo (1 canal = solo imagen objetivo;
        2 canales = imagen objetivo + imagen estimada).

        Args:
            img: Imagen uint8 (128, 128) en escala de grises.

        Returns:
            Ángulo estimado en grados [0°, 180°).
        """
        from env.fiber_env import FiberOrientationEnv
        from env.synthetic_generator import generate_fiber_image

        # Leer cuántos canales espera el modelo desde su observation_space guardado.
        # Después de VecTransposeImage el shape almacenado es (C, H, W).
        n_channels = self._ppo_model.observation_space.shape[0]

        # Construir entorno temporal sin VecEnv (más ligero para inferencia puntual)
        env = FiberOrientationEnv()
        env._img_objetivo = img
        env._theta_estimado = 90.0
        env._step_count = 0
        env._img_estimada = generate_fiber_image(90.0, size=env.size)

        def _make_obs() -> np.ndarray:
            """Construye la observación en el formato exacto que espera el modelo."""
            if n_channels == 1:
                # Modelo entrenado solo con la imagen objetivo → (1, 1, H, W)
                return img.astype(np.float32)[np.newaxis, np.newaxis] / 255.0
            else:
                # Modelo entrenado con 2 canales [objetivo, estimada] → (1, 2, H, W)
                return np.transpose(env._get_obs(), (2, 0, 1)).astype(np.float32)[np.newaxis] / 255.0

        obs = _make_obs()
        done = False
        theta_estimated = 90.0

        while not done:
            action, _ = self._ppo_model.predict(obs, deterministic=True)
            _, _, terminated, truncated, info = env.step(action.flatten())
            obs = _make_obs()
            done = terminated or truncated
            theta_estimated = info.get("theta_estimated", theta_estimated)

        env.close()
        return float(theta_estimated)

    def _predict_ellipse(self, img: np.ndarray) -> float:
        """Estima el ángulo mediante ajuste de elipse mínima (fallback geométrico).

        Args:
            img: Imagen uint8 (128, 128) en escala de grises.

        Returns:
            Ángulo estimado en grados [0°, 180°).
        """
        from utils.ellipse_fallback import estimate_orientation_ellipse
        return estimate_orientation_ellipse(img)

    # ------------------------------------------------------------------
    # Métodos privados — visualización
    # ------------------------------------------------------------------

    def _build_visualization_b64(self, img: np.ndarray, angle_deg: float) -> str:
        """Genera imagen PNG con el vector de orientación superpuesto en base64.

        Args:
            img: Imagen uint8 (128, 128) en escala de grises.
            angle_deg: Ángulo de orientación estimado en grados.

        Returns:
            String base64 del PNG resultante (sin prefijo data:image/...).
        """
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cx, cy = _IMG_SIZE // 2, _IMG_SIZE // 2
        length = 45
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Vector bidireccional (fibras no tienen sentido único)
        x1 = int(cx - length * cos_a)
        y1 = int(cy - length * sin_a)
        x2 = int(cx + length * cos_a)
        y2 = int(cy + length * sin_a)

        cv2.arrowedLine(img_color, (x1, y1), (x2, y2), (0, 200, 50), 2, tipLength=0.15)
        cv2.arrowedLine(img_color, (x2, y2), (x1, y1), (0, 200, 50), 2, tipLength=0.15)
        cv2.putText(
            img_color, f"{angle_deg:.1f}deg",
            (4, _IMG_SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 200, 50), 1, cv2.LINE_AA,
        )

        _, buf = cv2.imencode(".png", img_color)
        return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Uso directo desde la línea de comandos (prueba rápida)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inferencia rápida del modelo preentrenado")
    parser.add_argument("image", help="Ruta a la imagen PNG o JPG")
    parser.add_argument("--model", default="models/ppo_v1", help="Ruta al modelo PPO")
    parser.add_argument("--method", choices=["auto", "rl", "ellipse"], default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    m = FiberOrientationModel(model_path=args.model, method=args.method)
    print(m)

    angle, viz_b64 = m.predict(args.image, return_visualization=True)
    print(f"Ángulo estimado: {angle:.2f}°  (método: {m.active_method})")

    # Guardar visualización
    if viz_b64:
        out_path = Path(args.image).with_stem(Path(args.image).stem + "_result").with_suffix(".png")
        out_path.write_bytes(base64.b64decode(viz_b64))
        print(f"Visualización guardada en: {out_path}")
