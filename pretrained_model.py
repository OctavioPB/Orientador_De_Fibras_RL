"""Interfaz de inferencia standalone del modelo preentrenado.

Uso:
    from pretrained_model import FiberOrientationModel

    model = FiberOrientationModel("models/ppo_v1")
    angle = model.predict("fibra.png")
    angle, viz_b64 = model.predict("fibra.png", return_visualization=True)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from utils.visualization import build_visualization_b64

logger = logging.getLogger(__name__)

_IMG_SIZE = 128


class FiberOrientationModel:
    """Modelo preentrenado de orientación de fibras musculares.

    Carga el modelo PPO una vez y lo reutiliza. Si el modelo no está disponible,
    cae automáticamente al método geométrico de ajuste de elipse.

    Args:
        model_path: Ruta al archivo PPO (con o sin extensión .zip).
                    Si es None o no existe, se usa el fallback geométrico.
        method: "auto" (RL si existe, sino elipse) | "rl" | "ellipse".

    Example:
        >>> model = FiberOrientationModel("models/ppo_v1")
        >>> angle = model.predict("imagen.png")
    """

    def __init__(self, model_path: Optional[str] = "models/ppo_v1", method: str = "auto") -> None:
        self._ppo_model = None
        self._method = method
        self._model_path = model_path
        self._active_method: str = "ellipse"
        self._load_model()

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        return_visualization: bool = False,
    ) -> Union[float, tuple[float, str]]:
        """Estima el ángulo de orientación dominante de las fibras.

        Args:
            image: Ruta PNG/JPG o array numpy uint8.
            return_visualization: Si True, devuelve también el PNG con vector superpuesto en base64.

        Returns:
            float con el ángulo en grados [0°, 180°), o tupla (angle, viz_b64).
        """
        img = self._load_image(image)
        angle = self._predict_rl(img) if self._active_method == "rl" else self._predict_ellipse(img)
        logger.debug("predict → %.2f° (método=%s)", angle, self._active_method)

        if return_visualization:
            return angle, build_visualization_b64(img, angle)
        return angle

    @property
    def active_method(self) -> str:
        return self._active_method

    @property
    def is_rl_loaded(self) -> bool:
        return self._ppo_model is not None

    def __repr__(self) -> str:
        return (
            f"FiberOrientationModel(model_path={self._model_path!r}, "
            f"active_method={self._active_method!r}, rl_loaded={self.is_rl_loaded})"
        )

    def _load_model(self) -> None:
        if self._method == "ellipse":
            self._active_method = "ellipse"
            return

        if not self._model_path:
            logger.warning("No se especificó ruta de modelo. Usando fallback ellipse.")
            return

        path = Path(self._model_path)
        zip_path = path.with_suffix(".zip")
        if not path.exists() and not zip_path.exists():
            logger.warning("Modelo no encontrado en '%s'. Usando fallback ellipse.", self._model_path)
            return

        try:
            from stable_baselines3 import PPO
            self._ppo_model = PPO.load(str(path))
            self._active_method = "rl"
            logger.info("Modelo PPO cargado desde '%s'.", self._model_path)
        except Exception as exc:
            logger.error("Error cargando modelo PPO (%s). Usando fallback ellipse.", exc)

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image}")
        elif isinstance(image, np.ndarray):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Tipo de imagen no soportado: {type(image)}")

        if img.shape != (_IMG_SIZE, _IMG_SIZE):
            img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)
        return img

    def _predict_rl(self, img: np.ndarray) -> float:
        from env.fiber_env import FiberOrientationEnv
        from env.synthetic_generator import generate_fiber_image

        # El shape almacenado en observation_space es (C, H, W) tras VecTransposeImage
        n_channels = self._ppo_model.observation_space.shape[0]

        env = FiberOrientationEnv()
        env._img_objetivo = img
        env._theta_estimado = 90.0
        env._step_count = 0
        env._img_estimada = generate_fiber_image(90.0, size=env.size)

        def _obs() -> np.ndarray:
            if n_channels == 1:
                return img.astype(np.float32)[np.newaxis, np.newaxis] / 255.0
            return np.transpose(env._get_obs(), (2, 0, 1)).astype(np.float32)[np.newaxis] / 255.0

        obs = _obs()
        done = False
        theta = 90.0

        while not done:
            action, _ = self._ppo_model.predict(obs, deterministic=True)
            _, _, terminated, truncated, info = env.step(action.flatten())
            obs = _obs()
            done = terminated or truncated
            theta = info.get("theta_estimated", theta)

        env.close()
        return float(theta)

    def _predict_ellipse(self, img: np.ndarray) -> float:
        from utils.ellipse_fallback import estimate_orientation_ellipse
        return estimate_orientation_ellipse(img)


if __name__ == "__main__":
    import argparse
    import base64

    parser = argparse.ArgumentParser(description="Inferencia rápida del modelo preentrenado")
    parser.add_argument("image", help="Ruta a la imagen PNG o JPG")
    parser.add_argument("--model", default="models/ppo_v1")
    parser.add_argument("--method", choices=["auto", "rl", "ellipse"], default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    m = FiberOrientationModel(model_path=args.model, method=args.method)
    print(m)

    angle, viz_b64 = m.predict(args.image, return_visualization=True)
    print(f"Ángulo estimado: {angle:.2f}°  (método: {m.active_method})")

    out_path = Path(args.image).with_stem(Path(args.image).stem + "_result").with_suffix(".png")
    out_path.write_bytes(base64.b64decode(viz_b64))
    print(f"Visualización guardada en: {out_path}")
