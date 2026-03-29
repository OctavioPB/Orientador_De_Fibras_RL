"""Entorno Gymnasium para estimación de orientación de fibras musculares."""

import logging
from typing import Any, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from env.synthetic_generator import generate_fiber_image
from utils.reward import compute_reward

logger = logging.getLogger(__name__)

MAX_DELTA_DEG = 10.0
MAX_STEPS = 200
TERMINATION_THRESHOLD_DEG = 5.0
INITIAL_THETA_ESTIMATE = 90.0


def angular_distance(a: float, b: float) -> float:
    """Calcula la distancia angular entre dos ángulos en [0°, 90°].

    Las fibras son simétricas (0° == 180°), por lo que se usa:
        min(|a - b|, 180 - |a - b|)

    Args:
        a: Primer ángulo en grados.
        b: Segundo ángulo en grados.

    Returns:
        Distancia angular en grados, en el rango [0, 90].
    """
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


class FiberOrientationEnv(gymnasium.Env):
    """Entorno de orientación de fibras musculares.

    Observation: 2 canales HWC (H, W, 2) uint8:
                   canal 0 = imagen objetivo, canal 1 = imagen estimada actual.
                 Esto permite al agente comparar ambas imágenes y ajustar su estimación.
    Action:      continua, shape (1,), rango [-1, 1].
                 Se mapea a delta_theta: acción * MAX_DELTA_DEG.
    Reward:      compute_reward(img_objetivo, img_estimada_actual).
    Episode:     termina si |error_angular| < 5° o si steps > MAX_STEPS (200).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None, size: int = 128) -> None:
        """Inicializa el entorno.

        Args:
            render_mode: Modo de renderizado ('human' o None).
            size: Tamaño de la imagen cuadrada en píxeles.
        """
        super().__init__()
        self.render_mode = render_mode
        self.size = size

        # HWC 2 canales: [img_objetivo, img_estimada] — el agente ve ambas para comparar
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 2),
            dtype=np.uint8,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._theta_objetivo: float = 0.0
        self._theta_estimado: float = INITIAL_THETA_ESTIMATE
        self._img_objetivo: Optional[np.ndarray] = None
        self._img_estimada: Optional[np.ndarray] = None
        self._step_count: int = 0

        self._fig = None
        self._axes = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reinicia el entorno sorteando un nuevo ángulo objetivo.

        Args:
            seed: Semilla para reproducibilidad.
            options: Opciones adicionales (no usadas).

        Returns:
            Tupla (observación, info).
        """
        # super().reset() inicializa self.np_random con la semilla dada
        super().reset(seed=seed)

        # Sortear un ángulo objetivo aleatorio; el agente debe encontrarlo
        self._theta_objetivo = self.np_random.uniform(0.0, 180.0)
        # La estimación inicial siempre arranca en 90° (punto neutro del rango)
        self._theta_estimado = INITIAL_THETA_ESTIMATE
        self._step_count = 0

        self._img_objetivo = generate_fiber_image(self._theta_objetivo, size=self.size)
        self._img_estimada = generate_fiber_image(self._theta_estimado, size=self.size)

        obs = self._get_obs()
        info = self._get_info()

        logger.debug("reset() → theta_objetivo=%.2f", self._theta_objetivo)

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Ejecuta un paso en el entorno.

        Args:
            action: Acción del agente, shape (1,), rango [-1, 1].

        Returns:
            Tupla (obs, reward, terminated, truncated, info).
        """
        # Convertir acción ∈ [-1, 1] a un delta angular ∈ [-MAX_DELTA_DEG, MAX_DELTA_DEG]
        delta = float(action[0]) * MAX_DELTA_DEG
        # Clampear a [0°, 180°) para mantener el ángulo dentro del espacio válido
        self._theta_estimado = float(np.clip(self._theta_estimado + delta, 0.0, 179.999))
        self._step_count += 1

        self._img_estimada = generate_fiber_image(self._theta_estimado, size=self.size)
        reward = compute_reward(self._img_objetivo, self._img_estimada)

        error = angular_distance(self._theta_estimado, self._theta_objetivo)
        terminated = bool(error < TERMINATION_THRESHOLD_DEG)
        truncated = bool(self._step_count >= MAX_STEPS)

        obs = self._get_obs()
        info = self._get_info(error=error)

        logger.debug(
            "step=%d  action=%.3f  delta=%.2f  theta_est=%.2f  error=%.2f  reward=%.4f  term=%s",
            self._step_count, float(action[0]), delta, self._theta_estimado, error, reward, terminated,
        )

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Renderiza el estado actual del entorno en modo 'human'."""
        if self.render_mode == "human":
            self._render_frame()

    def close(self) -> None:
        """Libera recursos de visualización."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Devuelve observación 2 canales (H, W, 2) uint8: [objetivo, estimada]."""
        return np.stack([self._img_objetivo, self._img_estimada], axis=-1)  # (H, W, 2)

    def _get_info(self, error: Optional[float] = None) -> dict[str, Any]:
        """Construye el diccionario info."""
        if error is None:
            error = angular_distance(self._theta_estimado, self._theta_objetivo)
        return {
            "error_deg": error,
            "theta_target": self._theta_objetivo,
            "theta_estimated": self._theta_estimado,
        }

    def _render_frame(self) -> None:
        """Muestra imagen objetivo e imagen estimada lado a lado."""
        import matplotlib.pyplot as plt

        img_estimada = self._img_estimada if self._img_estimada is not None else generate_fiber_image(self._theta_estimado, size=self.size)

        if self._fig is None:
            self._fig, self._axes = plt.subplots(1, 2, figsize=(8, 4))
            plt.ion()

        self._axes[0].clear()
        self._axes[1].clear()

        self._axes[0].imshow(self._img_objetivo, cmap="gray", vmin=0, vmax=255)
        self._axes[0].set_title(f"Objetivo: {self._theta_objetivo:.1f}°")
        self._axes[0].axis("off")

        self._axes[1].imshow(img_estimada, cmap="gray", vmin=0, vmax=255)
        self._axes[1].set_title(f"Estimado: {self._theta_estimado:.1f}°")
        self._axes[1].axis("off")

        self._fig.suptitle(
            f"Step {self._step_count} | Error: {angular_distance(self._theta_estimado, self._theta_objetivo):.1f}°"
        )
        plt.pause(0.01)
