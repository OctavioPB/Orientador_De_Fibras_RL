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
    """Distancia angular simétrica entre dos ángulos en [0°, 90°].

    Las fibras son simétricas (0° == 180°), por lo que se usa:
        min(|a - b|, 180 - |a - b|)
    """
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


class FiberOrientationEnv(gymnasium.Env):
    """Entorno de orientación de fibras musculares.

    Observation: (H, W, 2) uint8 — canal 0 = imagen objetivo, canal 1 = imagen estimada.
    Action:      continua (1,) en [-1, 1] → Δθ ∈ [-10°, 10°].
    Reward:      2·SSIM(objetivo, estimada) − 1 − 0.01.
    Episode:     termina si error < 5° o steps > 200.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None, size: int = 128) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.size = size

        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 2), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._theta_objetivo: float = 0.0
        self._theta_estimado: float = INITIAL_THETA_ESTIMATE
        self._img_objetivo: Optional[np.ndarray] = None
        self._img_estimada: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._fig = None
        self._axes = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._theta_objetivo = self.np_random.uniform(0.0, 180.0)
        self._theta_estimado = INITIAL_THETA_ESTIMATE
        self._step_count = 0
        self._img_objetivo = generate_fiber_image(self._theta_objetivo, size=self.size)
        self._img_estimada = generate_fiber_image(self._theta_estimado, size=self.size)

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        delta = float(action[0]) * MAX_DELTA_DEG
        self._theta_estimado = float(np.clip(self._theta_estimado + delta, 0.0, 179.999))
        self._step_count += 1

        self._img_estimada = generate_fiber_image(self._theta_estimado, size=self.size)
        reward = compute_reward(self._img_objetivo, self._img_estimada)

        error = angular_distance(self._theta_estimado, self._theta_objetivo)
        terminated = bool(error < TERMINATION_THRESHOLD_DEG)
        truncated = bool(self._step_count >= MAX_STEPS)

        logger.debug(
            "step=%d  action=%.3f  delta=%.2f  theta_est=%.2f  error=%.2f  reward=%.4f  term=%s",
            self._step_count, float(action[0]), delta, self._theta_estimado, error, reward, terminated,
        )

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), reward, terminated, truncated, self._get_info(error=error)

    def render(self) -> None:
        if self.render_mode == "human":
            self._render_frame()

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    def _get_obs(self) -> np.ndarray:
        return np.stack([self._img_objetivo, self._img_estimada], axis=-1)

    def _get_info(self, error: Optional[float] = None) -> dict[str, Any]:
        if error is None:
            error = angular_distance(self._theta_estimado, self._theta_objetivo)
        return {
            "error_deg": error,
            "theta_target": self._theta_objetivo,
            "theta_estimated": self._theta_estimado,
        }

    def _render_frame(self) -> None:
        import matplotlib.pyplot as plt

        img_est = self._img_estimada or generate_fiber_image(self._theta_estimado, size=self.size)

        if self._fig is None:
            self._fig, self._axes = plt.subplots(1, 2, figsize=(8, 4))
            plt.ion()

        self._axes[0].clear()
        self._axes[1].clear()
        self._axes[0].imshow(self._img_objetivo, cmap="gray", vmin=0, vmax=255)
        self._axes[0].set_title(f"Objetivo: {self._theta_objetivo:.1f}°")
        self._axes[0].axis("off")
        self._axes[1].imshow(img_est, cmap="gray", vmin=0, vmax=255)
        self._axes[1].set_title(f"Estimado: {self._theta_estimado:.1f}°")
        self._axes[1].axis("off")
        self._fig.suptitle(
            f"Step {self._step_count} | Error: {angular_distance(self._theta_estimado, self._theta_objetivo):.1f}°"
        )
        plt.pause(0.01)
