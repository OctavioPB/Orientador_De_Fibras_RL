"""Función de recompensa basada en similitud estructural (SSIM)."""

import logging

import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def compute_reward(img_real: np.ndarray, img_synthetic: np.ndarray, step_penalty: float = 0.01) -> float:
    """Calcula la recompensa como 2·SSIM − 1 − step_penalty.

    Args:
        img_real: Imagen objetivo de referencia.
        img_synthetic: Imagen estimada por el agente.
        step_penalty: Penalización por paso para incentivar convergencia rápida.

    Returns:
        Recompensa escalar en el rango aproximado [-1, 1].
    """
    if img_real.shape != img_synthetic.shape:
        raise ValueError(f"Shape mismatch: {img_real.shape} vs {img_synthetic.shape}")

    # data_range indica a SSIM el rango máximo posible de los píxeles (255 para uint8, 1.0 para float)
    data_range = 255.0 if img_real.dtype == np.uint8 else 1.0
    similarity = ssim(img_real.astype(np.float64), img_synthetic.astype(np.float64), data_range=data_range)
    reward = 2.0 * similarity - 1.0 - step_penalty

    logger.debug("SSIM=%.4f  reward=%.4f", similarity, reward)
    return float(reward)
