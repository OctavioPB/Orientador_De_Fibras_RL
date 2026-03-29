"""Función de recompensa basada en similitud estructural (SSIM)."""

import logging

import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def compute_reward(
    img_real: np.ndarray,
    img_synthetic: np.ndarray,
    step_penalty: float = 0.01,
) -> float:
    """Calcula recompensa como función de similitud estructural.

    La recompensa se normaliza al rango [-1, 1] usando la fórmula:
        reward = 2 * SSIM(img_real, img_synthetic) - 1 - step_penalty

    Args:
        img_real: Imagen objetivo de referencia.
        img_synthetic: Imagen estimada actual por el agente.
        step_penalty: Penalización por paso para incentivar convergencia rápida.

    Returns:
        Recompensa escalar en el rango aproximado [-1, 1].
    """
    if img_real.shape != img_synthetic.shape:
        raise ValueError(
            f"Las imágenes deben tener el mismo shape: {img_real.shape} vs {img_synthetic.shape}"
        )

    # SSIM requiere float64; el dtype original se guarda antes de la conversión
    # para determinar el data_range correcto (255 para uint8, 1.0 para float)
    original_dtype = img_real.dtype
    real = img_real.astype(np.float64)
    synth = img_synthetic.astype(np.float64)

    # data_range le indica a SSIM cuál es el rango máximo posible de los píxeles
    data_range = 255.0 if original_dtype == np.uint8 else 1.0

    # SSIM ∈ [0, 1] → reescalar a [-1, 1] para que el agente reciba
    # recompensa negativa al inicio (cuando la estimación es mala)
    similarity = ssim(real, synth, data_range=data_range)
    reward = 2.0 * similarity - 1.0 - step_penalty

    logger.debug("SSIM=%.4f  reward=%.4f", similarity, reward)
    return float(reward)
