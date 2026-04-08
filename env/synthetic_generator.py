"""Generador de imágenes sintéticas de fibras musculares con orientación controlada."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_fiber_image(
    theta: float,
    n_fibers: int = 12,
    noise_std: float = 8.0,
    size: int = 128,
) -> np.ndarray:
    """Genera imagen sintética de fibras con orientación theta (grados).

    Args:
        theta: Ángulo de orientación dominante en grados [0, 180).
        n_fibers: Número de fibras (5–20).
        noise_std: Desviación estándar del ruido gaussiano (0–15).
        size: Tamaño del lado de la imagen cuadrada en píxeles.

    Returns:
        Array uint8 de shape (size, size).
    """
    image = np.zeros((size, size), dtype=np.float32)
    rng = np.random.RandomState(int(theta * 1000) % (2**31))

    for _ in range(n_fibers):
        cx = rng.randint(0, size)
        cy = rng.randint(0, size)
        length = rng.randint(30, 61)
        width = rng.randint(4, 9)
        # Variación ±5° para simular irregularidades de tinción
        fiber_angle = theta + rng.uniform(-5.0, 5.0)

        cv2.ellipse(
            image,
            (cx, cy),
            (length // 2, width // 2),
            angle=fiber_angle,
            startAngle=0,
            endAngle=360,
            color=rng.uniform(180, 255),
            thickness=-1,
        )

    if noise_std > 0:
        image += rng.normal(0.0, noise_std, image.shape).astype(np.float32)

    logger.debug("Imagen generada: theta=%.1f, n_fibers=%d, noise_std=%.1f", theta, n_fibers, noise_std)
    return np.clip(image, 0, 255).astype(np.uint8)
