"""Utilidades de visualización compartidas entre la API y el módulo de inferencia."""

import base64

import cv2
import numpy as np

_IMG_SIZE = 128
_ARROW_COLOR = (0, 200, 50)


def build_visualization_b64(img: np.ndarray, angle_deg: float) -> str:
    """Dibuja el vector de orientación sobre la imagen y devuelve el PNG en base64.

    Args:
        img: Imagen uint8 (H, W) en escala de grises.
        angle_deg: Ángulo estimado en grados [0°, 180°).

    Returns:
        PNG codificado en base64 (sin prefijo data:image/...).
    """
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cx = cy = _IMG_SIZE // 2
    length = 50
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    p1 = (int(cx - length * cos_a), int(cy - length * sin_a))
    p2 = (int(cx + length * cos_a), int(cy + length * sin_a))

    # Doble flecha bidireccional: las fibras no tienen sentido único
    cv2.arrowedLine(img_color, p1, p2, _ARROW_COLOR, 2, tipLength=0.15)
    cv2.arrowedLine(img_color, p2, p1, _ARROW_COLOR, 2, tipLength=0.15)
    cv2.putText(
        img_color, f"{angle_deg:.1f}deg",
        (4, _IMG_SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX,
        0.4, _ARROW_COLOR, 1, cv2.LINE_AA,
    )

    success, buf = cv2.imencode(".png", img_color)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if success else ""
