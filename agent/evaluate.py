"""Evaluación formal del agente PPO sobre imágenes sintéticas con ángulos conocidos."""

import csv
import logging
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from env.fiber_env import FiberOrientationEnv, angular_distance
from env.synthetic_generator import generate_fiber_image

logger = logging.getLogger(__name__)


def evaluate(
    model_path: str,
    n_images: int = 100,
    output_csv: str = "results/evaluation.csv",
) -> dict:
    """Evalúa el modelo entrenado sobre n_images sintéticas con ángulos conocidos.

    Genera imágenes con ángulos uniformemente distribuidos en [0°, 180°),
    ejecuta inferencia con el agente y calcula métricas angulares.

    Args:
        model_path: Ruta al modelo PPO guardado (sin extensión .zip).
        n_images: Número de imágenes de evaluación.
        output_csv: Ruta del CSV de salida con columnas [theta_true, theta_predicted, error_deg].

    Returns:
        Diccionario con métricas:
            - mae: Error absoluto medio en grados.
            - pct_lt5: Porcentaje de imágenes con error < 5°.
            - pct_lt10: Porcentaje de imágenes con error < 10°.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Cargando modelo desde '%s'.", model_path)
    model = PPO.load(model_path)

    thetas_true = np.linspace(0.0, 180.0, n_images, endpoint=False)
    results = []

    env = FiberOrientationEnv()

    for theta_true in thetas_true:
        # Inyectar theta conocido directamente en el entorno
        env._theta_objetivo = float(theta_true)
        env._theta_estimado = 90.0
        env._step_count = 0
        env._img_objetivo = generate_fiber_image(float(theta_true), size=env.size)
        env._img_estimada = generate_fiber_image(env._theta_estimado, size=env.size)

        # Observación en formato CHW float32 normalizado (lo que espera CnnPolicy)
        obs = _to_policy_obs(env._get_obs())

        done = False
        theta_predicted = env._theta_estimado

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Aplanar acción: model.predict devuelve (1,1) sin VecEnv
            _, _, terminated, truncated, info = env.step(action.flatten())
            obs = _to_policy_obs(env._get_obs())
            done = terminated or truncated
            theta_predicted = info.get("theta_estimated", theta_predicted)

        error = angular_distance(theta_predicted, float(theta_true))
        results.append((float(theta_true), float(theta_predicted), float(error)))
        logger.debug("theta_true=%.2f  theta_pred=%.2f  error=%.2f", theta_true, theta_predicted, error)

    env.close()

    errors = [r[2] for r in results]
    mae = float(np.mean(errors))
    pct_lt5 = float(np.mean([e < 5.0 for e in errors]) * 100)
    pct_lt10 = float(np.mean([e < 10.0 for e in errors]) * 100)

    metrics = {"mae": mae, "pct_lt5": pct_lt5, "pct_lt10": pct_lt10}
    logger.info("MAE=%.2f°  | <5°: %.1f%%  | <10°: %.1f%%", mae, pct_lt5, pct_lt10)

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else "results", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_true", "theta_predicted", "error_deg"])
        writer.writerows(results)

    logger.info("Resultados guardados en '%s'.", output_csv)
    return metrics


def _to_policy_obs(obs_hwc: np.ndarray) -> np.ndarray:
    """Convierte observación (H, W, C) uint8 → (1, C, H, W) float32 normalizado.

    Replica lo que hacen DummyVecEnv + VecTransposeImage + normalize_images.

    Args:
        obs_hwc: Observación en formato (H, W, C) uint8.

    Returns:
        Array (1, C, H, W) float32 listo para model.predict.
    """
    # HWC → CHW
    obs_chw = np.transpose(obs_hwc, (2, 0, 1)).astype(np.float32) / 255.0
    return obs_chw[np.newaxis, ...]  # (1, C, H, W)
