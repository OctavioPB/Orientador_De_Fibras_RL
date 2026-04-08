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
    """Evalúa el modelo sobre n_images sintéticas con ángulos uniformes en [0°, 180°).

    Args:
        model_path: Ruta al modelo PPO guardado (sin extensión .zip).
        n_images: Número de imágenes de evaluación.
        output_csv: Ruta del CSV de salida con columnas [theta_true, theta_predicted, error_deg].

    Returns:
        Diccionario con mae, pct_lt5 y pct_lt10.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Cargando modelo desde '%s'.", model_path)
    model = PPO.load(model_path)

    thetas_true = np.linspace(0.0, 180.0, n_images, endpoint=False)
    results = []
    env = FiberOrientationEnv()

    for theta_true in thetas_true:
        env._theta_objetivo = float(theta_true)
        env._theta_estimado = 90.0
        env._step_count = 0
        env._img_objetivo = generate_fiber_image(float(theta_true), size=env.size)
        env._img_estimada = generate_fiber_image(env._theta_estimado, size=env.size)

        obs = _to_policy_obs(env._get_obs())
        done = False
        theta_predicted = env._theta_estimado

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            _, _, terminated, truncated, info = env.step(action.flatten())
            obs = _to_policy_obs(env._get_obs())
            done = terminated or truncated
            theta_predicted = info.get("theta_estimated", theta_predicted)

        error = angular_distance(theta_predicted, float(theta_true))
        results.append((float(theta_true), float(theta_predicted), float(error)))
        logger.debug("theta_true=%.2f  theta_pred=%.2f  error=%.2f", theta_true, theta_predicted, error)

    env.close()

    errors = [r[2] for r in results]
    metrics = {
        "mae": float(np.mean(errors)),
        "pct_lt5": float(np.mean([e < 5.0 for e in errors]) * 100),
        "pct_lt10": float(np.mean([e < 10.0 for e in errors]) * 100),
    }
    logger.info("MAE=%.2f°  | <5°: %.1f%%  | <10°: %.1f%%", metrics["mae"], metrics["pct_lt5"], metrics["pct_lt10"])

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else "results", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_true", "theta_predicted", "error_deg"])
        writer.writerows(results)

    logger.info("Resultados guardados en '%s'.", output_csv)
    return metrics


def _to_policy_obs(obs_hwc: np.ndarray) -> np.ndarray:
    """Convierte observación (H, W, C) uint8 → (1, C, H, W) float32 para model.predict."""
    return np.transpose(obs_hwc, (2, 0, 1)).astype(np.float32)[np.newaxis] / 255.0
