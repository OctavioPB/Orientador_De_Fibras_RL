"""Evaluación formal del agente PPO sobre imágenes sintéticas con ángulos conocidos."""

import csv
import logging
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

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

    def make_env():
        return FiberOrientationEnv()

    vec_env = DummyVecEnv([make_env])
    vec_env = VecTransposeImage(vec_env)

    for theta_true in thetas_true:
        obs = vec_env.reset()

        # Inyectar imagen con theta conocido sobreescribiendo el estado interno
        raw_env: FiberOrientationEnv = vec_env.envs[0]  # type: ignore[attr-defined]
        raw_env._theta_objetivo = float(theta_true)
        raw_env._theta_estimado = 90.0
        raw_env._step_count = 0
        raw_env._img_objetivo = generate_fiber_image(theta_true, size=raw_env.size)

        # Reconstruir obs con la imagen inyectada
        img_norm = raw_env._img_objetivo.astype(np.float32) / 255.0
        obs_np = img_norm[np.newaxis, np.newaxis, ...]  # (1, 1, H, W) → VecTransposeImage convierte a (1, H, W, 1) → luego transpone

        # Re-wrap: usar obs directamente del env (más limpio)
        obs_list = [raw_env._get_obs()]
        obs = np.array(obs_list)  # (1, H, W, 1) → VecTransposeImage convierte a (1, 1, H, W)

        done = False
        theta_predicted = raw_env._theta_estimado

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            theta_predicted = infos[0].get("theta_estimated", theta_predicted)

        error = angular_distance(theta_predicted, theta_true)
        results.append((float(theta_true), float(theta_predicted), float(error)))
        logger.debug("theta_true=%.2f  theta_pred=%.2f  error=%.2f", theta_true, theta_predicted, error)

    vec_env.close()

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
