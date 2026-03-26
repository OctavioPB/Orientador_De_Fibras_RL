"""Loop de entrenamiento PPO para el agente de orientación de fibras."""

import logging
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from env.fiber_env import FiberOrientationEnv

logger = logging.getLogger(__name__)


class MeanAngularErrorCallback(BaseCallback):
    """Callback que detiene el entrenamiento si el MAE < umbral por N evaluaciones consecutivas.

    Args:
        eval_env: Entorno de evaluación.
        mae_threshold: MAE angular objetivo (grados).
        n_consecutive: Evaluaciones consecutivas por debajo del umbral para detener.
        eval_freq: Frecuencia de evaluación en timesteps.
        n_eval_episodes: Episodios por evaluación.
    """

    def __init__(
        self,
        eval_env: DummyVecEnv,
        mae_threshold: float = 8.0,
        n_consecutive: int = 3,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
    ) -> None:
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.mae_threshold = mae_threshold
        self.n_consecutive = n_consecutive
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._consecutive_count = 0
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps
        errors = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.eval_env.step(action)
                done = dones[0]
                if done:
                    errors.append(infos[0].get("error_deg", 90.0))

        mae = float(np.mean(errors)) if errors else 90.0
        logger.info("MAE angular (eval) = %.2f°", mae)

        if mae < self.mae_threshold:
            self._consecutive_count += 1
            logger.info(
                "MAE < %.1f° por %d evaluación(es) consecutiva(s) (objetivo: %d)",
                self.mae_threshold, self._consecutive_count, self.n_consecutive,
            )
            if self._consecutive_count >= self.n_consecutive:
                logger.info("Criterio de parada anticipada alcanzado. Deteniendo entrenamiento.")
                return False
        else:
            self._consecutive_count = 0

        return True


def train(
    total_timesteps: int = 500_000,
    save_path: str = "models/ppo_fiber_orientation",
    log_dir: str = "logs/",
) -> PPO:
    """Entrena agente PPO en FiberOrientationEnv.

    Args:
        total_timesteps: Número máximo de pasos de entrenamiento.
        save_path: Ruta donde se guarda el modelo final.
        log_dir: Directorio para logs de TensorBoard.

    Returns:
        Modelo PPO entrenado.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        return FiberOrientationEnv()

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    eval_env_raw = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env_raw)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=None,
        policy_kwargs={"normalize_images": True},
    )

    checkpoint_callback = _CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.dirname(save_path) or "models",
        name_prefix=os.path.basename(save_path),
    )

    early_stop_callback = MeanAngularErrorCallback(
        eval_env=eval_env,
        mae_threshold=8.0,
        n_consecutive=3,
        eval_freq=10_000,
        n_eval_episodes=20,
    )

    logger.info("Iniciando entrenamiento PPO por %d timesteps.", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, early_stop_callback],
    )

    model.save(save_path)
    logger.info("Modelo guardado en '%s'.", save_path)

    env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Checkpoint callback ligero (evita dep. de sb3-contrib)
# ---------------------------------------------------------------------------

class _CheckpointCallback(BaseCallback):
    """Guarda checkpoints del modelo cada save_freq timesteps."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str) -> None:
        super().__init__(verbose=0)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps",
            )
            self.model.save(path)
            logger.info("Checkpoint guardado: %s", path)
        return True
