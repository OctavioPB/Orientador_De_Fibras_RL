"""Tests para env/fiber_env.py."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.fiber_env import FiberOrientationEnv, angular_distance


def test_check_env():
    """El entorno debe pasar la verificación estándar de Gymnasium."""
    env = FiberOrientationEnv()
    check_env(env, warn=True)
    env.close()


def test_full_episode_no_exception():
    """Un episodio completo no debe lanzar excepciones."""
    env = FiberOrientationEnv()
    obs, info = env.reset(seed=0)
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    assert steps > 0
    assert "error_deg" in info
    env.close()


def test_info_contains_required_keys():
    env = FiberOrientationEnv()
    _, info = env.reset(seed=1)
    assert "error_deg" in info
    assert "theta_target" in info
    assert "theta_estimated" in info
    env.close()


def test_directed_agent_reduces_error():
    """Un agente que siempre aplica acción dirigida debe reducir el error en promedio."""
    env = FiberOrientationEnv()
    obs, info = env.reset(seed=42)

    errors = []
    for _ in range(10):
        theta_t = info["theta_target"]
        theta_e = info["theta_estimated"]
        # Acción sign(theta_target - theta_estimated) normalizada
        delta = theta_t - theta_e
        # Corregir para simetría
        if delta > 90:
            delta -= 180
        elif delta < -90:
            delta += 180
        action = np.array([np.sign(delta)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        errors.append(info["error_deg"])
        if terminated or truncated:
            break

    # El error en los últimos pasos debe ser menor que en los primeros
    if len(errors) >= 4:
        first_half = np.mean(errors[: len(errors) // 2])
        second_half = np.mean(errors[len(errors) // 2 :])
        assert second_half <= first_half + 5.0, (
            f"Se esperaba reducción de error: primera mitad {first_half:.2f}° → segunda mitad {second_half:.2f}°"
        )

    env.close()


def test_angular_distance_symmetry():
    assert angular_distance(0.0, 180.0) == pytest.approx(0.0, abs=1e-6)
    assert angular_distance(10.0, 170.0) == pytest.approx(20.0, abs=1e-6)
    assert angular_distance(0.0, 90.0) == pytest.approx(90.0, abs=1e-6)
    assert angular_distance(45.0, 135.0) == pytest.approx(90.0, abs=1e-6)


def test_observation_shape_and_range():
    env = FiberOrientationEnv()
    obs, _ = env.reset(seed=7)
    assert obs.shape == (128, 128, 2)
    assert obs.dtype == np.uint8
    assert obs.min() >= 0
    assert obs.max() <= 255
    env.close()
