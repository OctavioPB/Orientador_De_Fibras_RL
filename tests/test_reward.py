"""Tests para utils/reward.py."""

import numpy as np
import pytest

from env.synthetic_generator import generate_fiber_image
from utils.reward import compute_reward


def test_identical_images_give_max_reward():
    img = generate_fiber_image(theta=45.0)
    reward = compute_reward(img, img, step_penalty=0.0)
    # SSIM de imágenes idénticas = 1 → reward = 2*1 - 1 = 1.0
    assert abs(reward - 1.0) < 1e-4, f"Reward esperado ≈ 1.0, obtenido {reward:.4f}"


def test_orthogonal_images_give_negative_reward():
    img_0 = generate_fiber_image(theta=0.0)
    img_90 = generate_fiber_image(theta=90.0)
    reward = compute_reward(img_0, img_90, step_penalty=0.0)
    assert reward < 0.0, f"Reward esperado < 0 para imágenes ortogonales, obtenido {reward:.4f}"


def test_step_penalty_reduces_reward():
    img = generate_fiber_image(theta=30.0)
    r_no_penalty = compute_reward(img, img, step_penalty=0.0)
    r_with_penalty = compute_reward(img, img, step_penalty=0.01)
    assert r_with_penalty < r_no_penalty


def test_shape_mismatch_raises():
    img1 = generate_fiber_image(theta=10.0, size=128)
    img2 = generate_fiber_image(theta=10.0, size=64)
    with pytest.raises(ValueError):
        compute_reward(img1, img2)
