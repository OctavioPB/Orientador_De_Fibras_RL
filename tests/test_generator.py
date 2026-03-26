"""Tests para env/synthetic_generator.py."""

import numpy as np
import pytest

from env.synthetic_generator import generate_fiber_image


def test_output_shape_and_dtype():
    img = generate_fiber_image(theta=45.0)
    assert img.shape == (128, 128), f"Shape inesperado: {img.shape}"
    assert img.dtype == np.uint8, f"dtype inesperado: {img.dtype}"


def test_different_thetas_produce_different_images():
    img0 = generate_fiber_image(theta=0.0)
    img90 = generate_fiber_image(theta=90.0)
    assert not np.array_equal(img0, img90), "Ángulos distintos deben producir imágenes distintas"


def test_no_exception_for_valid_theta_range():
    for theta in np.linspace(0.0, 179.9, 18):
        generate_fiber_image(theta=float(theta))


def test_custom_size():
    img = generate_fiber_image(theta=30.0, size=64)
    assert img.shape == (64, 64)


def test_no_noise_produces_valid_image():
    img = generate_fiber_image(theta=60.0, noise_std=0.0)
    assert img.dtype == np.uint8
    assert img.shape == (128, 128)
