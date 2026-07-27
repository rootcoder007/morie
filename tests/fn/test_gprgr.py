"""Tests for gprgr.gaussian_process_regression."""

import numpy as np
import pytest

from morie.fn.gprgr import gaussian_process_regression


def test_gprgr_interpolates_training_points_with_small_noise():
    """With tiny noise the GP posterior mean passes through the data and
    the posterior variance collapses there (Rasmussen & Williams 2006,
    eq. 2.19)."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.sin(x)
    r = gaussian_process_regression(x, y, x_test=x, noise_var=1e-8)
    np.testing.assert_allclose(np.asarray(r["mean"], dtype=float), y, atol=1e-3)
    assert float(np.max(np.asarray(r["var"], dtype=float))) < 1e-3


def test_gprgr_uncertainty_grows_away_from_the_data():
    x = np.linspace(0, 1, 8)
    y = np.cos(3 * x)
    r = gaussian_process_regression(x, y, x_test=np.array([0.5, 3.0]), noise_var=1e-6)
    v = np.asarray(r["var"], dtype=float)
    assert v[1] > 10 * v[0]


def test_gprgr_smooths_noisy_data_toward_the_truth():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2 * np.pi, 60)
    y = np.sin(x) + 0.2 * rng.standard_normal(60)
    r = gaussian_process_regression(x, y, x_test=x, noise_var=0.04)
    err = np.mean((np.asarray(r["mean"], dtype=float) - np.sin(x)) ** 2)
    assert err < np.mean((y - np.sin(x)) ** 2)  # beats the raw noise
