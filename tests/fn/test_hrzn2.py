"""Tests for hrzn2.horowitz_deconvolution."""

import numpy as np
import pytest

from morie.fn.hrzn2 import horowitz_deconvolution


def test_hrzn2_density_estimate_is_a_density():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    y = x + rng.laplace(scale=0.5 / np.sqrt(2), size=600)
    r = horowitz_deconvolution(y, sigma_u=0.5, noise="laplace")
    grid = np.asarray(r["grid"], dtype=float)
    f = np.asarray(r["density"] if "density" in r else r["estimate"], dtype=float)
    mass = float(np.trapezoid(np.clip(f, 0, None), grid))
    assert mass == pytest.approx(1.0, abs=0.25)


def test_hrzn2_recovers_the_center_of_the_signal_density():
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 1.0, 800)
    y = x + rng.laplace(scale=0.4 / np.sqrt(2), size=800)
    r = horowitz_deconvolution(y, sigma_u=0.4, noise="laplace")
    grid = np.asarray(r["grid"], dtype=float)
    f = np.asarray(r["density"] if "density" in r else r["estimate"], dtype=float)
    mode = float(grid[np.argmax(f)])
    assert mode == pytest.approx(2.0, abs=0.5)
