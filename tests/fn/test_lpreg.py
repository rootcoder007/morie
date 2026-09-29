"""Tests for morie.fn.lpreg — Local polynomial regression."""

import pytest

from morie.fn import _array_core as np
from morie.fn.lpreg import lpreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = x**2 + rng.normal(0, 0.1, 100)
    result = lpreg(x, y, degree=2)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "coefficients", "degree", "bandwidth", "n_obs"):
        assert key in result


def test_degree_zero_is_nw():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = x + rng.normal(0, 0.1, 50)
    result = lpreg(x, y, degree=0)
    assert result["degree"] == 0


def test_negative_degree_raises():
    with pytest.raises(ValueError, match="degree"):
        lpreg(np.ones(10), np.ones(10), degree=-1)


def test_fits_quadratic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 200)
    y = x**2 + rng.normal(0, 0.05, 200)
    result = lpreg(x, y, degree=2)
    y_hat = np.asarray(result["y_hat"])
    assert np.corrcoef(y, y_hat)[0, 1] > 0.85


def test_local_quadratic_coefficients_solve_the_weighted_normal_equations():
    import math

    x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    y = [1.0, 1.4, 0.9, 0.2, -0.3, 0.1, 0.8]
    h, x0 = 0.9, 1.2
    r = lpreg(x, y, [x0], degree=2, bandwidth=h)
    b = r["coefficients"][0]
    w = [math.exp(-0.5 * ((a - x0) / h) ** 2) / math.sqrt(2 * math.pi) for a in x]
    d = [a - x0 for a in x]
    for p in range(3):
        lhs = sum(w[i] * d[i] ** p * (y[i] - sum(b[q] * d[i] ** q for q in range(3))) for i in range(7))
        assert lhs == pytest.approx(0.0, abs=1e-10)
    assert r["y_hat"][0] == pytest.approx(b[0], rel=1e-15)
