"""Tests for morie.fn.llreg — Local linear regression."""

import pytest

from morie.fn import _array_core as np
from morie.fn.llreg import llreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 100)
    result = llreg(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "slope", "bandwidth", "n_obs"):
        assert key in result


def test_fits_linear():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 200)
    y = 3 * x + 1 + rng.normal(0, 0.05, 200)
    result = llreg(x, y)
    y_hat = np.asarray(result["y_hat"])
    corr = np.corrcoef(y, y_hat)[0, 1]
    assert corr > 0.9


def test_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        llreg(np.ones(10), np.ones(5))


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 3"):
        llreg(np.ones(2), np.ones(2))


def test_local_linear_fit_recomputed():
    """Weighted least squares of y on (1, x - x0) with Gaussian weights."""
    import math

    x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    y = [1.0, 1.4, 0.9, 0.2, -0.3, 0.1]
    h = 0.8
    r = llreg(x, y, [0.3, 1.9], bandwidth=h)
    for j, x0 in enumerate((0.3, 1.9)):
        w = [math.exp(-0.5 * ((a - x0) / h) ** 2) / math.sqrt(2 * math.pi) for a in x]
        d = [a - x0 for a in x]
        s0, s1, s2 = sum(w), sum(wi * di for wi, di in zip(w, d)), sum(wi * di * di for wi, di in zip(w, d))
        t0 = sum(wi * yi for wi, yi in zip(w, y))
        t1 = sum(wi * di * yi for wi, di, yi in zip(w, d, y))
        det = s0 * s2 - s1 * s1
        assert r["y_hat"][j] == pytest.approx((s2 * t0 - s1 * t1) / det, rel=1e-11)
        assert r["slope"][j] == pytest.approx((s0 * t1 - s1 * t0) / det, rel=1e-11)
