"""Tests for gpsps.gp_spectral_mixture."""

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gpsps import gp_spectral_mixture


def _sm_kernel(tau, w, v, m):
    return sum(
        w_q * math.exp(-2.0 * math.pi ** 2 * tau ** 2 * v_q) * math.cos(2.0 * math.pi * tau * m_q)
        for w_q, v_q, m_q in zip(w, v, m)
    )


def test_gpsps_basic():
    """Test basic functionality against the documented spectral-mixture formula."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, 100)
    y = rng.normal(0.0, 1.0, 100)
    X_test = rng.normal(0.0, 1.0, 30)

    q = 3
    w = [0.4, 0.3, 0.3]
    v = [0.5, 0.25, 0.75]
    m = [0.2, 0.5, 1.0]
    noise = 0.01

    result = gp_spectral_mixture(
        X, y, X_test, Q=q, weights=w, variances=v, means=m, noise=noise
    )

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert "variance" in result
    assert "loglik" in result
    assert "n" in result
    assert "method" in result
    assert "k_zero" in result

    assert result["n"] == 100
    assert len(result["mean"]) == 30
    assert len(result["variance"]) == 30
    assert result["estimate"] == result["mean"][0]

    expected_mean = []
    expected_var = []
    k0 = _sm_kernel(0.0, w, v, m)
    expected_mean.append(result["mean"][0])
    expected_var.append(k0)

    assert result["k_zero"] == k0
    assert all(v >= 0.0 for v in result["variance"])

    one = gp_spectral_mixture(X, y)
    assert isinstance(one, dict)
    assert len(one["mean"]) == 100
    assert len(one["variance"]) == 100
    assert one["estimate"] == one["mean"][0]


def test_gpsps_edge():
    """Test edge cases: Q=1 and default hyperparameters produce consistent output."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 10)
    y = np.random.default_rng(43).normal(0.0, 1.0, 10)

    result = gp_spectral_mixture(X, y, Q=1)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert "variance" in result
    assert "loglik" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 10
    assert len(result["mean"]) == 10
    assert len(result["variance"]) == 10

    w = [1.0]
    v = [1.0 / (2.0 * math.pi) ** 2]
    m = [0.1]
    expected_k0 = _sm_kernel(0.0, w, v, m)
    assert result["k_zero"] == expected_k0
    assert all(v_ >= 0.0 for v_ in result["variance"])
