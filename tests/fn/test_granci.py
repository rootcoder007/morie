"""Tests for granci.granger_causality_info."""

from morie.fn import _array_core as np
import pytest

from morie.fn.granci import granger_causality_info


def test_granci_basic():
    rng = np.random.default_rng(42)
    n = 1500
    x = np.zeros(n); y = np.zeros(n)
    ex = rng.normal(size=n); ey = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + ey[t]
    out = granger_causality_info(x, y, lag=1)
    assert out["mi"] > 0.05  # measured ~0.11 nats
    assert out["p_value"] < 0.01
    assert granger_causality_info(y, x, lag=1)["mi"] < 0.01


def test_granci_edge():
    with pytest.raises(ValueError):
        granger_causality_info([1.0, 2.0], [1.0, 2.0], lag=1)  # too short
    with pytest.raises(ValueError):
        granger_causality_info([1.0] * 20, [1.0] * 25, lag=1)  # length mismatch
