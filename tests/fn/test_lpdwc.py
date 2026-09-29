"""Tests for lpdwc.log_pointwise_predictive_density."""

from morie.fn import _array_core as np
from morie.fn.lpdwc import log_pointwise_predictive_density


def test_lpdwc_basic():
    """Test basic functionality."""
    logdens = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = log_pointwise_predictive_density(logdens)
    assert isinstance(result, dict)
    assert "lppd" in result


def test_lpdwc_edge():
    """Test edge cases."""
    logdens = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = log_pointwise_predictive_density(logdens)
    assert isinstance(result, dict)


def test_lppd_and_waic_recomputed():
    import math

    import pytest

    L = [[-1.0, -0.5, -2.0], [-1.2, -0.4, -1.5], [-0.8, -0.7, -2.5]]
    S = 3
    lp = [math.log(sum(math.exp(L[s][i]) for s in range(S)) / S) for i in range(3)]
    pv = []
    for i in range(3):
        col = [L[s][i] for s in range(S)]
        m = sum(col) / S
        pv.append(sum((v - m) ** 2 for v in col) / (S - 1))
    r = log_pointwise_predictive_density(L)
    assert r["lppd"] == pytest.approx(sum(lp), rel=1e-13)
    assert r["p_waic"] == pytest.approx(sum(pv), rel=1e-13)
    assert r["waic"] == pytest.approx(-2 * (sum(lp) - sum(pv)), rel=1e-13)
