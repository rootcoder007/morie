"""Tests for waicd.waic_diagnostic."""

from morie.fn import _array_core as np
from morie.fn.waicd import waic_diagnostic


def test_waicd_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = waic_diagnostic(log_lik)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_waicd_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = waic_diagnostic(log_lik)
    assert isinstance(result, dict)


def test_waic_and_high_variance_count():
    import math

    import pytest

    L = [[-1.0, -0.5, -3.0], [-1.2, -0.4, -0.5], [-0.8, -0.7, -2.5], [-1.1, -0.6, -1.9]]
    S = 4
    lppd = pw = 0.0
    high = 0
    for i in range(3):
        col = [L[s][i] for s in range(S)]
        m = sum(col) / S
        v = sum((t - m) ** 2 for t in col) / (S - 1)
        lppd += math.log(sum(math.exp(t) for t in col) / S)
        pw += v
        high += v > 0.4
    r = waic_diagnostic(L)
    assert r["estimate"] == pytest.approx(-2 * (lppd - pw), rel=1e-13)
    assert r["n_high_var"] == high == 1
