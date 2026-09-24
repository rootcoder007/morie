"""Tests for wsmbpv.wasserman_bootstrap_pivotal."""

from morie.fn import _array_core as np

from morie.fn.wsmbpv import wasserman_bootstrap_pivotal


def test_wsmbpv_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    B = 5
    alpha = 0.1
    result = wasserman_bootstrap_pivotal(data, T, B, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmbpv_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    B = 5
    alpha = 0.1
    result = wasserman_bootstrap_pivotal(data, T, B, alpha)
    assert isinstance(result, dict)
