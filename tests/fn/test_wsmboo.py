"""Tests for wsmboo.wasserman_bootstrap."""

from morie.fn import _array_core as np

from morie.fn.wsmboo import wasserman_bootstrap


def test_wsmboo_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = wasserman_bootstrap(data, T)
    assert isinstance(result, dict)
    assert "value" in result


def test_wsmboo_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0.0, 1.0, 40)
    T = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    result = wasserman_bootstrap(data, T)
    assert isinstance(result, dict)
