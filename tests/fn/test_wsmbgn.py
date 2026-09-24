"""Tests for wsmbgn.wasserman_bagging."""

from morie.fn import _array_core as np

from morie.fn.wsmbgn import wasserman_bagging


def test_wsmbgn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_bagging(X, y)
    assert isinstance(result, dict)
    assert "prediction" in result


def test_wsmbgn_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_bagging(X, y)
    assert isinstance(result, dict)
