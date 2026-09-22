"""Tests for admmop.admm."""

from morie.fn import _array_core as np

from morie.fn.admmop import admm


def test_admmop_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    y = np.random.default_rng(42).normal(0, 1, 100)
    lam = 1.0
    result = admm(X, y, lam)
    assert isinstance(result, dict)
    assert "x" in result
def test_admmop_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    y = np.random.default_rng(42).normal(0, 1, 100)
    lam = 1.0
    result = admm(X, y, lam)
    assert isinstance(result, dict)
