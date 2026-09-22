"""Tests for agdcon.agd_constrained."""

from morie.fn import _array_core as np

from morie.fn.agdcon import agd_constrained


def test_agdcon_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    y = np.random.default_rng(42).normal(0, 1, 100)
    result = agd_constrained(X, y)
    assert isinstance(result, dict)
    assert "beta" in result
def test_agdcon_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 3))
    y = np.random.default_rng(42).normal(0, 1, 100)
    result = agd_constrained(X, y)
    assert isinstance(result, dict)
