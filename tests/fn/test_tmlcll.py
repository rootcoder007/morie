"""Tests for tmlcll.tmle_cross_lagged."""

from morie.fn import _array_core as np

from morie.fn.tmlcll import tmle_cross_lagged


def test_tmlcll_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_cross_lagged(y, D, X, time)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlcll_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_cross_lagged(y, D, X, time)
    assert isinstance(result, dict)
