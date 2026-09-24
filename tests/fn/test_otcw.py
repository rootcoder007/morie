"""Tests for otcw.ot_cyclical_weight."""

from morie.fn import _array_core as np

from morie.fn.otcw import ot_cyclical_weight


def test_otcw_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Cost = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    perm = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ot_cyclical_weight(X, Y, Cost, perm)
    assert isinstance(result, dict)
    assert "estimate" in result or "is_cm" in result


def test_otcw_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Cost = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    perm = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ot_cyclical_weight(X, Y, Cost, perm)
    assert isinstance(result, dict)
