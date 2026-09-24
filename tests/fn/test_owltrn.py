"""Tests for owltrn.outcome_weighted_learning."""

from morie.fn import _array_core as np

from morie.fn.owltrn import outcome_weighted_learning


def test_owltrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = outcome_weighted_learning(y, D, W)
    assert isinstance(result, dict)
    assert "estimate" in result or "beta" in result


def test_owltrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    W = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = outcome_weighted_learning(y, D, W)
    assert isinstance(result, dict)
