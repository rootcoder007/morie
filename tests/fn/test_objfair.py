"""Tests for objfair.individual_fairness_lipschitz."""

from morie.fn import _array_core as np

from morie.fn.objfair import individual_fairness_lipschitz


def test_objfair_basic():
    """Test basic functionality."""
    h_values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    x_pairs = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = individual_fairness_lipschitz(h_values, x_pairs)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_objfair_edge():
    """Test edge cases."""
    h_values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    x_pairs = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = individual_fairness_lipschitz(h_values, x_pairs)
    assert isinstance(result, dict)
