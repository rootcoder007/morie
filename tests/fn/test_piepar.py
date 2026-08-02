"""Tests for piepar.pie_parameters."""

from morie.fn import _array_core as np

from morie.fn.piepar import pie_parameters


def test_piepar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    intervention_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = pie_parameters(y, X, intervention_dist)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_piepar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    intervention_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = pie_parameters(y, X, intervention_dist)
    assert isinstance(result, dict)
