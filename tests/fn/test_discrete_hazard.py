"""Tests for discrete_hazard.discrete_hazard."""

from morie.fn import _array_core as np

from morie.fn.discrete_hazard import discrete_hazard


def test_ghs010_basic():
    """Test basic functionality."""
    p_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = discrete_hazard(p_j, j, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs010_edge():
    """Test edge cases."""
    p_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = discrete_hazard(p_j, j, X)
    assert isinstance(result, dict)
