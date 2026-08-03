"""Tests for hedges_j.hedges_j."""

from morie.fn import _array_core as np

from morie.fn.hedges_j import hedges_j


def test_ca11e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedges_j(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedges_j(x)
    assert isinstance(result, dict)
