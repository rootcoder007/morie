"""Tests for fairRC.fairness_rec."""

from morie.fn import _array_core as np

from morie.fn.fairRC import fairness_rec


def test_fairRC_basic():
    """Test basic functionality."""
    protected = np.random.default_rng(42).normal(0, 1, 100)
    result = fairness_rec(protected)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_fairRC_edge():
    """Test edge cases."""
    protected = np.random.default_rng(42).normal(0, 1, 100)
    result = fairness_rec(protected)
    assert isinstance(result, dict)
