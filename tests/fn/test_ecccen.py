"""Tests for ecccen.eccentricity_centrality."""

from morie.fn import _array_core as np

from morie.fn.ecccen import eccentricity_centrality


def test_ecccen_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = eccentricity_centrality(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ecccen_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = eccentricity_centrality(A)
    assert isinstance(result, dict)
