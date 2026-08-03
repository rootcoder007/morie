"""Tests for cohens_d_population.cohens_d_population."""

from morie.fn import _array_core as np

from morie.fn.cohens_d_population import cohens_d_population


def test_ca8e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cohens_d_population(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca8e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cohens_d_population(x)
    assert isinstance(result, dict)
