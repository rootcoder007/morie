"""Tests for cohens_d_population.cohens_d_population."""

from morie.fn import _array_core as np

from morie.fn.cohens_d_population import cohens_d_population


def test_ca8e2_basic():
    """Test basic functionality."""
    mu1, mu2, sigma = 5.0, 3.0, 2.0
    expected = (mu1 - mu2) / sigma
    result = cohens_d_population(mu1, mu2, sigma)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == expected


def test_ca8e2_edge():
    """Test edge cases."""
    mu1, mu2, sigma = 0.0, 0.0, 1.0
    expected = (mu1 - mu2) / sigma
    result = cohens_d_population(mu1, mu2, sigma)
    assert isinstance(result, dict)
    assert result["value"] == expected
