"""Tests for gaussian_approx_n.gaussian_approx_n."""

from morie.fn import _array_core as np

from morie.fn.gaussian_approx_n import (
    gaussian_approx_n,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_approx_n(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_approx_n(x)
    assert isinstance(result, dict)
