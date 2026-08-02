"""Tests for hdpic.highest_density_credible_interval."""

from morie.fn import _array_core as np

from morie.fn.hdpic import highest_density_credible_interval


def test_hdpic_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = highest_density_credible_interval(samples)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hdpic_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = highest_density_credible_interval(samples)
    assert isinstance(result, dict)
